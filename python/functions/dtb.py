"""
Core DTB prediction functions and utilities.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import joblib


def summarize_data_type(picks: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the number of picks with specific data types.
    
    Includes 'Shotholes' as boreholes and 'Contour' as 'Pseudo-observation'.
    
    Parameters
    ----------
    picks : pd.DataFrame
        DataFrame of picks data with a 'data_type' column.
    
    Returns
    -------
    pd.DataFrame
        Summarized data with the number of observations per data type.
    """
    # Recode data types
    picks_copy = picks.copy()
    picks_copy['data_type'] = picks_copy['data_type'].replace({
        'Contour': 'Pseudo-observation'
    })
    
    # Count occurrences
    cnts = picks_copy.groupby('data_type').size().reset_index(name='n')
    
    return cnts


def summarize_borehole_sources(picks: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the proportion of borehole sources in the picks data.
    
    Parameters
    ----------
    picks : pd.DataFrame
        DataFrame of picks data with 'data_type' and 'source_type' columns.
    
    Returns
    -------
    pd.DataFrame
        Summarized data with the number and proportion of borehole data sources.
    """
    # Filter to boreholes only
    boreholes = picks[picks['data_type'] == 'Borehole']
    
    # Count and calculate percentages
    source_perc = boreholes.groupby('source_type').size().reset_index(name='n')
    source_perc['pct'] = (source_perc['n'] / source_perc['n'].sum() * 100).round(3)
    
    return source_perc


class DTBPreprocessor:
    """
    Preprocessing pipeline for depth to bedrock prediction.
    
    Handles aspect decomposition, missing value imputation, and feature engineering.
    """
    
    def __init__(self, include_coords: bool = True, 
                 spatial_method: Optional[str] = None):
        """
        Initialize the preprocessor.
        
        Parameters
        ----------
        include_coords : bool, default=True
            Whether to include x and y coordinates as features.
        spatial_method : str, optional
            Spatial feature engineering method: 'edfs', 'lags', 'neighbors', or None.
        """
        self.include_coords = include_coords
        self.spatial_method = spatial_method
        self.mean_values_ = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the preprocessor to the training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series, optional
            Training target (not used, for sklearn compatibility).
            
        Returns
        -------
        self
        """
        # Store mean values for imputation
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.mean_values_ = X[numeric_cols].mean().to_dict()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.
            
        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        X_transformed = X.copy()
        
        # Decompose aspect into easterness and northerness
        if 'aspect' in X_transformed.columns:
            # Replace missing aspect with 0
            X_transformed['aspect'] = X_transformed['aspect'].fillna(0)
            
            # Convert to radians and decompose
            aspect_rad = X_transformed['aspect'] * (np.pi / 180)
            X_transformed['easternness'] = np.cos(aspect_rad)
            X_transformed['northerness'] = np.sin(aspect_rad)
            
            # Remove original aspect
            X_transformed = X_transformed.drop('aspect', axis=1)
        
        # Impute missing values with mean
        for col, mean_val in self.mean_values_.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].fillna(mean_val)
        
        # Remove coordinates if specified
        if not self.include_coords:
            coord_cols = ['xcoords', 'ycoords']
            X_transformed = X_transformed.drop(
                [c for c in coord_cols if c in X_transformed.columns], 
                axis=1
            )
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to fit and transform.
        y : pd.Series, optional
            Training target (not used).
            
        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        return self.fit(X, y).transform(X)


def train_random_forest(X: pd.DataFrame, y: pd.Series, 
                        n_estimators: int = 500,
                        max_features: Optional[int] = None,
                        min_samples_split: int = 2,
                        n_jobs: int = -1,
                        random_state: int = 1234) -> RandomForestRegressor:
    """
    Train a Random Forest model for DTB prediction.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training target (bedrock depth).
    n_estimators : int, default=500
        Number of trees in the forest.
    max_features : int, optional
        Number of features to consider for splits. If None, uses sqrt(n_features).
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all processors.
    random_state : int, default=1234
        Random seed for reproducibility.
        
    Returns
    -------
    RandomForestRegressor
        Trained Random Forest model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_split=min_samples_split,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    model.fit(X, y)
    
    return model


def tune_random_forest(X: pd.DataFrame, y: pd.Series,
                      param_grid: Optional[Dict[str, List]] = None,
                      cv: int = 3,
                      n_jobs: int = -1,
                      random_state: int = 1234) -> Tuple[RandomForestRegressor, Dict]:
    """
    Tune Random Forest hyperparameters using grid search.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training target.
    param_grid : dict, optional
        Parameter grid for grid search. If None, uses default grid.
    cv : int, default=3
        Number of cross-validation folds.
    n_jobs : int, default=-1
        Number of parallel jobs.
    random_state : int, default=1234
        Random seed.
        
    Returns
    -------
    tuple
        (best_model, best_params)
    """
    if param_grid is None:
        param_grid = {
            'max_features': [5, 15, 25, 38],
            'min_samples_split': [2, 5, 10]
        }
    
    base_model = RandomForestRegressor(
        n_estimators=500,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_


def predict_dtb(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Make DTB predictions using a trained model.
    
    Parameters
    ----------
    model : sklearn model
        Trained model.
    X : pd.DataFrame
        Features for prediction.
        
    Returns
    -------
    np.ndarray
        Predicted depths to bedrock.
    """
    return model.predict(X)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    dict
        Dictionary with RMSE, MAE, and R² metrics.
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Parameters
    ----------
    model : sklearn model
        Model to save.
    filepath : str
        Path where to save the model.
    """
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to the saved model.
        
    Returns
    -------
    sklearn model
        Loaded model.
    """
    return joblib.load(filepath)
