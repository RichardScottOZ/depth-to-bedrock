"""
Cross-validation and resampling utilities for spatial data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Iterator, Optional
from sklearn.model_selection import KFold
from sklearn.base import clone


class SpatialKFold:
    """
    Spatial K-Fold cross-validation using blocking to reduce spatial autocorrelation.
    
    Divides the study area into spatial blocks and ensures that samples from the same
    block are not split between training and validation sets.
    """
    
    def __init__(self, n_splits: int = 5, 
                 coord_cols: Tuple[str, str] = ('xcoords', 'ycoords'),
                 random_state: Optional[int] = None):
        """
        Initialize Spatial K-Fold.
        
        Parameters
        ----------
        n_splits : int, default=5
            Number of folds.
        coord_cols : tuple of str, default=('xcoords', 'ycoords')
            Names of coordinate columns.
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.coord_cols = coord_cols
        self.random_state = random_state
        
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
             groups: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features with coordinate columns.
        y : pd.Series, optional
            Target variable (not used, for sklearn compatibility).
        groups : np.ndarray, optional
            Group labels (not used).
            
        Yields
        ------
        train : np.ndarray
            Training set indices for this split.
        test : np.ndarray
            Test set indices for this split.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Get coordinates
        x_col, y_col = self.coord_cols
        x = X[x_col].values
        y_coord = X[y_col].values
        
        # Create spatial blocks
        # Divide area into roughly equal grid
        n_blocks_per_dim = int(np.ceil(np.sqrt(self.n_splits)))
        
        x_bins = np.linspace(x.min(), x.max(), n_blocks_per_dim + 1)
        y_bins = np.linspace(y_coord.min(), y_coord.max(), n_blocks_per_dim + 1)
        
        x_block = np.digitize(x, x_bins) - 1
        y_block = np.digitize(y_coord, y_bins) - 1
        
        # Create block IDs
        block_ids = x_block * n_blocks_per_dim + y_block
        
        # Randomly assign blocks to folds
        unique_blocks = np.unique(block_ids)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        np.random.shuffle(unique_blocks)
        
        # Split blocks into folds
        fold_blocks = np.array_split(unique_blocks, self.n_splits)
        
        # Generate splits
        for fold_idx in range(self.n_splits):
            test_blocks = fold_blocks[fold_idx]
            test_mask = np.isin(block_ids, test_blocks)
            
            train_indices = indices[~test_mask]
            test_indices = indices[test_mask]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.n_splits


def nested_cross_validate(X: pd.DataFrame, 
                          y: pd.Series,
                          model,
                          param_grid: dict,
                          outer_cv: int = 10,
                          inner_cv: int = 3,
                          scoring: str = 'neg_mean_squared_error',
                          coord_cols: Tuple[str, str] = ('xcoords', 'ycoords'),
                          random_state: int = 1) -> pd.DataFrame:
    """
    Perform nested cross-validation for hyperparameter tuning and model evaluation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Target variable.
    model : sklearn estimator
        Base model to tune.
    param_grid : dict
        Parameter grid for tuning.
    outer_cv : int, default=10
        Number of outer cross-validation folds.
    inner_cv : int, default=3
        Number of inner cross-validation folds for hyperparameter tuning.
    scoring : str, default='neg_mean_squared_error'
        Scoring metric.
    coord_cols : tuple of str, default=('xcoords', 'ycoords')
        Coordinate column names.
    random_state : int, default=1
        Random seed.
        
    Returns
    -------
    pd.DataFrame
        Results with predictions and best parameters for each outer fold.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Setup spatial CV
    outer_splitter = SpatialKFold(n_splits=outer_cv, coord_cols=coord_cols, 
                                  random_state=random_state)
    
    results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splitter.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner CV for hyperparameter tuning
        inner_splitter = SpatialKFold(n_splits=inner_cv, coord_cols=coord_cols,
                                      random_state=random_state + fold_idx)
        
        grid_search = GridSearchCV(
            clone(model),
            param_grid,
            cv=list(inner_splitter.split(X_train, y_train)),
            scoring=scoring,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Predict on outer test fold
        y_pred = grid_search.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'fold': fold_idx + 1,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'best_params': grid_search.best_params_,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        })
    
    return pd.DataFrame(results)


def stratified_spatial_cv(X: pd.DataFrame,
                         y: pd.Series,
                         n_splits: int = 10,
                         n_strata: int = 5,
                         coord_cols: Tuple[str, str] = ('xcoords', 'ycoords'),
                         random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified spatial cross-validation splits.
    
    Combines spatial blocking with stratification based on the target variable.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features with coordinate columns.
    y : pd.Series
        Target variable for stratification.
    n_splits : int, default=10
        Number of folds.
    n_strata : int, default=5
        Number of strata based on target quantiles.
    coord_cols : tuple of str, default=('xcoords', 'ycoords')
        Coordinate column names.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    list of tuple
        List of (train_indices, test_indices) for each fold.
    """
    # Create strata based on quantiles
    strata = pd.qcut(y, q=n_strata, labels=False, duplicates='drop')
    
    # Perform spatial CV within each stratum
    all_splits = [[] for _ in range(n_splits)]
    
    for stratum_id in range(strata.max() + 1):
        stratum_mask = strata == stratum_id
        stratum_indices = np.where(stratum_mask)[0]
        
        if len(stratum_indices) < n_splits:
            # If stratum too small, just distribute randomly
            if random_state is not None:
                np.random.seed(random_state + stratum_id)
            np.random.shuffle(stratum_indices)
            
            for fold_idx in range(n_splits):
                fold_size = len(stratum_indices) // n_splits
                start_idx = fold_idx * fold_size
                end_idx = start_idx + fold_size if fold_idx < n_splits - 1 else len(stratum_indices)
                all_splits[fold_idx].append(stratum_indices[start_idx:end_idx])
        else:
            # Perform spatial CV for this stratum
            X_stratum = X.iloc[stratum_indices]
            
            spatial_cv = SpatialKFold(n_splits=n_splits, coord_cols=coord_cols,
                                     random_state=random_state)
            
            for fold_idx, (_, test_idx_local) in enumerate(spatial_cv.split(X_stratum)):
                # Convert local indices to global indices
                test_idx_global = stratum_indices[test_idx_local]
                all_splits[fold_idx].append(test_idx_global)
    
    # Combine strata for each fold
    final_splits = []
    all_indices = np.arange(len(X))
    
    for fold_idx in range(n_splits):
        test_indices = np.concatenate(all_splits[fold_idx])
        train_indices = np.setdiff1d(all_indices, test_indices)
        final_splits.append((train_indices, test_indices))
    
    return final_splits


def get_cv_scores(splits: List[Tuple[np.ndarray, np.ndarray]],
                 X: pd.DataFrame,
                 y: pd.Series,
                 model) -> pd.DataFrame:
    """
    Evaluate model using provided cross-validation splits.
    
    Parameters
    ----------
    splits : list of tuple
        List of (train_indices, test_indices) tuples.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target variable.
    model : sklearn estimator
        Model to evaluate.
        
    Returns
    -------
    pd.DataFrame
        Cross-validation scores for each fold.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Clone and fit model
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        # Predict
        y_pred = model_clone.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'fold': fold_idx + 1,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        })
    
    return pd.DataFrame(results)
