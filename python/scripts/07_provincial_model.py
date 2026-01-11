"""
Provincial model training and prediction script for DTB.

This script:
1. Loads training data
2. Sets up preprocessing and spatial feature engineering
3. Trains a Random Forest model with hyperparameter tuning
4. Makes provincial-scale DTB predictions
5. Saves the model and predictions

Equivalent to R/07-provincial-model.R
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import rasterio
from rasterio.transform import from_bounds

# Add functions to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from functions.dtb import (
    DTBPreprocessor,
    train_random_forest,
    tune_random_forest,
    predict_dtb,
    evaluate_model,
    save_model
)
from functions.spatial import (
    compute_multiple_spatial_lags
)
from functions.resampling import (
    SpatialKFold,
    get_cv_scores
)


def main():
    """Main function for provincial DTB model."""
    
    print("="*60)
    print("Provincial Depth to Bedrock Model Training")
    print("="*60)
    
    # Define paths
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    models_dir = Path(__file__).parent.parent.parent / "models"
    outputs_dir = Path(__file__).parent.parent.parent / "outputs"
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    training_data_file = processed_dir / "training-data.parquet"
    
    if not training_data_file.exists():
        print(f"Error: {training_data_file} not found.")
        print("Please run 03_training_data.py first.")
        return
    
    print(f"\nLoading training data from {training_data_file}...")
    data = pd.read_parquet(training_data_file)
    print(f"Loaded {len(data)} training samples")
    print(f"Features: {data.shape[1]} columns")
    
    # Check for required columns
    required_cols = ['xcoords', 'ycoords', 'bedrock_dep']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Remove rows with missing target
    data = data.dropna(subset=['bedrock_dep'])
    print(f"After removing missing targets: {len(data)} samples")
    
    # Prepare features and target
    target_col = 'bedrock_dep'
    id_col = 'id' if 'id' in data.columns else None
    
    # Remove id column from features
    feature_cols = [col for col in data.columns if col not in [target_col, id_col]]
    
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    print(f"\nTarget variable (bedrock_dep) statistics:")
    print(y.describe())
    
    # Add spatial lag features
    print("\nAdding spatial lag features...")
    X_with_lags = compute_multiple_spatial_lags(
        X, y,
        coord_cols=('xcoords', 'ycoords'),
        neighbor_counts=[5, 10, 15],
        weight_func='gaussian'
    )
    print(f"Features after adding spatial lags: {X_with_lags.shape[1]}")
    
    # Preprocess features
    print("\nPreprocessing features...")
    preprocessor = DTBPreprocessor(include_coords=False)
    X_processed = preprocessor.fit_transform(X_with_lags)
    
    print(f"Final feature matrix shape: {X_processed.shape}")
    print(f"Features: {list(X_processed.columns)}")
    
    # Check for any remaining missing values
    missing_counts = X_processed.isnull().sum()
    if missing_counts.sum() > 0:
        print("\nWarning: Missing values found:")
        print(missing_counts[missing_counts > 0])
        # Fill any remaining missing values
        X_processed = X_processed.fillna(X_processed.mean())
    
    # Hyperparameter tuning
    print("\n" + "="*60)
    print("Hyperparameter Tuning")
    print("="*60)
    
    param_grid = {
        'max_features': [5, 15, 25],
        'min_samples_split': [2, 5, 10]
    }
    
    print(f"Parameter grid: {param_grid}")
    print("Running grid search with 3-fold CV...")
    
    best_model, best_params = tune_random_forest(
        X_processed, y,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        random_state=1234
    )
    
    print(f"\nBest parameters: {best_params}")
    
    # Cross-validation evaluation
    print("\n" + "="*60)
    print("Model Evaluation (10-Fold Spatial CV)")
    print("="*60)
    
    spatial_cv = SpatialKFold(
        n_splits=10,
        coord_cols=('xcoords', 'ycoords'),
        random_state=1
    )
    
    cv_results = get_cv_scores(
        list(spatial_cv.split(X_processed, y)),
        X_processed,
        y,
        best_model
    )
    
    print("\nCross-validation results:")
    print(cv_results)
    print(f"\nMean RMSE: {cv_results['rmse'].mean():.2f} ± {cv_results['rmse'].std():.2f} m")
    print(f"Mean MAE: {cv_results['mae'].mean():.2f} ± {cv_results['mae'].std():.2f} m")
    print(f"Mean R²: {cv_results['r2'].mean():.3f} ± {cv_results['r2'].std():.3f}")
    
    # Save CV results
    cv_results_file = models_dir / "cross-validation-dtb-prov.csv"
    cv_results.to_csv(cv_results_file, index=False)
    print(f"\nSaved CV results to {cv_results_file}")
    
    # Train final model on all data
    print("\n" + "="*60)
    print("Training Final Model")
    print("="*60)
    
    print("Training on full dataset...")
    final_model = train_random_forest(
        X_processed, y,
        n_estimators=500,
        max_features=best_params['max_features'],
        min_samples_split=best_params['min_samples_split'],
        n_jobs=-1,
        random_state=1234
    )
    
    # Feature importance
    print("\nTop 10 most important features:")
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Save feature importance
    importance_file = models_dir / "feature-importance-dtb-prov.csv"
    feature_importance.to_csv(importance_file, index=False)
    print(f"\nSaved feature importance to {importance_file}")
    
    # Save model
    model_file = models_dir / "model-dtb-prov.joblib"
    save_model(final_model, str(model_file))
    print(f"\nSaved model to {model_file}")
    
    # Save preprocessor
    preprocessor_file = models_dir / "preprocessor-dtb-prov.joblib"
    joblib.dump(preprocessor, str(preprocessor_file))
    print(f"Saved preprocessor to {preprocessor_file}")
    
    # Make predictions on training data (for validation)
    print("\n" + "="*60)
    print("Training Set Performance")
    print("="*60)
    
    y_pred = predict_dtb(final_model, X_processed)
    train_metrics = evaluate_model(y.values, y_pred)
    
    print(f"Training RMSE: {train_metrics['rmse']:.2f} m")
    print(f"Training MAE: {train_metrics['mae']:.2f} m")
    print(f"Training R²: {train_metrics['r2']:.3f}")
    
    print("\n" + "="*60)
    print("Model Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {model_file}")
    print(f"CV results saved to: {cv_results_file}")
    print(f"Feature importance saved to: {importance_file}")
    print("\nTo make predictions on new data, use the saved model and preprocessor.")


if __name__ == "__main__":
    main()
