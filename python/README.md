# Python Implementation for Depth to Bedrock Prediction

This directory contains a Python-based implementation of the depth to bedrock (DTB) prediction workflow, serving as an alternative to the R-based implementation in the main repository.

## Overview

The Python implementation provides equivalent functionality to the R workflow, including:

- **Data preprocessing**: Loading and preparing geological picks and predictor variables
- **Spatial feature engineering**: Computing spatial lags, Euclidean distance fields, and neighbor-based features
- **Machine learning**: Random Forest and XGBoost models with hyperparameter tuning
- **Spatial cross-validation**: Custom CV strategies that account for spatial autocorrelation
- **Prediction**: Making DTB predictions on raster data

## Installation

### Using pip

```bash
cd python
pip install -r requirements.txt
```

### Using conda

```bash
cd python
conda env create -f environment.yml
conda activate depth-to-bedrock
```

### Using pyproject.toml (recommended for development)

```bash
cd python
pip install -e .
```

To install with optional dependencies:

```bash
pip install -e ".[all]"
```

## Requirements

### Core Dependencies

- Python >= 3.8
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- Geopandas >= 0.10.0
- Rasterio >= 1.2.0
- Scikit-learn >= 1.0.0
- XGBoost >= 1.5.0

### System Dependencies

For geospatial operations, you'll need GDAL installed on your system:

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev
```

**macOS:**
```bash
brew install gdal
```

**Windows:**
Download GDAL from [OSGeo4W](https://www.osgeo.org/projects/osgeo4w/) or use conda.

## Project Structure

```
python/
├── functions/              # Core function modules
│   ├── __init__.py
│   ├── dtb.py             # DTB prediction functions
│   ├── preprocessing.py   # Data loading and preparation
│   ├── spatial.py         # Spatial feature engineering
│   └── resampling.py      # Cross-validation utilities
├── scripts/               # Workflow scripts
│   ├── 03_training_data.py
│   ├── 07_provincial_model.py
│   └── predict.py
├── notebooks/             # Jupyter notebooks (for interactive analysis)
├── requirements.txt       # Pip dependencies
├── environment.yml        # Conda environment
├── pyproject.toml         # Modern Python packaging
└── README.md             # This file
```

## Workflow

The Python workflow follows the same steps as the R implementation:

### 1. Training Data Preparation

```bash
python scripts/03_training_data.py
```

This script:
- Loads geological picks and NLP-predicted picks
- Combines pick datasets (thinning overlapping points)
- Extracts predictor values from rasters at pick locations
- Saves training data to `data/processed/training-data.parquet`

### 2. Model Training

```bash
python scripts/07_provincial_model.py
```

This script:
- Loads training data
- Adds spatial lag features
- Performs hyperparameter tuning using spatial cross-validation
- Trains final Random Forest model
- Saves model, preprocessor, and evaluation metrics

### 3. Making Predictions

```bash
python scripts/predict.py \
  --model models/model-dtb-prov.joblib \
  --preprocessor models/preprocessor-dtb-prov.joblib \
  --raster data/processed/predictors.tif \
  --output outputs/dtb-rf-prov-python.tif
```

## Key Features

### Spatial Feature Engineering

The implementation includes several spatial feature engineering methods:

```python
from functions.spatial import (
    compute_euclidean_distance_fields,
    compute_spatial_lag,
    compute_spatial_neighbors_features,
    SpatialFeatureEngineer
)

# Add spatial lag features
X_with_lags = compute_multiple_spatial_lags(
    X, y, 
    coord_cols=('xcoords', 'ycoords'),
    neighbor_counts=[5, 10, 15]
)

# Or use the transformer
spatial_engineer = SpatialFeatureEngineer(method='lags')
X_transformed = spatial_engineer.fit_transform(X, y)
```

### Spatial Cross-Validation

Custom spatial CV that accounts for spatial autocorrelation:

```python
from functions.resampling import SpatialKFold, nested_cross_validate

# Spatial K-Fold CV
spatial_cv = SpatialKFold(n_splits=10, random_state=1)

# Nested CV for hyperparameter tuning
results = nested_cross_validate(
    X, y, model, param_grid,
    outer_cv=10, inner_cv=3
)
```

### Preprocessing Pipeline

```python
from functions.dtb import DTBPreprocessor

# Create preprocessor
preprocessor = DTBPreprocessor(
    include_coords=False,
    spatial_method='lags'
)

# Fit and transform
X_processed = preprocessor.fit_transform(X)
```

## Usage Examples

### Training a Model

```python
import pandas as pd
from functions.dtb import DTBPreprocessor, train_random_forest
from functions.spatial import compute_multiple_spatial_lags

# Load data
data = pd.read_parquet('data/processed/training-data.parquet')

# Prepare features
X = data.drop(['bedrock_dep', 'id'], axis=1)
y = data['bedrock_dep']

# Add spatial features
X_spatial = compute_multiple_spatial_lags(X, y)

# Preprocess
preprocessor = DTBPreprocessor(include_coords=False)
X_processed = preprocessor.fit_transform(X_spatial)

# Train model
model = train_random_forest(X_processed, y, n_estimators=500)
```

### Making Predictions

```python
import joblib
from functions.dtb import predict_dtb

# Load model and preprocessor
model = joblib.load('models/model-dtb-prov.joblib')
preprocessor = joblib.load('models/preprocessor-dtb-prov.joblib')

# Prepare new data
X_new = ...  # Your new data
X_processed = preprocessor.transform(X_new)

# Predict
predictions = predict_dtb(model, X_processed)
```

## Differences from R Implementation

While the Python implementation provides equivalent functionality, there are some differences:

1. **Data formats**: Python uses Parquet/GeoPackage instead of RDS files
2. **Libraries**: Uses scikit-learn instead of tidymodels, geopandas instead of sf
3. **Spatial features**: Implements similar but not identical spatial feature engineering
4. **Performance**: May have different performance characteristics depending on the operation

## Testing

To run tests (once implemented):

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please ensure:

- Code follows PEP 8 style guidelines (use `black` formatter)
- Functions include docstrings with parameter descriptions
- New features include tests
- Updates to README for significant changes

## Citation

If you use this implementation, please cite the original PLOS ONE article:

> Evaluating spatially enabled machine learning approaches for depth to bedrock mapping.
> PLOS ONE. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296881

## License

This implementation follows the same license as the main repository. See the LICENSE file in the repository root.

## Support

For questions or issues:
- Open an issue on GitHub
- Check the main repository README for contact information
- Refer to the R implementation for methodological details

## Roadmap

Future enhancements:
- [ ] NLP module for litholog classification (Python equivalent of 02-nlp.qmd)
- [ ] Data download and preprocessing (Python equivalent of 01-grids.qmd)
- [ ] IDW and Kriging interpolation methods (equivalents of 05-idw.R and 06-kriging.R)
- [ ] Model comparison and experiments (equivalent of 04-experiments.R)
- [ ] Jupyter notebooks for interactive analysis
- [ ] Comprehensive test suite
- [ ] Benchmarking against R implementation
- [ ] GPU acceleration options
- [ ] Additional model types (neural networks, ensemble methods)
