# Python vs R Implementation Comparison

This document provides a comparison between the R and Python implementations for the Depth to Bedrock prediction workflow.

## Feature Comparison

| Feature | R Implementation | Python Implementation |
|---------|-----------------|---------------------|
| **Core ML Library** | tidymodels | scikit-learn |
| **Random Forest** | ranger | sklearn.ensemble.RandomForestRegressor |
| **XGBoost** | xgboost (via parsnip) | xgboost.XGBRegressor |
| **Spatial Data** | sf, terra | geopandas, rasterio |
| **Data Manipulation** | tidyverse (dplyr, tidyr) | pandas |
| **Recipe System** | recipes | Custom DTBPreprocessor class |
| **Cross-validation** | rsample (nested_cv) | Custom SpatialKFold class |
| **Data Format** | RDS, GeoPackage | Parquet, GeoPackage, CSV |
| **Notebooks** | Quarto (.qmd) | Jupyter (.ipynb) |

## Implemented Components

### R Implementation (Original)
- ✅ 01-grids.qmd - Data download and terrain analysis
- ✅ 02-nlp.qmd - Litholog NLP classification
- ✅ 03-training-data.R - Training data preparation
- ✅ 04-experiments.R - Model evaluation
- ✅ 05-idw.R - IDW interpolation
- ✅ 06-kriging.R - Kriging interpolation
- ✅ 07-provincial-model.R - Final model training
- ✅ 08-analysis-results.qmd - Results visualization

### Python Implementation (New)
- ⏳ 01-grids - Not yet implemented (can use R output)
- ⏳ 02-nlp - Not yet implemented (can use R output)
- ✅ 03_training_data.py - Training data preparation
- ⏳ 04-experiments - Planned
- ⏳ 05-idw - Planned
- ⏳ 06-kriging - Planned (can use pykrige)
- ✅ 07_provincial_model.py - Final model training
- ✅ predict.py - Prediction utility

### Python Function Modules
- ✅ `dtb.py` - Core DTB functions (preprocessing, training, evaluation)
- ✅ `preprocessing.py` - Data loading and preparation utilities
- ✅ `spatial.py` - Spatial feature engineering (EDFs, lags, neighbors)
- ✅ `resampling.py` - Spatial cross-validation utilities

## Key Equivalences

### Data Preprocessing

**R (recipe_dtb):**
```r
recipe_dtb <- function(data) {
  data |>
    select(-id) |>
    recipe(bedrock_dep ~ .) |>
    step_mutate(
      aspect = if_else(is.na(aspect), 0, aspect),
      easternness = cos(aspect * (pi / 180)),
      northerness = sin(aspect * (pi / 180))
    ) |>
    step_rm(all_of("aspect")) |>
    step_impute_mean(all_numeric_predictors())
}
```

**Python (DTBPreprocessor):**
```python
preprocessor = DTBPreprocessor(include_coords=False)
X_processed = preprocessor.fit_transform(X)
# Handles aspect decomposition, imputation automatically
```

### Spatial Feature Engineering

**R (recipe_lags):**
```r
recipe_lags <- function(data) {
  recipe_dtb(data) |>
    step_normalize(all_numeric_predictors()) |>
    step_spatial_lag(
      all_of(c("xcoords", "ycoords")),
      outcome = "bedrock_dep",
      weight_func = "gaussian",
      neighbors = 5
    ) |>
    # ... more lags
}
```

**Python:**
```python
from functions.spatial import compute_multiple_spatial_lags

X_with_lags = compute_multiple_spatial_lags(
    X, y,
    coord_cols=('xcoords', 'ycoords'),
    neighbor_counts=[5, 10, 15],
    weight_func='gaussian'
)
```

### Model Training

**R:**
```r
regr <- rand_forest(trees = 500L, min_n = tune(), mtry = tune()) |>
  set_mode("regression") |>
  set_engine("ranger", seed = 1234, splitrule = "extratrees")

wflow <- workflow() |>
  add_recipe(rec) |>
  add_model(regr)
```

**Python:**
```python
from functions.dtb import tune_random_forest

best_model, best_params = tune_random_forest(
    X, y,
    param_grid={'max_features': [5, 15, 25], 
                'min_samples_split': [2, 5, 10]},
    cv=3,
    random_state=1234
)
```

### Cross-validation

**R:**
```r
resamples <- nested_cv(
  data,
  outside = vfold_cv(v = 10L, strata = "bedrock_dep"),
  inside = vfold_cv(v = 3L, strata = "bedrock_dep")
)
```

**Python:**
```python
from functions.resampling import SpatialKFold, nested_cross_validate

results = nested_cross_validate(
    X, y, model, param_grid,
    outer_cv=10, inner_cv=3,
    random_state=1
)
```

## Usage Workflows

### R Workflow
```bash
# 1. Prepare data
Rscript R/03-training-data.R

# 2. Train model
Rscript R/07-provincial-model.R
```

### Python Workflow
```bash
# 1. Install dependencies
pip install -r python/requirements.txt

# 2. Prepare data
python python/scripts/03_training_data.py

# 3. Train model
python python/scripts/07_provincial_model.py

# 4. Make predictions
python python/scripts/predict.py \
  --model models/model-dtb-prov.joblib \
  --preprocessor models/preprocessor-dtb-prov.joblib \
  --raster data/processed/predictors.tif \
  --output outputs/dtb-rf-prov-python.tif
```

## Performance Considerations

### R Advantages
- ✅ Established geospatial ecosystem (sf, terra, stars)
- ✅ Rich spatial statistics packages
- ✅ Excellent for statistical modeling and visualization
- ✅ Quarto for reproducible reports
- ✅ Complete implementation of all workflow steps

### Python Advantages
- ✅ Larger ML ecosystem (sklearn, xgboost, pytorch, tensorflow)
- ✅ Better deep learning support
- ✅ More deployment options
- ✅ Easier integration with web services
- ✅ Jupyter notebooks for interactive analysis
- ✅ Growing geospatial capabilities (geopandas, rasterio)

## Integration

Both implementations can work together:

1. **Use R for data preparation** (01-grids, 02-nlp) → outputs in data/processed/
2. **Use Python for modeling** → train models with Python scripts
3. **Compare results** → Both save to same outputs/ directory

Or vice versa - use Python for some steps and R for others.

## Future Enhancements

### Python Implementation Roadmap
- [ ] Complete NLP module for litholog classification
- [ ] Add grid/terrain data download functionality
- [ ] Implement IDW and Kriging methods
- [ ] Add model comparison experiments
- [ ] Create Jupyter notebooks for analysis
- [ ] Add comprehensive tests
- [ ] Benchmark against R implementation
- [ ] Add GPU acceleration options
- [ ] Implement additional models (neural networks)

## Conclusion

Both implementations provide robust depth to bedrock prediction capabilities. The R implementation is complete and production-ready, while the Python implementation provides core functionality with room for extension. Users can choose based on their:

- Programming language preference
- Deployment requirements
- Integration needs
- Team expertise
- Available ecosystem tools

The side-by-side implementations also facilitate:
- Cross-validation of methodologies
- Language-agnostic collaboration
- Flexibility in deployment environments
- Educational comparison of ML frameworks
