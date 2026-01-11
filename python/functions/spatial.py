"""
Spatial feature engineering utilities for DTB prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def compute_euclidean_distance_fields(X: pd.DataFrame,
                                     coord_cols: Tuple[str, str] = ('xcoords', 'ycoords'),
                                     n_reference_points: int = 5) -> pd.DataFrame:
    """
    Compute Euclidean Distance Fields (EDFs) as spatial features.
    
    Creates distance features to reference points (corners and center of the study area).
    
    Parameters
    ----------
    X : pd.DataFrame
        Input data with coordinate columns.
    coord_cols : tuple of str, default=('xcoords', 'ycoords')
        Names of coordinate columns.
    n_reference_points : int, default=5
        Number of reference points (5 uses corners + center).
        
    Returns
    -------
    pd.DataFrame
        Input data with added EDF columns (edf1, edf2, ...).
    """
    X_out = X.copy()
    
    x_col, y_col = coord_cols
    x = X[x_col].values
    y = X[y_col].values
    
    # Define reference points
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    reference_points = [
        (x_min, y_max),  # top-left
        (x_max, y_max),  # top-right
        (x_min, y_min),  # bottom-left
        (x_max, y_min),  # bottom-right
        (x_center, y_center)  # center
    ][:n_reference_points]
    
    # Compute distances to each reference point
    coords = np.column_stack([x, y])
    for i, ref_point in enumerate(reference_points, 1):
        ref_array = np.array([ref_point])
        distances = cdist(coords, ref_array, metric='euclidean').flatten()
        X_out[f'edf{i}'] = distances
    
    return X_out


def compute_spatial_lag(X: pd.DataFrame,
                       y: pd.Series,
                       coord_cols: Tuple[str, str] = ('xcoords', 'ycoords'),
                       neighbors: int = 5,
                       weight_func: str = 'gaussian') -> pd.DataFrame:
    """
    Compute spatial lag features using weighted mean of neighboring points.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features with coordinate columns.
    y : pd.Series
        Target variable.
    coord_cols : tuple of str, default=('xcoords', 'ycoords')
        Names of coordinate columns.
    neighbors : int, default=5
        Number of nearest neighbors to use.
    weight_func : str, default='gaussian'
        Weighting function: 'gaussian', 'inverse', or 'uniform'.
        
    Returns
    -------
    pd.DataFrame
        Input data with added spatial lag column.
    """
    X_out = X.copy()
    
    x_col, y_col = coord_cols
    coords = X[[x_col, y_col]].values
    
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(coords)
    
    # Query for k+1 neighbors (including the point itself)
    distances, indices = tree.query(coords, k=neighbors + 1)
    
    # Exclude the point itself (distance 0)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # Compute weights
    if weight_func == 'gaussian':
        # Gaussian weights based on distance
        sigma = distances.mean()
        weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    elif weight_func == 'inverse':
        # Inverse distance weighting
        weights = 1.0 / (distances + 1e-10)  # Add small value to avoid division by zero
    elif weight_func == 'uniform':
        # Uniform weights
        weights = np.ones_like(distances)
    else:
        raise ValueError(f"Unknown weight_func: {weight_func}")
    
    # Normalize weights
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Compute weighted mean of target values
    y_array = y.values
    spatial_lag = np.sum(weights * y_array[indices], axis=1)
    
    X_out[f'spatial_lag_{neighbors}'] = spatial_lag
    
    return X_out


def compute_multiple_spatial_lags(X: pd.DataFrame,
                                  y: pd.Series,
                                  coord_cols: Tuple[str, str] = ('xcoords', 'ycoords'),
                                  neighbor_counts: List[int] = [5, 10, 15],
                                  weight_func: str = 'gaussian') -> pd.DataFrame:
    """
    Compute spatial lag features for multiple neighborhood sizes.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features with coordinate columns.
    y : pd.Series
        Target variable.
    coord_cols : tuple of str, default=('xcoords', 'ycoords')
        Names of coordinate columns.
    neighbor_counts : list of int, default=[5, 10, 15]
        List of neighbor counts to use.
    weight_func : str, default='gaussian'
        Weighting function: 'gaussian', 'inverse', or 'uniform'.
        
    Returns
    -------
    pd.DataFrame
        Input data with added spatial lag columns for each neighbor count.
    """
    X_out = X.copy()
    
    for neighbors in neighbor_counts:
        X_out = compute_spatial_lag(
            X_out, y, coord_cols, neighbors, weight_func
        )
    
    return X_out


def compute_spatial_neighbors_features(X: pd.DataFrame,
                                      y: pd.Series,
                                      coord_cols: Tuple[str, str] = ('xcoords', 'ycoords'),
                                      neighbors: int = 16) -> pd.DataFrame:
    """
    Add features based on values and distances to nearest neighbors.
    
    This creates features that capture the local spatial structure around each point.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features with coordinate columns.
    y : pd.Series
        Target variable.
    coord_cols : tuple of str, default=('xcoords', 'ycoords')
        Names of coordinate columns.
    neighbors : int, default=16
        Number of nearest neighbors.
        
    Returns
    -------
    pd.DataFrame
        Input data with neighbor-based features.
    """
    X_out = X.copy()
    
    x_col, y_col = coord_cols
    coords = X[[x_col, y_col]].values
    
    # Build KD-tree
    tree = cKDTree(coords)
    
    # Query for neighbors (excluding the point itself)
    distances, indices = tree.query(coords, k=neighbors + 1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # Get target values of neighbors
    y_array = y.values
    neighbor_values = y_array[indices]
    
    # Create features
    X_out[f'neighbor_mean_{neighbors}'] = neighbor_values.mean(axis=1)
    X_out[f'neighbor_std_{neighbors}'] = neighbor_values.std(axis=1)
    X_out[f'neighbor_min_{neighbors}'] = neighbor_values.min(axis=1)
    X_out[f'neighbor_max_{neighbors}'] = neighbor_values.max(axis=1)
    X_out[f'neighbor_dist_mean_{neighbors}'] = distances.mean(axis=1)
    X_out[f'neighbor_dist_min_{neighbors}'] = distances.min(axis=1)
    X_out[f'neighbor_dist_max_{neighbors}'] = distances.max(axis=1)
    
    return X_out


class SpatialFeatureEngineer:
    """
    Spatial feature engineering transformer for DTB prediction.
    
    Supports multiple spatial feature engineering strategies:
    - 'edfs': Euclidean Distance Fields
    - 'lags': Spatial lag features
    - 'neighbors': Neighbor-based features
    """
    
    def __init__(self, method: str = 'lags',
                 coord_cols: Tuple[str, str] = ('xcoords', 'ycoords'),
                 neighbor_counts: List[int] = [5, 10, 15]):
        """
        Initialize the spatial feature engineer.
        
        Parameters
        ----------
        method : str, default='lags'
            Feature engineering method: 'edfs', 'lags', or 'neighbors'.
        coord_cols : tuple of str, default=('xcoords', 'ycoords')
            Names of coordinate columns.
        neighbor_counts : list of int, default=[5, 10, 15]
            Neighbor counts for lag/neighbor methods.
        """
        self.method = method
        self.coord_cols = coord_cols
        self.neighbor_counts = neighbor_counts
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the transformer (does nothing, included for sklearn compatibility).
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series, optional
            Training target.
            
        Returns
        -------
        self
        """
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform features by adding spatial features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.
        y : pd.Series, optional
            Target variable (required for 'lags' and 'neighbors' methods).
            
        Returns
        -------
        pd.DataFrame
            Transformed features with spatial features added.
        """
        if self.method == 'edfs':
            return compute_euclidean_distance_fields(X, self.coord_cols)
        
        elif self.method == 'lags':
            if y is None:
                raise ValueError("Target variable y is required for spatial lag method")
            return compute_multiple_spatial_lags(
                X, y, self.coord_cols, self.neighbor_counts
            )
        
        elif self.method == 'neighbors':
            if y is None:
                raise ValueError("Target variable y is required for neighbors method")
            # Use the first neighbor count
            n_neighbors = self.neighbor_counts[0] if self.neighbor_counts else 16
            return compute_spatial_neighbors_features(
                X, y, self.coord_cols, n_neighbors
            )
        
        else:
            raise ValueError(f"Unknown spatial method: {self.method}")
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.
        y : pd.Series, optional
            Target variable.
            
        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        return self.fit(X, y).transform(X, y)
