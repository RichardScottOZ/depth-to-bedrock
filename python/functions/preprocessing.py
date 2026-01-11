"""
Data preprocessing utilities for DTB prediction.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional, List
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point


def load_picks_data(filepath: str, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Load picks data and convert to GeoDataFrame.
    
    Parameters
    ----------
    filepath : str
        Path to the picks data file (CSV, parquet, or geopackage).
    crs : str, default="EPSG:4326"
        Coordinate reference system for the geometry.
        
    Returns
    -------
    gpd.GeoDataFrame
        Picks data with geometry column.
    """
    # Try to read as various formats
    if filepath.endswith('.gpkg'):
        gdf = gpd.read_file(filepath)
    else:
        # Read as regular dataframe first
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        # Convert to GeoDataFrame if it has coordinate columns
        if 'longitude' in df.columns and 'latitude' in df.columns:
            geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        else:
            raise ValueError("Data must have 'longitude' and 'latitude' columns")
    
    return gdf


def transform_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """
    Transform GeoDataFrame to target CRS.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame.
    target_crs : str
        Target coordinate reference system (e.g., "EPSG:3402").
        
    Returns
    -------
    gpd.GeoDataFrame
        Transformed GeoDataFrame.
    """
    return gdf.to_crs(target_crs)


def thin_points_by_distance(gdf1: gpd.GeoDataFrame, 
                            gdf2: gpd.GeoDataFrame,
                            max_distance: float,
                            k: int = 100) -> gpd.GeoDataFrame:
    """
    Thin points from gdf2 that are within max_distance of points in gdf1.
    
    This is useful for combining multiple pick datasets where you want to keep
    high-quality picks (gdf1) and only add additional picks (gdf2) where they
    don't overlap.
    
    Parameters
    ----------
    gdf1 : gpd.GeoDataFrame
        Primary picks to keep.
    gdf2 : gpd.GeoDataFrame
        Secondary picks to thin.
    max_distance : float
        Maximum distance threshold (in CRS units).
    k : int, default=100
        Number of nearest neighbors to check.
        
    Returns
    -------
    gpd.GeoDataFrame
        Thinned version of gdf2.
    """
    from scipy.spatial import cKDTree
    
    # Get coordinates
    coords1 = np.array([[p.x, p.y] for p in gdf1.geometry])
    coords2 = np.array([[p.x, p.y] for p in gdf2.geometry])
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(coords1)
    
    # Query for points within distance
    distances, _ = tree.query(coords2, k=min(k, len(coords1)))
    
    # For each point in gdf2, check if any of k nearest neighbors is within max_distance
    min_distances = distances.min(axis=1) if distances.ndim > 1 else distances
    keep_mask = min_distances > max_distance
    
    return gdf2[keep_mask].copy()


def extract_raster_values(gdf: gpd.GeoDataFrame, 
                          raster_path: str,
                          band: int = 1) -> pd.DataFrame:
    """
    Extract raster values at point locations.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Points where to extract values.
    raster_path : str
        Path to the raster file.
    band : int, default=1
        Band number to extract (1-indexed).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with extracted values.
    """
    with rasterio.open(raster_path) as src:
        # Make sure CRS matches
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        # Extract coordinates
        coords = [(p.x, p.y) for p in gdf.geometry]
        
        # Sample the raster at point locations
        values = [x[band-1] for x in src.sample(coords)]
    
    return pd.DataFrame({'value': values})


def extract_multiband_raster_values(gdf: gpd.GeoDataFrame,
                                    raster_path: str,
                                    band_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract all bands from a multiband raster at point locations.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Points where to extract values.
    raster_path : str
        Path to the multiband raster file.
    band_names : list of str, optional
        Names for the bands. If None, uses 'band_1', 'band_2', etc.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each band.
    """
    with rasterio.open(raster_path) as src:
        # Make sure CRS matches
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        # Extract coordinates
        coords = [(p.x, p.y) for p in gdf.geometry]
        
        # Sample all bands
        values = list(src.sample(coords))
        values_array = np.array(values)
        
        # Create column names
        if band_names is None:
            band_names = [f'band_{i+1}' for i in range(values_array.shape[1])]
        elif len(band_names) != values_array.shape[1]:
            raise ValueError(f"Number of band_names ({len(band_names)}) must match "
                           f"number of bands ({values_array.shape[1]})")
        
        # Create dataframe
        df = pd.DataFrame(values_array, columns=band_names)
    
    return df


def prepare_training_data(picks_gdf: gpd.GeoDataFrame,
                         predictors_raster: str,
                         target_col: str = 'bedrock_dep',
                         id_col: str = 'id') -> pd.DataFrame:
    """
    Prepare training data by extracting predictor values at pick locations.
    
    Parameters
    ----------
    picks_gdf : gpd.GeoDataFrame
        Pick locations with target variable.
    predictors_raster : str
        Path to multiband raster with predictor variables.
    target_col : str, default='bedrock_dep'
        Name of the target variable column.
    id_col : str, default='id'
        Name of the ID column.
        
    Returns
    -------
    pd.DataFrame
        Training data with predictors and target.
    """
    with rasterio.open(predictors_raster) as src:
        # Get band names from raster metadata if available
        band_names = src.descriptions
        if not any(band_names):  # If descriptions are empty
            band_names = [f'predictor_{i+1}' for i in range(src.count)]
    
    # Extract predictor values
    predictors_df = extract_multiband_raster_values(
        picks_gdf, predictors_raster, band_names=list(band_names)
    )
    
    # Add coordinates
    coords_df = pd.DataFrame({
        'xcoords': [p.x for p in picks_gdf.geometry],
        'ycoords': [p.y for p in picks_gdf.geometry]
    })
    
    # Combine with target and id
    result_df = pd.concat([coords_df, predictors_df], axis=1)
    
    if target_col in picks_gdf.columns:
        result_df[target_col] = picks_gdf[target_col].values
    
    if id_col in picks_gdf.columns:
        result_df[id_col] = picks_gdf[id_col].values
    
    # Average DTB in case of multiple picks per cell
    if 'xcoords' in result_df.columns and 'ycoords' in result_df.columns:
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        result_df = result_df.groupby(['xcoords', 'ycoords'])[numeric_cols].mean().reset_index()
    
    return result_df


def save_geopackage(gdf: gpd.GeoDataFrame, filepath: str, layer: Optional[str] = None) -> None:
    """
    Save GeoDataFrame to GeoPackage format.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Data to save.
    filepath : str
        Output file path.
    layer : str, optional
        Layer name. If None, uses default.
    """
    if layer:
        gdf.to_file(filepath, driver='GPKG', layer=layer)
    else:
        gdf.to_file(filepath, driver='GPKG')


def load_geopackage(filepath: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame from GeoPackage format.
    
    Parameters
    ----------
    filepath : str
        Input file path.
    layer : str, optional
        Layer name to load. If None, loads first layer.
        
    Returns
    -------
    gpd.GeoDataFrame
        Loaded data.
    """
    if layer:
        return gpd.read_file(filepath, layer=layer)
    else:
        return gpd.read_file(filepath)
