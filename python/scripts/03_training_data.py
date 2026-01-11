"""
Training data preparation script for DTB prediction.

This script:
1. Loads picks data (geological picks and NLP-predicted picks)
2. Combines pick sets (thinning overlapping points)
3. Extracts predictor values from raster at pick locations
4. Saves training data

Equivalent to R/03-training-data.R
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

# Add functions to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from functions.preprocessing import (
    load_geopackage,
    transform_crs,
    thin_points_by_distance,
    prepare_training_data,
    save_geopackage
)


def main():
    """Main function to prepare training data."""
    
    print("Loading picks data...")
    
    # Define paths
    data_dir = Path(__file__).parent.parent.parent / "projdata"
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    
    # Create output directories if they don't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load picks data
    # Note: R uses .rds files, we'll need to convert or use CSV/GeoPackage
    picks_comp_file = data_dir / "picks.gpkg"  # or picks.csv
    picks_nlp_file = processed_dir / "picks-nlp.gpkg"  # or picks-nlp.csv
    predictors_file = processed_dir / "predictors.tif"
    
    # Check if files exist
    if not picks_comp_file.exists():
        print(f"Warning: {picks_comp_file} not found. Using sample data path.")
        # In practice, you'd load from the actual data source
    
    if not predictors_file.exists():
        print(f"Warning: {predictors_file} not found.")
        print("Please run 01_grids.py first to generate predictor rasters.")
        return
    
    # For demonstration, we'll show the expected workflow
    # In practice, these files would be created by previous steps
    
    # Load geological picks (high quality)
    try:
        picks_comp = load_geopackage(str(picks_comp_file))
        # Transform to projected CRS (EPSG:3402 - Alberta 10-TM)
        picks_comp = transform_crs(picks_comp, "EPSG:3402")
        print(f"Loaded {len(picks_comp)} geological picks")
    except Exception as e:
        print(f"Could not load geological picks: {e}")
        print("Creating sample data structure...")
        # Sample structure for reference
        picks_comp = gpd.GeoDataFrame()
    
    # Load NLP-predicted picks (from water well lithologs)
    try:
        picks_nlp = load_geopackage(str(picks_nlp_file))
        picks_nlp = transform_crs(picks_nlp, "EPSG:3402")
        print(f"Loaded {len(picks_nlp)} NLP-predicted picks")
    except Exception as e:
        print(f"Could not load NLP picks: {e}")
        picks_nlp = gpd.GeoDataFrame()
    
    # Combine pick sets
    # Only use predicted picks where geological picks are sparse
    if len(picks_comp) > 0 and len(picks_nlp) > 0:
        print("\nThinning NLP picks to avoid overlap with geological picks...")
        picks_nlp_thinned = thin_points_by_distance(
            picks_comp, 
            picks_nlp,
            max_distance=3000,  # 3 km threshold
            k=100
        )
        print(f"Retained {len(picks_nlp_thinned)} non-overlapping NLP picks")
        
        # Combine both datasets
        picks = pd.concat([picks_comp, picks_nlp_thinned], ignore_index=True)
        print(f"Total picks: {len(picks)}")
        
        # Save combined picks
        combined_picks_file = output_dir / "picked-combined.gpkg"
        save_geopackage(picks, str(combined_picks_file))
        print(f"Saved combined picks to {combined_picks_file}")
    else:
        print("\nUsing only available picks data...")
        picks = picks_comp if len(picks_comp) > 0 else picks_nlp
    
    # Extract training data
    if len(picks) > 0 and predictors_file.exists():
        print("\nExtracting predictor values at pick locations...")
        training_data = prepare_training_data(
            picks,
            str(predictors_file),
            target_col='bedrock_dep',
            id_col='id'
        )
        
        print(f"Training data shape: {training_data.shape}")
        print(f"Columns: {list(training_data.columns)}")
        
        # Save training data
        training_data_file = processed_dir / "training-data.parquet"
        training_data.to_parquet(training_data_file, index=False)
        print(f"Saved training data to {training_data_file}")
        
        # Also save as CSV for easy inspection
        training_data_csv = processed_dir / "training-data.csv"
        training_data.to_csv(training_data_csv, index=False)
        print(f"Saved training data to {training_data_csv}")
        
        # Print summary statistics
        print("\nTraining data summary:")
        print(training_data.describe())
    else:
        print("\nCannot extract training data - missing picks or predictors.")
        print("Please ensure previous steps have been completed.")
    
    print("\nTraining data preparation complete!")


if __name__ == "__main__":
    main()
