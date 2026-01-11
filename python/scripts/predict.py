"""
Simple script to make DTB predictions using a trained model.

This script loads a trained model and makes predictions on new raster data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import joblib

# Add functions to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from functions.dtb import predict_dtb


def predict_raster(model, preprocessor, raster_path: str, output_path: str,
                   chunk_size: int = 1000):
    """
    Make predictions on a raster file using a trained model.
    
    Parameters
    ----------
    model : sklearn model
        Trained model.
    preprocessor : DTBPreprocessor
        Fitted preprocessor.
    raster_path : str
        Path to input raster with predictor bands.
    output_path : str
        Path for output prediction raster.
    chunk_size : int, default=1000
        Size of chunks for processing (to manage memory).
    """
    print(f"Opening raster: {raster_path}")
    
    with rasterio.open(raster_path) as src:
        # Get metadata
        meta = src.meta.copy()
        meta.update(count=1, dtype='float32')
        
        # Get band names/descriptions
        band_names = src.descriptions
        if not any(band_names):
            band_names = [f'band_{i+1}' for i in range(src.count)]
        
        print(f"Raster shape: {src.height} x {src.width}")
        print(f"Number of bands: {src.count}")
        print(f"Bands: {list(band_names)}")
        
        # Create output raster
        with rasterio.open(output_path, 'w', **meta) as dst:
            # Process in chunks
            n_chunks_y = (src.height + chunk_size - 1) // chunk_size
            n_chunks_x = (src.width + chunk_size - 1) // chunk_size
            total_chunks = n_chunks_y * n_chunks_x
            
            print(f"\nProcessing {total_chunks} chunks...")
            chunk_idx = 0
            
            for i in range(0, src.height, chunk_size):
                for j in range(0, src.width, chunk_size):
                    chunk_idx += 1
                    
                    # Define window
                    height = min(chunk_size, src.height - i)
                    width = min(chunk_size, src.width - j)
                    window = Window(j, i, width, height)
                    
                    # Read data
                    data = src.read(window=window)
                    
                    # Reshape to (n_pixels, n_bands)
                    n_bands, h, w = data.shape
                    data_reshaped = data.reshape(n_bands, -1).T
                    
                    # Create dataframe
                    df = pd.DataFrame(data_reshaped, columns=band_names)
                    
                    # Handle nodata
                    nodata_mask = (df == src.nodata).any(axis=1) if src.nodata is not None else np.zeros(len(df), dtype=bool)
                    
                    # Make predictions for valid pixels
                    predictions = np.full(len(df), src.nodata or -9999, dtype='float32')
                    
                    if not nodata_mask.all():
                        valid_data = df[~nodata_mask]
                        
                        # Preprocess
                        try:
                            valid_processed = preprocessor.transform(valid_data)
                            
                            # Predict
                            valid_preds = predict_dtb(model, valid_processed)
                            predictions[~nodata_mask] = valid_preds
                        except Exception as e:
                            print(f"Warning: Error processing chunk {chunk_idx}: {e}")
                    
                    # Reshape back to 2D
                    predictions_2d = predictions.reshape(h, w)
                    
                    # Write to output
                    dst.write(predictions_2d, 1, window=window)
                    
                    if chunk_idx % 10 == 0:
                        print(f"  Processed {chunk_idx}/{total_chunks} chunks ({chunk_idx*100/total_chunks:.1f}%)")
            
            print(f"\nPrediction complete! Output saved to: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Make DTB predictions using a trained model'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model file (.joblib)'
    )
    parser.add_argument(
        '--preprocessor',
        required=True,
        help='Path to preprocessor file (.joblib)'
    )
    parser.add_argument(
        '--raster',
        required=True,
        help='Path to input raster with predictor bands'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path for output prediction raster'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size for processing (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Load model and preprocessor
    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)
    
    print(f"Loading preprocessor from {args.preprocessor}...")
    preprocessor = joblib.load(args.preprocessor)
    
    # Make predictions
    predict_raster(
        model,
        preprocessor,
        args.raster,
        args.output,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
