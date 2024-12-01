import numpy as np
import rasterio

# Paths
grayscale_path = r"3RIMG_14OCT2024_0015_L1B_STD_V01R00_IMG_VIS (1).tif"  # Path to your grayscale file
rgb_output_path = r"output_rgb_file.tif"  # Writable directory for output file

# Open the grayscale raster
with rasterio.open(grayscale_path) as src:
    gray_data = src.read(1)  # Read the first (and only) band
    profile = src.profile

    # Create RGB data by duplicating the grayscale data into 3 bands
    rgb_data = np.stack([gray_data, gray_data, gray_data])

    # Update profile for 3 bands
    profile.update(count=3, dtype=gray_data.dtype)

    # Write the new RGB raster to the writable directory
    with rasterio.open(rgb_output_path, "w", **profile) as dst:
        dst.write(rgb_data)

print("RGB raster file created successfully at:", rgb_output_path)
