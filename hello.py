import rasterio

# Path to your raster file
file_path = r"3RIMG_14OCT2024_0015_L1B_STD_V01R00_IMG_VIS (1).tif"

try:
    # Open the raster file
    with rasterio.open(file_path) as src:
        print("Driver:", src.driver)
        print("Width:", src.width)
        print("Height:", src.height)
        print("Number of Bands:", src.count)
        print("CRS:", src.crs)
        print("Transform:", src.transform)

        # Metadata of the raster
        print("\nMetadata:")
        print(src.meta)

        # Read metadata specific to tags
        print("\nTags:")
        print(src.tags())

except Exception as e:
    print(f"An error occurred: {e}")
