import rasterio

# Open your GeoTIFF file
with rasterio.open("3RIMG_14OCT2024_0015_L1B_STD_V01R00_IMG_VIS.tif") as src:
    print(src.crs)
