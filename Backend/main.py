from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import requests
from PIL import Image
from io import BytesIO
import os

app = FastAPI()

# Directory to save the images
IMAGE_SAVE_PATH = "images"
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

def fetch_wms_image(
    geoserver_url: str,
    layers: str,
    bbox: str,
    width: int = 512,
    height: int = 512,
    image_format: str = "image/png"
) -> str:
    """
    Fetches a WMS image from the GeoServer URL using the provided parameters.

    Args:
        geoserver_url (str): URL of the GeoServer WMS service.
        layers (str): Layers to be fetched.
        bbox (str): Bounding box in 'minX,minY,maxX,maxY' format.
        width (int): Width of the output image.
        height (int): Height of the output image.
        image_format (str): Image format (default is 'image/png').

    Returns:
        str: Path to the saved image file.
    """
    # Parse the bounding box
    min_x, min_y, max_x, max_y = map(float, bbox.split(","))

    # Build the WMS GetMap request URL
    wms_params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": layers,
        "bbox": f"{min_x},{min_y},{max_x},{max_y}",
        "width": width,
        "height": height,
        "srs": "EPSG:4326",
        "format": image_format,
    }

    # Make the WMS request
    response = requests.get(geoserver_url, params=wms_params, stream=True)
    response.raise_for_status()

    # Save the image locally
    image_name = f"wms_image_{min_x}_{min_y}_{max_x}_{max_y}.png"
    image_path = os.path.join(IMAGE_SAVE_PATH, image_name)

    image = Image.open(BytesIO(response.content))
    image.save(image_path)

    return image_path

@app.get("/get_image/")
async def get_wms_image(
    geoserver_url: str = Query(..., description="GeoServer WMS service URL"),
    layers: str = Query(..., description="Comma-separated layers to be fetched"),
    bbox: str = Query(..., description="Bounding box in 'minX,minY,maxX,maxY' format"),
    width: int = Query(512, description="Width of the output image"),
    height: int = Query(512, description="Height of the output image"),
    format: str = Query("image/png", description="Image format (default: image/png)")
):
    """
    FastAPI route that fetches a WMS image from the provided GeoServer URL and bounding box.
    """
    try:
        # Use the reusable function to fetch the image
        image_path = fetch_wms_image(
            geoserver_url=geoserver_url,
            layers=layers,
            bbox=bbox,
            width=width,
            height=height,
            image_format=format
        )
        return FileResponse(image_path, media_type=format, filename=os.path.basename(image_path))

    except Exception as e:
        return {"error": str(e)}
