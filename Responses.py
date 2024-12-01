import requests

def fetch_geoserver_data(bbox):
    base_url = "http://localhost:8080/geoserver/Sample_Visible/wms"
    params = {
        "service": "WMS",
        "version": "1.1.0",
        "request": "GetMap",
        "layers": "Sample_Visible:3RIMG_14OCT2024_0015_L1B_STD_V01R00_IMG_VIS",
        "bbox": bbox,  # Pass your manual bbox here
        "width": 768,
        "height": 762,
        "srs": "EPSG:4326",
        "styles": "",
        "format": "application/openlayers"
    }

    # Make the HTTP GET request
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        # Save the map image or data locally
        with open("output_map.png", "wb") as file:
            file.write(response.content)
        print("Map image saved successfully!")
    else:
        print("Failed to fetch data:", response.status_code, response.text)

# Example bbox: "xmin,ymin,xmax,ymax"
manual_bbox = "-7.290232060415251,-80.70208807014171,155.28087866883524,80.69326497372339"
fetch_geoserver_data(manual_bbox)
