import requests

def fetch_and_save_html(url, params, output_file):
    try:
        # Send a GET request to the server
        response = requests.get(url, params=params)

        # Check if the response contains HTML
        if "text/html" in response.headers.get('Content-Type', ''):
            print("The server returned an HTML page.")

            # Save the HTML content to a file
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(response.text)
            print(f"HTML content saved to {output_file}. You can open it in a browser to view.")
        else:
            print("The server did not return an HTML page.")

    except Exception as e:
        print("An error occurred:", str(e))

# GeoServer URL and parameters
url = "http://localhost:8080/geoserver/Sample_Visible/wms"
params = {
    "service": "WMS",
    "version": "1.1.0",
    "request": "GetMap",
    "layers": "Sample_Visible:3RIMG_14OCT2024_0015_L1B_STD_V01R00_IMG_VIS",
    "bbox": "-7.290232060415251,-80.70208807014171,155.28087866883524,80.69326497372339",
    "width": 768,
    "height": 762,
    "srs": "EPSG:4326",
    "styles": "",
    "format": "application/openlayers"  # Adjust this if needed
}

# Save HTML response to a file
output_file = "response.html"
fetch_and_save_html(url, params, output_file)
