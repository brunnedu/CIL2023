from typing import Tuple
import json
import io
import os

import click
import numpy as np
from tqdm import tqdm
import requests
from PIL import Image
from datetime import datetime

base_url = "https://maps.googleapis.com/maps/api/staticmap"
api_key = os.getenv("GMAPS_API_KEY")


def coordinate_generator(n, bbox):
    """Returns n random coordinates within the bbox."""
    for i in range(n):
        yield i, np.random.uniform(bbox[1]["lat"], bbox[0]["lat"]), np.random.uniform(bbox[0]["lng"], bbox[1]["lng"])


def process_image(lat: float, lng: float, roadmap=True, zoom=18, size=400) -> Tuple[Image.Image, str]:
    """
    Calls the Google Maps API for the location and processes the images to extract roads.

    :param lat: latitude
    :param lng: longitude
    :param roadmap: if roadmap images should be used, otherwise satellite images will be fetched
    :param zoom: zoom of images
    :param size: size of the images
    :return: tuple(image, Google Maps api url without API key)
    """
    map_type = "roadmap" if roadmap else "satellite"
    border_size = 25
    actual_size = size + border_size * 2  # borders of 25 pixels are removed because of watermark

    # Construct Google Maps URL
    # Roads are marked in red, all points of interest and labels removed
    url = f"{base_url}?center={lat},{lng}&zoom={zoom}&size={actual_size}x{actual_size}" \
          f"&maptype={map_type}&key={api_key}" \
          "&style=feature:road|element:geometry|color:0xff0000" \
          "&style=feature:poi|visibility:off" \
          "&style=feature:transit|visibility:off" \
          "&style=feature:administrative|visibility:off" \
          "&style=feature:all|element:labels|visibility:off"

    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    # crop image and remove watermark
    image = image.crop((border_size, border_size, actual_size - border_size, actual_size - border_size))

    if roadmap:
        # extract roads and mask them
        img_roadmap = image.convert('RGB')
        img_array = np.asarray(img_roadmap)
        img_mask = (img_array[:, :, 0] >= 254) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 0)
        img_streets = np.array(img_mask, dtype=np.uint8) * 255
        image = Image.fromarray(img_streets)

    return image, url.replace(api_key, "GMAPS_API_KEY")


def process_location(location, log_file, satellite_path, roadmap_path) -> None:
    """
    Processes location according to config file.

    :param location: location dictionary, see configs/data_2022.json
    :param log_file: log file
    :param satellite_path: output directory for satellite images
    :param roadmap_path :output directory for roadmap images
    :return: None
    """
    print(f"Collecting {location['n_images']} image{'s' if location['n_images'] == 1 else 's'} "
          f"for location '{location['name']}'...")

    for i, lat, lng in tqdm(coordinate_generator(location["n_images"], location["bbox"]), total=location["n_images"]):
        img_satellite, _ = process_image(lat, lng, roadmap=False)
        img_roadmap, roadmap_url = process_image(lat, lng, roadmap=True)

        filename = f"{location['name']}_satimage_{i}.png"
        img_satellite.save(os.path.join(satellite_path, filename))
        img_roadmap.save(os.path.join(roadmap_path, filename))
        log_file.write(f"{location['name']},{lat},{lng},{filename},{roadmap_url}\n")


@click.command()
@click.option("--config", "config_path", help="JSON config path", required=True)
@click.option("--output", "output_path", help="Output directory path", required=True)
def main(config_path, output_path):
    """
    Collects additional training data from Google Maps.
    Requires a Google Maps API Key in the environment variable 'GMAPS_API_KEY'.
    """
    # check if google maps api key is available
    if not api_key:
        print(f"Google Maps API Key in environment variable 'GMAPS_API_KEY' required!")
    elif os.path.exists(output_path) and len(os.listdir(output_path)) != 0:
        print(f"Output directory {output_path} is not empty!")
    else:
        try:
            with open(config_path) as config_file:
                config = json.load(config_file)

            satellite_path = os.path.join(output_path, "images")
            roadmap_path = os.path.join(output_path, "groundtruth")
            os.makedirs(satellite_path, exist_ok=True)
            os.makedirs(roadmap_path, exist_ok=True)

            # create log file for locations that contain name, coordinates, ...
            log_path = os.path.join(output_path, datetime.now().strftime("logs_%Y%m%d%H%M%S"))
            with open(log_path, "w") as log_file:
                for location in config:
                    process_location(location, log_file, satellite_path, roadmap_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
