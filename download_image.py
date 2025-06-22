import requests
import shutil
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from sfs import horn_sfs

def download_image(url, filename):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, stream=True, proxies=None)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        print(f"Image downloaded successfully as {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")

if __name__ == "__main__":
    image_url = "https://upload.wikimedia.org/wikipedia/commons/c/ce/Virtanen_lunar_crater.jpg"
    output_filename = "virtanen_crater.jpg"
    download_image(image_url, output_filename) 