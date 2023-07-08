#!/usr/bin/env python3
import io
import uuid
import requests
import base64
from PIL import Image

URL = 'http://127.0.0.1:8000/faceswap'

SOURCE_IMAGE = '/Users/ashley/src/inswapper/data/src.jpg'
TARGET_IMAGE = '/Users/ashley/src/inswapper/data/target.jpg'


if __name__ == '__main__':
    files = {
        'source_image': open(SOURCE_IMAGE, 'rb'),
        'target_image': open(TARGET_IMAGE, 'rb')
    }

    r = requests.post(
        URL,
        files=files
    )

    print(f'HTTP status code: {r.status_code}')

    resp_json = r.json()

    img = Image.open(io.BytesIO(base64.b64decode(resp_json['image'])))
    output_file = f'{uuid.uuid4()}.jpg'

    with open(output_file, 'wb') as f:
        print(f'Saving image: {output_file}')
        img.save(f, format='JPEG')
