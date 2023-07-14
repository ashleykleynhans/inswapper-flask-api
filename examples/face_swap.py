#!/usr/bin/env python3
import io
import uuid
import requests
import base64
from PIL import Image

URL = 'http://127.0.0.1:5000/faceswap'

SOURCE_IMAGE = '/Users/ashley/src/inswapper/data/src.jpg'
TARGET_IMAGE = '/Users/ashley/src/inswapper/data/target.jpg'


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        encoded_data = base64.b64encode(image_data).decode('utf-8')
        return encoded_data


def save_result_image(resp_json):
    img = Image.open(io.BytesIO(base64.b64decode(resp_json['output']['image'])))
    output_file = f'{uuid.uuid4()}.jpg'

    with open(output_file, 'wb') as f:
        print(f'Saving image: {output_file}')
        img.save(f, format='JPEG')


if __name__ == '__main__':
    payload = {
        'source_image': encode_image_to_base64(SOURCE_IMAGE),
        'target_image': encode_image_to_base64(TARGET_IMAGE)
    }

    r = requests.post(
        URL,
        json=payload
    )

    print(f'HTTP status code: {r.status_code}')

    resp_json = r.json()

    img = Image.open(io.BytesIO(base64.b64decode(resp_json['image'])))
    output_file = f'{uuid.uuid4()}.jpg'

    with open(output_file, 'wb') as f:
        print(f'Saving image: {output_file}')
        img.save(f, format='JPEG')
