#!/usr/bin/env python3
import sys
import os
import io
import argparse
import uuid
import base64
import logging
import time
import copy
import cv2
import insightface
import numpy as np
from typing import List, Union
from PIL import Image
from werkzeug.utils import secure_filename
from restoration import *
from flask import Flask, request, jsonify, make_response

LOG_LEVEL = logging.INFO
TMP_PATH = '/tmp/inswapper'
script_dir = os.path.dirname(os.path.abspath(__file__))
log_path = ''

# Mac does not have permission to /var/log for example
if sys.platform == 'linux':
    log_path = '/var/log/'

logging.basicConfig(
    filename=f'{log_path}inswapper.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=LOG_LEVEL
)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_elapsed_time(self):
        end = time.time()
        return round(end - self.start, 1)


def get_args():
    parser = argparse.ArgumentParser(
        description='Inswapper REST API'
    )

    parser.add_argument(
        '-p', '--port',
        help='Port to listen on',
        type=int,
        default=8000
    )

    parser.add_argument(
        '-H', '--host',
        help='Host to bind to',
        default='0.0.0.0'
    )

    return parser.parse_args()


def get_face_swap_model(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def get_face_analyser(model_path: str,
                      det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints")
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser,
                 frame:np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser,
                   frame:np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(face_swapper,
              source_face,
              target_face,
              temp_frame):
    """
    paste source_face on target image
    """
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def process(source_img: Union[Image.Image, List],
            target_img: Image.Image,
            model: str):

    # load face_analyser
    face_analyser = get_face_analyser(model)

    # load face_swapper
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = get_face_swap_model(model_path)

    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    # detect faces that will be replaced in target_img
    target_faces = get_many_faces(face_analyser, target_img)
    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if isinstance(source_img, list) and len(source_img) == len(target_faces):
            # replace faces in target image from the left to the right by order
            for i in range(len(target_faces)):
                target_face = target_faces[i]
                source_face = get_one_face(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                if source_face is None:
                    raise Exception("No source face found!")
                temp_frame = swap_face(face_swapper, source_face, target_face, temp_frame)
        else:
            # replace all faces in target image to same source_face
            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            source_face = get_one_face(face_analyser, source_img)
            if source_face is None:
                raise Exception("No source face found!")
            for target_face in target_faces:
                temp_frame = swap_face(face_swapper, source_face, target_face, temp_frame)
        result = temp_frame
    else:
        print("No target faces found!")

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image


def face_swap(src_img_path, target_img_path):
    source_img_paths = src_img_path.split(';')
    source_img = [Image.open(img_path) for img_path in source_img_paths]
    target_img = Image.open(target_img_path)

    # download from https://huggingface.co/deepinsight/inswapper/tree/main
    model = os.path.join(script_dir, 'checkpoints/inswapper_128.onnx')
    result_image = process(source_img, target_img, model)

    # make sure the ckpts downloaded successfully
    check_ckpts()

    # https://huggingface.co/spaces/sczhou/CodeFormer
    upsampler = set_realesrgan()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    codeformer_net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=['32', '64', '128', '256'],
    ).to(device)

    ckpt_path = os.path.join(script_dir, 'CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth')
    checkpoint = torch.load(ckpt_path)['params_ema']
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()
    result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

    background_enhance = True
    face_upsample = True
    upscale = 1
    codeformer_fidelity = 0.5

    result_image = face_restoration(
        result_image,
        background_enhance,
        face_upsample,
        upscale,
        codeformer_fidelity,
        upsampler,
        codeformer_net,
        device
    )

    result_image = Image.fromarray(result_image)
    output_buffer = io.BytesIO()
    result_image.save(output_buffer, format='JPEG')
    image_data = output_buffer.getvalue()

    return base64.b64encode(image_data).decode('utf-8')


app = Flask(__name__)


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify(
        {
            'status': 'error',
            'msg': f'Bad Request',
            'detail': str(error)
        }
    ), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify(
        {
            'status': 'error',
            'msg': f'{request.url} not found',
            'detail': str(error)
        }
    ), 404)


@app.errorhandler(500)
def internal_server_error(error):
    return make_response(jsonify(
        {
            'status': 'error',
            'msg': 'Internal Server Error',
            'detail': str(error)
        }
    ), 500)


@app.route('/', methods=['GET'])
def ping():
    return make_response(jsonify(
        {
            'status': 'ok'
        }
    ), 200)


@app.route('/faceswap', methods=['POST'])
def face_swap_api():
    total_timer = Timer()
    logging.info('Received face swap API request')

    if not os.path.exists(TMP_PATH):
        logging.info(f'Creating temporary directory: {TMP_PATH}')
        os.makedirs(TMP_PATH)

    # Get the source image file
    source_file = request.files['source_image']
    source_filename = secure_filename(source_file.filename)
    source_unique_id = uuid.uuid4()
    source_file_extension = os.path.splitext(source_filename)[1]
    source_image_path = f'{TMP_PATH}/{source_unique_id}{source_file_extension}'
    logging.info(f'Saving face swap source image to disk: {source_image_path}')
    source_file.save(source_image_path)
    logging.info(f'Successfully saved face swap source image: {source_image_path}')

    # Get the target image file
    target_file = request.files['target_image']
    target_filename = secure_filename(target_file.filename)
    target_unique_id = uuid.uuid4()
    target_file_extension = os.path.splitext(target_filename)[1]
    target_image_path = f'{TMP_PATH}/{target_unique_id}{target_file_extension}'
    logging.info(f'Saving face swap target image to disk: {target_image_path}')
    target_file.save(target_image_path)
    logging.info(f'Successfully saved face swap target image: {target_image_path}')

    try:
        logging.debug('Swapping face')
        face_swap_timer = Timer()
        result_image = face_swap(source_image_path, target_image_path)
        face_swap_time = face_swap_timer.get_elapsed_time()
        logging.info(f'Time taken to swap face: {face_swap_time} seconds')
    except Exception as e:
        logging.error(e)
        raise Exception('Face swap failed')

    # Clean up temporary images
    logging.debug('Face swap image created successfully')
    logging.debug(f'Deleting temporary source face swap image: {source_image_path}')
    os.remove(source_image_path)
    logging.debug(f'Deleting temporary target face swap image: {target_image_path}')
    os.remove(target_image_path)

    total_time = total_timer.get_elapsed_time()
    logging.info(f'Total time taken for face swap API call {total_time} seconds')

    return make_response(jsonify(
        {
            'status': 'ok',
            'image': result_image
        }
    ), 200)


if __name__ == '__main__':
    args = get_args()

    app.run(
        host=args.host,
        port=args.port
    )
