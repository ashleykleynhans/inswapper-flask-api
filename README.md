# inswapper-flask-api

Python Flask API for Face Swapper and Restoration
powered by [insightface](https://github.com/deepinsight/insightface).

## Installation

### Clone this repository

```bash
git clone https://github.com/ashleykleynhans/inswapper-flask-api.git
cd inswapper-flask-api
```

### Install the required Python dependencies

#### Linux and Mac

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

#### Windows

```
python3 -m venv venv
venv\Scripts\activate
pip3 install -r requirements.txt
```

## Download Checkpoints

You will need to download the [face swap model](
https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx) and
save it under `./checkpoints`.

```bash
mkdir checkpoints
wget -O ./checkpoints/inswapper_128.onnx https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx 
```

## Face Restoration

In order to obtain better results, it is highly recommended to enable
a face restoration model, which will improve image quality.
This application uses [CodeFormer](https://github.com/sczhou/CodeFormer)
for face restoration.

The required models will be downloaded automatically the first time
the face swap API is called.

You will require [Git Large File Storage](
https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
to be installed on your system before you can run the following command.

```bash
git lfs install
git clone https://huggingface.co/spaces/sczhou/CodeFormer
```

## Examples

Refer to the [examples](./examples) provided for getting started
with making calls to the API.

## Acknowledgements

1. This project is inspired by [inswapper](https://huggingface.co/deepinsight/inswapper/tree/main).
2. Thanks [insightface.ai](https://insightface.ai/) for releasing their powerful swap model that made this possible.
3. This codebase is built on top of [inswapper](https://github.com/haofanwang/inswapper) and [CodeFormer](
   https://huggingface.co/spaces/sczhou/CodeFormer).
4. [inswapper](https://github.com/haofanwang/inswapper) is built on the top of [sd-webui-roop](
   https://github.com/s0md3v/sd-webui-roop) and [CodeFormer](https://huggingface.co/spaces/sczhou/CodeFormer).