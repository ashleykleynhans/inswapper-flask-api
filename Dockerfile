FROM python:3.9.9

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y


RUN apt-get install -y git git-lfs

WORKDIR /app

COPY  . /app/


RUN git lfs install

RUN pip3 install -r requirements.txt

CMD ["python3","app.py","-H","0.0.0.0","-p","5000"]