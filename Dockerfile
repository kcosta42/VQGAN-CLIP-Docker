FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /app

COPY ./requirements.txt /requirements.txt
RUN python -m pip install -r /requirements.txt
