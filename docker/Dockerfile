FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

RUN apt update \
    && apt install -y \
#    wget \
#    curl \
#    sudo \
    python3.8 \
    python3-pip

RUN pip install -U pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt