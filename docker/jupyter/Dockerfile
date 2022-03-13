FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

RUN apt update \
    && apt install -y \
#    wget \
#    curl \
#    sudo \
    python3 \
    python3-pip

RUN pip3 install -U pip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
