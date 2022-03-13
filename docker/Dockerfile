FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

RUN apt update \
    && apt install -y \
#    wget \
#    curl \
#    sudo \
    python3 \
    python3-pip
#    nvidia-container-toolkit

#RUN gedit nvidia-container-runtime-script.sh
#RUN curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
#    sudo apt-key add - distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
#    curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
#    sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list \
#    sudo apt-get update
#RUN sh nvidia-container-runtime-script.sh

#RUN systemctl restart dockerd

RUN pip3 install -U pip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
#ENTRYPOINT [ "python" ]
