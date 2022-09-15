# syntax=docker/dockerfile:1
# FROM ufoym/deepo:all-jupyter-py38
FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends git python3.8 python3 python3-pip


COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

RUN cd / && rm requirements.txt

COPY . /home/one

RUN export TF_FORCE_GPU_ALLOW_GROWTH=true && \
    export CUDA_HOME=/usr/local/cuda/

RUN cd /home/one && pip install -e .

# WORKDIR /home