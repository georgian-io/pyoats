# syntax=docker/dockerfile:1
FROM ufoym/deepo:all-jupyter-py38

RUN apt-get -qq update && \
    apt-get -y -qq -o=Dpkg::Use-Pty=0 install libssl-dev libcurl4-openssl-dev \
    libyaml-dev build-essential libopenblas-dev libcap-dev ffmpeg

COPY ./requirements.txt ./requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt && \
    git clone https://github.com/datamllab/tods.git && \
    cd tods && pip3 install --no-cache-dir --no-deps . && \
    pip3 install --no-cache-dir --no-deps tamu_d3m==2022.05.23 && \
    cd .. && rm -rf tods && rm requirements.txt

COPY . /home/one
WORKDIR /home/one
