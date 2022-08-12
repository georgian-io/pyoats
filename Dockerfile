# syntax=docker/dockerfile:1
FROM ufoym/deepo:all-jupyter-py38

RUN apt-get -qq update && \
    apt-get -y -qq -o=Dpkg::Use-Pty=0 install libssl-dev libcurl4-openssl-dev \
    libyaml-dev build-essential libopenblas-dev libcap-dev ffmpeg

COPY ./requirements.txt ./requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir --no-deps tamu_d3m==2022.05.23 \
    pyod==1.0.4 tamu_axolotl==2021.4.8 \
    combo==0.1.3 simplejson==3.12.0 PyWavelets==1.1.1 \
    pillow==7.1.2 pyod==1.0.4 nimfa==1.4.0 \
    stumpy==1.4.0 more-itertools==8.5.0


RUN pip3 install --no-cache-dir --no-deps tods==0.0.2

# RUN cd /home/one && git clone https://github.com/datamllab/tods.git && \
#     cd tods && pip3 install --no-deps -e . && \
#     cd / && rm requirements.txt

COPY . /home/one
WORKDIR /home/one
