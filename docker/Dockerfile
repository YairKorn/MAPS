FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
# MAINTAINER Wendelin Boehmer (of the PyMARL original: Tabish Rashid)

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64
ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu Packages
RUN apt-get update -y 
RUN DEBIAN_FRONTEND=noninteractive apt-get -yq install apt-utils
RUN apt-get install software-properties-common -y 
RUN add-apt-repository -y multiverse && apt-get update -y 
RUN apt-get upgrade -y 
RUN apt-get install -y apt-utils nano vim man build-essential wget sudo
RUN rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git

# This line from PyMARL's original Dockerfile has to commented out to work with the Ubuntu 18.04 image
# RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
#     rm -rf /var/lib/apt/lists/*

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Python packages we use (or used at one point...)
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger
RUN pip3 install pygame

RUN mkdir /install
WORKDIR /install

RUN pip3 install jsonpickle==0.9.6
# install Sacred (from OxWhirl fork)
RUN pip3 install setuptools
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN git clone --branch Splited_Layer https://github.com/YairKorn/MAPS.git /install/sacred

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip3 install torch
RUN pip3 install torchvision snakeviz pytest probscale

### -- This library must be added for DCG, one may need to adapt it to newer torch versions
RUN pip3 install torch_scatter

### -- SMAC
# RUN pip3 install git+https://github.com/oxwhirl/smac.git
# ENV SC2PATH /pymarl/3rdparty/StarCraftII

RUN pip3 install sacred
WORKDIR /install/sacred
# CMD ["python3", "src/main.py", "--config=dcg", "--env-config=het_adv_coverage"]

# ENV parameters to control setup of the docker
ENV map ""
ENV time 0
ENV alpha 1.0

# RUN echo hello
COPY docker_test.py /install/sacred/
CMD ["python3", "docker_test.py"]