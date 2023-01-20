FROM ubuntu:18.04

RUN apt update && apt install pip git

FROM continuumio/anaconda3

RUN conda create -n py36 python=3.6 anaconda
RUN activate py36

RUN conda install -n py36 pytorch=0.4.1 torchvision=0.2.1 -c pytorch
RUN conda install -n py36 opencv=3.3.1 

RUN pip install tensorboardX==1.4


RUN git clone https://github.com/nianticlabs/monodepth2.git
RUN mkdir /images

# how to use:

# docker build . -t monodepth2
# docker run --rm -it --entrypoint=/bin/bash -v ~/your_image_folder_path:/images monodepth2

# then, inside docker container:
# source activate py36
# cd monodepth2
# python test_simple.py --image_path ../images/your_image.jpg --model_name mono+stereo_640x192
