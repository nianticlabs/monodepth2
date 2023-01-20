FROM ubuntu:18.04
#RUN rm /bin/sh && ln -s /bin/bash /bin/sh
#RUN apt update && apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
RUN apt update && apt install pip

#FROM continuumio/miniconda3
FROM continuumio/anaconda3

RUN conda create -n py36 python=3.6 anaconda
#RUN source ~/anaconda3/etc/profile.d/conda.sh
#RUN conda init bash
RUN activate py36

RUN conda install -n py36 pytorch=0.4.1 torchvision=0.2.1 -c pytorch
RUN conda install -n py36 opencv=3.3.1 

RUN pip install tensorboardX==1.4

RUN apt install git

RUN git clone https://github.com/nianticlabs/monodepth2.git

#RUN source activate py36
