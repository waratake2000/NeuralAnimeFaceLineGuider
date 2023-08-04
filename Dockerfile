# FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
# docker pull nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=nointeractive

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y git wget vim

RUN apt-get install -y python3-pip
RUN apt-get install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install torch torchvision
# RUN pip3 install torch torchvision torchaudio
RUN pip3 install opencv-python
RUN pip3 install streamlit
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install tqdm
RUN pip3 install seaborn
RUN pip3 install gputil
RUN pip3 install imgaug
RUN pip3 install psutil
RUN pip3 install mlflow
RUN pip3 install torchinfo
RUN pip3 install omegaconf
RUN pip3 install pretrainedmodels
