FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3

# JETSON NANO IS NOT ABLE TO HANDLE THE TRAINING. 
# SO PLEASE JUST USE THE DOCKER CONTAINER FOR THE DETECTION PROGRESS.

# Packages
RUN apt-get update && apt-get install vim -y
RUN apt-get install wget unzip -y

# Git
RUN git clone https://github.com/IW276/IW276WS20-P9.git 
WORKDIR /IW276WS20-P9/
RUN git pull 

# Python
RUN pip3 install --upgrade pip
RUN pip3 install tqdm
RUN apt-get install python3-opencv -y

# Large Files
WORKDIR weights/
RUN wget "https://www.dropbox.com/s/nzmibytp2hmv666/yolov3_tiny_best.pt"
RUN wget "https://www.dropbox.com/s/jdhx0ivslj9zlxd/yolov3_best.pt"
RUN wget "https://www.dropbox.com/s/kb79bx1utukho7s/yolov3_last.pt"

WORKDIR ../data/images/valid/
RUN wget "https://www.dropbox.com/s/nwehy278dybo2m8/image_valid.zip"
RUN unzip image_valid.zip
RUN mv image_valid/*.jpg ../valid
RUN rm -rf image_valid.zip __MACOSX image_valid
WORKDIR ../../../


