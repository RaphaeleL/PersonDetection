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

# Git
RUN git pull 
RUN Detect/
RUN bash Detect/source/get_remaining_data.sh
