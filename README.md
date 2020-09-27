# Project-Template for IW276 Autonome Systeme Labor

Person Detection.

<p align="center">
  <img src="Detect/data/demo_images/image_2.jpg"/>
</p>

> This work was done by Christian Z. (zach1011), Fernando C. P. (pafe1011), Raphaele S. L. (lira1011) during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

This repo contains inference and training code for YoloV3 and YoloV3-Tiny in PyTorch based on the Repository of <a url="https://github.com/ultralytics/yolov3">Ultralytics</a> 

This code is adjusted to the projects case in which we had to detect a large number of persons on images and videos
This means the configurations of YOLOv3 (the files in the cfg folder) are adjusted to only use one class of objects
and the classes file (classes.names in the data folder) only contains the person class


## Table of Contents

* [Prerequisites](#prerequisites)
* [Running](#running)
* [Docker](#docker)
* [Training](#training)
* [Acknowledgments](#acknowledgments)

## Prerequisites
Install requirements:
```bash 
$ pip install -r requirements.txt
```
Start Docker Daemon on Jetson Nano
```bash
$ sudo service docker start
```

## Running

Descriptions about the main Topics of this Project, e. g. the Training Part of our A.I.

### Docker 

This Project doesn't own a mounted Folder for the Docker Container. So if you start this Project in the Docker Container and want to watch the Output Pictures of the Detection (e. g. the BBoxes), you need to do the following steps:
1. Copy the Images from your Docker to your Host Machine (here: Jetson Nano via SSH)
```bash
sudo docker ps
sudo docker cp <container_id>:<docker_output_folder> <host_target_folder>
```
2. If you run it with the Jetson Nano via SSH and now want to Copy those Images to your Running Working Machine you need to do following Command:
```bash
scp -r <name>@<ip>:<remote_output_folder> <local_target_folder>
```

**Build Container** To build the Docker Container (on the Jetson Nano) run the following Command.
```bash
sudo docker build --no-cache . -t image_name_neu
```

**Run Container** To run the Docker Container run the following Command.
```bash
sudo docker run -it --rm --runtime nvidia --network host image_name_neu
```


### Detection

<p align="center">
  <img src="Detect/data/demo_images/image_1.jpg"/>
</p>

This part of the Project, doesn't need any big Changes. 

1. Download all the Files which are not in this Part of the Repository Included. (If you go with Docker, the Dockerfile make the following Command by it self - Jump to 2.)
```bash
$ bash get_remaining_data.sh
```

2. Now you own every File, which are necessarly for the `detect.py`. If you downloaded all files correctly you should be able to start the default YoloV3 Person Detection with the following command:
```bash
$ bash detect.py 
```
If there are any mistakes, you want to set your own files or you want to start the YoloV3-Tiny Person Detection, you have to do it his way:
```bash
$ bash detect.py --cfg <path_to_config_file> --weights <path_to_weights_file> --names <path_to_names_file> --source <path_to_image_folder> --output <path_to_output_folder>
```
You are able to use more Arguments, just watch Line 158 - 174 in the `detect.py` File.

### Training

**Prepare Training**

todo

**Pretrained Weights**

We absolutely recommend starting your training with already pretrained weights. Even if your case might be different from the usual detection you should use pretrained weights.
Some of those weights already have many weeks of training behind them and will definitely improve your overall detection rates.

For YoloV3 yolov3-spp-ultralytics.pt is a really well trained weight file and yolov3-tiny.pt for Tiny YOLOv3

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)

> By the way: pt stands for pytorch. With our code you can also use .weights file but we decided to use Pytorch instead of Darknet, so we rather use pt files and also save the resulting weights as pt files. 


** Prepare your Custom Data

For this you should do the 7 steps in the chapter "Train On Custom Data" of the following wiki page. 
But before you start we want to mention that we created two small scripts to make some of the steps easier. 
The first one is `create_image_list.py`. It takes paths to files  as well as the resulting filename as arguments and lists all the found files.
e.g.
```bash
$ python create_image_list.py --images_path data\images\image_train\*.jpg --file_name data\train.txt
```

https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data

**Start Training:** 
To begin training after preparing training and validation data
```bash
$ python3 train.py --cfg <your_cfg_path> --weight <pretrained_weight_path>` 
```
default cfg is cfg/yolov3-spp.cfg (or cfg/yolov3-tiny.cfg if --tiny is set) so if you just adjusted this cfg to your requirements then you don't have to add --cfg

**Resume Training:** 
to resume training from `weights/last.pt` (or `weights/last_tiny.pt` if --tiny is set).
```bash
$ python3 train.py --resume --cfg <your_cfg_path>`
```

Again default cfg is cfg/yolov3-spp.cfg (or cfg/yolov3-tiny.cfg if --tiny is set)

>In those Pytorch files the number of epoches run and the currently best evaluation is also saved, so if for some reason the training is stopped, 
you can easily continue from where the training left of with --resume. 

**Results** after each epoch last.pt (or last_tiny.pt) will be updated and if the evaluation of the current neural net is the best best.pt will be overwritten.
Also a result.txt if created in whích for each evaluation the results are saved. If you want to analyze the evaluation data we recommend plotting with the following commands.

```bash
from utils import utils
utils.plot_results()
```

The results of the plotting will be saved in a results.png file.

If you want to learn how the object detection is evaluated we recommend taking a look at this site: https://medium.com/analytics-vidhya/understanding-the-map-mean-average-precision-evaluation-metric-for-object-detection-432f5cca53b7

## Acknowledgments

This repo is based on
  - [Ultralytics - YOLOv3](https://github.com/ultralytics/yolov3)
  - [Eriklindernoren - PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
 
Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.





