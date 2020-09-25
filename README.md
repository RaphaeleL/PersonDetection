# Project-Template for IW276 Autonome Systeme Labor

Short introduction to project assigment.

<p align="center">
  Screenshot / GIF <br />
  Link to Demo Video
</p>

> This work was done by Christian Z. (zach1011), Fernando C. P. (pafe1011), Raphaele S. L. (lira1011) during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

This repo contains inference and training code for YOLOv3 and Tiny Yolov3 in PyTorch based on the Repository of Ultralytics https://github.com/ultralytics/yolov3

This code is adjusted to projects case in which we had to detect a large number of persons on images and videos
This means the configurations of Yolov3 (the files in the cfg folder) are adjusted to only use one class of objects 
and the classes file (classes.names in the data folder) only contains the person.class


## Table of Contents

* [Prerequisites](#prerequisites)
* [Running](#running)
* [Docker](#docker)
* [Training](#training)
* [Acknowledgments](#acknowledgments)

## Prerequisites
1. Install requirements:
```
pip install -r requirements.txt
```

## Running

To run the demo, pass path to the pre-trained checkpoint and camera id (or path to video file):
```
python src/demo.py --model model/student-jetson-model.pth --video 0
```
> Additional comment about the demo.

## Docker
HOW TO

## Training

**Prepare Training** TODO

** Pretrained Weights

We absolutely recommend starting your training with already pretrained weights. Even if your case might be different from the usual detection you should use pretrained weights.
Some of those weights already have many weeks of training behind them and will definitely improve your overall detection rates.

For YoloV3 yolov3-spp-ultralytics.pt is a really well trained weight file and yolov3-tiny.pt for Tiny YOLOv3

By the way: pt stands for pytorch. With our code you can also use .weights file but we decided to use Pytorch instead of Darknet, so we rather use pt files and also save the resulting weights as pt files. 

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)

** Prepare your Custom Data

For this you should do the 7 steps in the chapter "Train On Custom Data" of the following wiki page. 

https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data

**Start Training:** `python3 train.py --cfg <your_cfg_path> --weight <pretrained_weight_path>` to begin training after perparing training and validation data

default cfg is cfg/yolov3-spp.cfg (or cfg/yolov3-tiny.cfg if --tiny is set) so if you just adjusted this cfg to your requirements then you don't have to add --cfg

**Resume Training:** `python3 train.py --resume --cfg <your_cfg_path>` to resume training from `weights/last.pt` (or `weights/last_tiny.pt` if --tiny is set).

again default cfg is cfg/yolov3-spp.cfg (or cfg/yolov3-tiny.cfg if --tiny is set)

In those Pytorch files the number of epoches run and the currently best evaluation is also saved, so if for some reason the training is stopped, 
you can easily continue from where the training left of with --resume. 

**Results** after each epoch last.pt (or last_tiny.pt) will be updated and if the evaluation of the current neural net is the best best.pt will be overwritten.
Also a result.txt if created in whích for each evaluation the results are saved. If you want to analyze the evaluation data we recommend plotting with the following commands.

`from utils import utils; utils.plot_results()`

The results of the plotting will be saved in a results.png file.

If you want to learn how the object detection is evaluated we recommend taking a look at this site: https://medium.com/analytics-vidhya/understanding-the-map-mean-average-precision-evaluation-metric-for-object-detection-432f5cca53b7


## Acknowledgments

This repo is based on
  - [Ultralytics](https://github.com/ultralytics/yolov3)
 
Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.





