# Project-Template for IW276 Autonome Systeme Labor

Short introduction to project assigment.

<p align="center">
  Screenshot / GIF <br />
  Link to Demo Video
</p>

> This work was done by Christian Z. (zach1011), Fernando C. P. (pafe1011), Raphaele S. L. (lira1011) during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

This repo contains inference and training code for YOLOv3 in PyTorch based on the Repository of Ultralytics https://github.com/ultralytics/yolov3

This code is adjusted to projects case in which we had to detect a large number of persons on images and videos
This means the configurations of Yolov3 (the files in the cfg folder) are adjusted to only use one class of objects 
and the classes file (classes.names in the data folder) only contains the person.class


## Table of Contents

* [Prerequisites](#prerequisites)
* [Running](#running)
* [Custom Training](#custom training)
* [Acknowledgments](#acknowledgments)

## Prerequisites
1. Install requirements:
```
pip install -r requirements.txt
```

## Custom Training

**Prepare Training** TODO

** Pretrained Checkpoints

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)

I recommend yolov3-spp-ultralytics.pt for YOLOv3 and yolov3-tiny.pt for Tiny YOLOv3


**Start Training:** `python3 train.py` to begin training after perparing training and validation data

**Resume Training:** `python3 train.py --resume` to resume training from `weights/last.pt` (or `weights/last_tiny.pt` if --tiny is set).

**Results** TODO 

**Plot Training:** `from utils import utils; utils.plot_results()`

## Running

To run the demo, pass path to the pre-trained checkpoint and camera id (or path to video file):
```
python src/demo.py --model model/student-jetson-model.pth --video 0
```
> Additional comment about the demo.

## Docker
HOW TO

## Acknowledgments

This repo is based on
  - [Ultralytics](https://github.com/ultralytics/yolov3)
 
Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.





