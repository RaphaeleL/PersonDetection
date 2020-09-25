This repo contains Inference and training code for YOLOv3 in PyTorch based on the Repository of Ultralytics https://github.com/ultralytics/yolov3

This code is adjusted to out projects case in which we had to detect a large number of persons.
This means the configurations of Yolov3 (the files in the cfg folder) are adjusted to only use one class of objects 
and the classes file (classes.names in the data folder) only contains the person.class

## Requirements

All necessary requirements are listed in the requirements.txt. To install them execute :
```bash
$ pip install -r requirements.txt
```

## Training

**Prepare Training** TODO

** Pretrained Checkpoints

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)

I recommend yolov3-spp-ultralytics.pt for YOLOv3 and yolov3-tiny.pt for Tiny YOLOv3


**Start Training:** `python3 train.py` to begin training after perparing training and validation data

**Resume Training:** `python3 train.py --resume` to resume training from `weights/last.pt` (or `weights/last_tiny.pt` if --tiny is set).

**Results** TODO 

**Plot Training:** `from utils import utils; utils.plot_results()`


