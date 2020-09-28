import json, glob
from PIL import Image
import torch 

def get_id(source, path):
    found = []
    data = json.load(open("<path_to_your_valid_coco.json"))
    #images = glob.glob(source + "/*.jpg")
    key = path.split("/")[-1]
    for item in data["images"]:
        if key in item["file_name"]:
            print(item["id"])
            return item["id"]

def make_json(source, bbox_list): 
    data, new_input = [], {}
    Image.MAX_IMAGE_PIXELS = None
    for bbox, path, conf in bbox_list:
        size = Image.open(path).size
        heigth, width = size[0], size[1]
        new_input["image_id"] = get_id(source, path)
        new_input["bbox"] = [bbox[0]*width, bbox[1]*heigth, bbox[2]*width, bbox[3]*heigth]
        new_input["category_id"] = 1
        new_input["score"] = conf.tolist()
        data.append(new_input)
        new_input = {}

    data_str = json.dumps(data, indent=4)
    with open("result.json", "w+") as f: 
        f.write(data_str)
    