import json
import argparse
import glob

def xyxy2xywh(x1, y1, x2, y2):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    # y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y = []
    calc1 = (x1 + x2) / 2
    calc2 = (y1 + y2) / 2
    calc3 = x2 - x1
    calc4 = y2 - y1
    y.append(calc1)  # x center
    y.append(calc2) # y center
    y.append(calc3) # width
    y.append(calc4) # height
    return y
    
def create_label_data(classes_path, json_path, images_path, labels_path, images_path_size, body_part):
    data, classes, anno_id = {}, {}, 0
    with open(classes_path, "r") as fp: 
        for cnt, line in enumerate(fp):
            classes[cnt] = line[:-1]
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    paths = glob.glob(images_path)
    for path in paths: 
        path = path[images_path_size:]
        labels = ''
        for index in range(len(data[path]["objects list"])):
            category = data[path]["objects list"][index]["category"]
            for key, value in classes.items(): 
                if value == category:
                    anno_id = key
                    tl_x = data[path]["objects list"][index]["rects"][body_part]["tl"]["x"]
                    tl_y = data[path]["objects list"][index]["rects"][body_part]["tl"]["y"]
                    br_x = data[path]["objects list"][index]["rects"][body_part]["br"]["x"]
                    br_y = data[path]["objects list"][index]["rects"][body_part]["br"]["y"]
                
                    coco_format = xyxy2xywh(tl_x, tl_y, br_x, br_y)
                
                    labels += str(anno_id) + " " + str(coco_format[0]) + " " + str(coco_format[1]) + " " + str(coco_format[2]) + " " + str(coco_format[3]) + "\n"
                    
            directory = labels_path
            if not os.path.exists(directory):
                os.makedirs(directory)
        with open(labels_path + path[:-3] + "txt", "w+") as f:
            f.write(labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default='data/images/image_train/*.jpg')  
    parser.add_argument('--annos_path', type=str, default='data/image_annos/train.json')
    parser.add_argument('--class_names_path', type=str, default='data/classes.names', help='classes of the objects ')
    parser.add_argument('--labels_path', type=str, default='data/lables')
    
    opt = parser.parse_args()
    
    image_directory_length = len(opt.images_path) - 5 #remove the *.jpg to get the directory to the images
    
    create_label_data(opt.class_names_path,opt.annos_path, opt.images_path, opt.labels_path, image_directory_length, "visible body")