import argparse

from converting.make_json import *
from utils.datasets import *
from utils.utils import *
from utils.torch_utils import *

from utils.layers import *
from model.models import *

fps_count_g = []
bbox_list = []
conf_list = []

def init(out, device):
    device_2 = torch_utils.select_device(device='cpu' if ONNX_EXPORT else device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    return device_2

def load_weights(weights, model, device):
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

def second_stage_classifier(classify=False):
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()
    return classify

def half_precision(model, device):
    half = device.type != 'cpu' 
    if half: model.half()
    return half

def set_dataloader(source, imgsz):
    return LoadImages(source, img_size=imgsz), True

def get_bbox_data():
    return ["person"], [[0,0,0]]

def prediction(img, model, augment):
    start_t = torch_utils.time_synchronized()
    pred = model(img, augment=augment)[0]
    end_t = torch_utils.time_synchronized()
    return pred, (1/(end_t - start_t))

def set_img(img, device, half):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def init_img(imgsz, device):
    return torch.zeros((1, 3, imgsz, imgsz), device=device)

def validate_exported_model(f):
    import onnx
    model = onnx.load(f)  # Load the ONNX model
    onnx.checker.check_model(model)  # Check that the IR is well formed
    return

def export_mode(model, imgsz):
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'], output_names=['classes', 'boxes'])
        validate_exported_model(f)

def print_results(det, names):
    s = ""
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        if int(c) == 0:
            s += '%g %ss, ' % (n, names[int(c)])  # add to string
        else: 
            continue
    return s

def write_results(det, save_txt, gn, path, save_path, save_img, view_img, im0, colors, names):
    for *xyxy, conf, cls in reversed(det):
        conf_list.append(conf)
        if save_txt:  
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
            bbox_list.append((xywh, path, conf))
            if type(conf.tolist()) == "float":
                conf_list.append(conf.tolist())   
            with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                file.write(('%g ' * 5 + '\n') % (cls, *xywh)) 

        if save_img or view_img:
            if int(cls) == 0:
                label = '%s %.2f' % (names[int(cls)], conf*100)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
            else: 
                continue
    return conf_list, bbox_list

def save_results(save_img, dataset, save_path, im0, vid_cap):
    if save_img:
        if dataset.mode == 'images':
            cv2.imwrite(save_path, im0)
        else:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
            vid_writer.write(im0)

def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size 
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    device = init(out, opt.device)
    model = Darknet(opt.cfg, imgsz)
    load_weights(weights, model, device)
    classify = second_stage_classifier(False)
    model.to(device).eval()
    export_mode(model, imgsz)
    half = half_precision(model, device)
    dataset, save_img = set_dataloader(source, imgsz)
    names, colors = get_bbox_data()
    
    img = init_img(imgsz, device)
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  
    for path, img, im0s, vid_cap in dataset:
        img = set_img(img, device, half)
        pred, time_needed = prediction(img, model, opt.augment)
        fps_count_g.append(time_needed)
        if half: pred = pred.float()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
        if classify: pred = apply_classifier(pred, model, img, im0s)
        for i, det in enumerate(pred): 
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                print_results(det, names)
                write_results(det, save_txt, torch.tensor(im0.shape)[[1, 0, 1, 0]], path, save_path, save_img, view_img, im0, colors, names)
            save_results(save_img, dataset, save_path, im0, vid_cap)
        plot_data(path, fps_count_g, conf_list)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/classes.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3_last.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/images/valid', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='data/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.075, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt', default="data/output")
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)
    opt.names = check_file(opt.names)  
    with torch.no_grad():
        detect()
        #make_json(opt.source, bbox_list)
        print("AVG FPS: ", sum(fps_count_g)/len(fps_count_g))
        print("AVG CONF: ", (sum(conf_list)/len(conf_list)).tolist())
