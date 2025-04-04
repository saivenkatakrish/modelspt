import argparse
import os
import sys
import torch
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (check_img_size, non_max_suppression, scale_boxes, strip_optimizer, colorstr)
from utils.plots import Annotator
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(weights, source, imgsz, conf_thres, iou_thres, device, view_img, save_txt, save_conf, save_crop, nosave, project, name, exist_ok):
    # Load models
    models = {
        "drone": DetectMultiBackend(weights[0], device=device, dnn=False),
        "bird": DetectMultiBackend(weights[1], device=device, dnn=False),
        "aeroplane": DetectMultiBackend(weights[2], device=device, dnn=False)
    }
    
    device = select_device(device)
    stride, names, pt = models["drone"].stride, models["drone"].names, models["drone"].pt
    imgsz = check_img_size(imgsz, s=stride)
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.unsqueeze(0) if img.ndimension() == 3 else img
        
        results = {}
        for model_name, model in models.items():
            pred = model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            results[model_name] = pred
            
        for model_name, pred in results.items():
            for det in pred:
                annotator = Annotator(im0s, line_width=3, example=names)
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
                    for *xyxy, conf, cls in det:
                        label = f'{model_name}: {names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label)

        if not nosave:
            save_path = str(save_dir / Path(path).name)
            print(f"Saving result to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=['modelspt/drones.pt', 'modelspt/bird.pt', 'modelspt/aeroplane.pt'], help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--imgsz', '--img', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in .txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    run(**vars(opt))
