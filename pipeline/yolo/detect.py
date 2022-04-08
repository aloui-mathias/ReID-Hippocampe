# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

from config import *
from .utils.torch_utils import select_device, time_sync
from .utils.plots import Annotator, colors, save_one_box
from .utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from .utils.datasets import LoadImages
from .models.common import DetectMultiBackend
import argparse
from os import mkdir
from pathlib import Path

import torch

@torch.no_grad()
def run(source,  # file/dir/URL/glob, 0 for webcam
        conf_thres,  # confidence threshold
        iou_thres,  # NMS IOU threshold
        max_det,  # maximum detections per image
        device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        ):
    
    source = str(source)
    project = environ["TEMP_FOLDER"]
    name = "yolo"
    weights = environ["YOLO_WEIGHTS"]
    data = environ["YOLO_YAML"]
    imgsz = [640, 640]

    # Directories
    save_dir = Path(project) / name
    mkdir(save_dir)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

 
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=False)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir)  # im.jpg
            s += '%gx%g ' % im.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy()
            annotator = Annotator(
                im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                        ) / gn).view(-1).tolist()  # normalized xywh
                    # label format
                    line = (cls, *xywh, conf)
                    with open(save_path + '/boxes.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    save_one_box(
                        xyxy, imc, file=save_dir / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='image path(s)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))
