# coding:utf-8
# ----------------------------------------------------------------------------
# Pytorch multi-GPU YOLOV5 based UMT
# Licensed under The MIT License [see LICENSE for details]
# Written by Huayi Zhou, based on code from
# https://github.com/kinredon/umt
# https://github.com/ultralytics/yolov5
# ----------------------------------------------------------------------------

"""Test a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/test.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import json
import os
import sys
import torch
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from threading import Thread

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
# from utils.datasets import create_dataloader
from utils.datasets_single import create_dataloader_single
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, \
    check_dataset_umt


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        # task='val',  # train, val, test, speed or study
        task='test_target_real',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a cocoapi-compatible JSON results file
        project='runs/test',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        wandb_logger=None,
        compute_loss=None,
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        with open(data) as f:
            data = yaml.safe_load(f)
        # check_dataset(data)  # check
        check_dataset_umt(data)  # check, need to be re-write or command out

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    # is_coco = type(data['val']) is str and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    is_coco = type(data['test_target_real']) is str and data['test_target_real'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        # task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        task = task if task in ('train_source_real', 'train_source_fake', 'train_target_real', 
            'train_target_fake', 'test_target_real') else 'test_target_real'
        dataloader = create_dataloader_single(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    coco91class = coco80_to_coco91_class()
    results_dict = {}
    jdict, stats = [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        out, train_out = model(img, augment=augment)  # inference and training outputs

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        
        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_json:
                image_id = int(path.stem)+1 if path.stem.isnumeric() else path.stem
                box = predn[:, :4]
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                # for p, b in zip(pred.tolist(), box.tolist()):
                #     jdict.append({  image_id:
                #                   {
                #                     'box': [round(x, 3) for x in b],
                #                     'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                #                     'score': round(p[4], 5)
                #                   }
                #                     })
                for p,b in zip(pred.tolist() ,box.tolist()):
                    # 取得檔案名稱，例如 "fog/public_test/2934.png"
                    filename = f"{image_id}.png"
                    # 初始化或取得這個檔案對應的資料
                    if filename not in results_dict:
                        results_dict[filename] = {
                            "boxes": [],
                            "labels": [],
                            "scores": []
                        }
                    # print([round(x, 3) for x in b])
                    # print(coco91class[int(p[5])] if is_coco else int(p[5]))
                    # print(round(p[4], 5))
                    # 將這個預測框的資訊加入到對應的資料中
                    results_dict[filename]["boxes"].append([round(x, 3) for x in b])
                    results_dict[filename]["labels"].append(coco91class[int(p[5])] if is_coco else int(p[5]))
                    results_dict[filename]["scores"].append(round(p[4], 5))

        


    # Save JSON
    if save_json and len(results_dict):

        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nsaving json file in %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(results_dict, f)


def parse_opt():
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    # print(colorstr('test: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
