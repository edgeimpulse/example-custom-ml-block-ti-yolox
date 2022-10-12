import numpy as np
import argparse, math, shutil, os, json, time
from PIL import Image

parser = argparse.ArgumentParser(description='Edge Impulse => YOLOX')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)

args = parser.parse_args()

# Load data (images are in X_*.npy, labels are in JSON in Y_*.npy)
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')

with open(os.path.join(args.data_directory, 'Y_split_train.npy'), 'r') as f:
    Y_train = json.loads(f.read())
with open(os.path.join(args.data_directory, 'Y_split_test.npy'), 'r') as f:
    Y_test = json.loads(f.read())

image_width, image_height, image_channels = list(X_train.shape[1:])

out_dir = args.out_directory
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)

class_count = 0

print('Transforming Edge Impulse data format into something compatible with YOLOX')

def current_ms():
    return round(time.time() * 1000)

total_images = len(X_train) + len(X_test)
zf = len(str(total_images))
last_printed = current_ms()
converted_images = 0

def convert(X, Y, category):
    global class_count, total_images, zf, last_printed, converted_images

    all_images = []
    annotations_file = os.path.join(out_dir, 'annotations', 'instances_' + category + '.json')
    if not os.path.exists(os.path.dirname(annotations_file)):
        os.makedirs(os.path.dirname(annotations_file), exist_ok=True)

    metadata = {
        "info": {
            "year": 2022,
            "version": "1.0",
            "description": "Custom model",
            "date_created": "2022"
        },
        "images": [],
        "licenses": [{
            "id": 1,
            "name": "Proprietary",
            "url": "https://edgeimpulse.com"
        }],
        "type": "instances",
        "annotations": [],
        "categories": [],
    }

    for ix in range(0, len(X)):
        raw_img_data = (np.reshape(X[ix], (image_width, image_height, image_channels)) * 255).astype(np.uint8)
        labels = Y[ix]['boundingBoxes']

        images_dir = os.path.join(out_dir, category)
        os.makedirs(images_dir, exist_ok=True)

        img_file = os.path.join(images_dir, str(ix).zfill(12) + '.jpg')

        all_images.append(img_file)

        im = Image.fromarray(raw_img_data)
        im.save(img_file)

        img_id = len(metadata['images']) + 1

        for l in labels:
            if (l['label'] > class_count):
                class_count = l['label']

            x = l['x']
            y = l['y']
            w = l['w']
            h = l['h']

            metadata['annotations'].append({
                "segmentation": [],
                "area": w * h,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [x, y, w, h],
                "category_id": l['label'],
                "id": len(metadata['annotations']) + 1
            })

        metadata['images'].append({
            "date_captured": "2022",
            "file_name": os.path.basename(img_file),
            "id": img_id,
            "height": image_height,
            "width": image_width
        })

        converted_images = converted_images + 1
        if (converted_images == 1 or current_ms() - last_printed > 3000):
            print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')
            last_printed = current_ms()

    for c in range(0, class_count):
        metadata['categories'].append({
            "id": c + 1,
            "name": str(c),
            "supercategory": str(c)
        })

    with open(annotations_file, 'w') as f:
        f.write(json.dumps(metadata, indent=4))

convert(X=X_train, Y=Y_train, category='train2017')
convert(X=X_test, Y=Y_test, category='val2017')

print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')

print('Transforming Edge Impulse data format into something compatible with YOLOX OK')
print('')

img_tuple = "(" + str(image_width) + ", " + str(image_height) + ")"

cfg = """#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = """ + img_tuple + """
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = """ + img_tuple + """
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.max_epoch = """ + str(args.epochs) + """
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = "relu"
        self.num_classes = """ + str(class_count) + """

    def get_model(self, sublinear=False):
        from yolox.utils import freeze_module

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, depthwise=False, conv_focus=True, split_max_pool_kernel=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act, depthwise=False)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        freeze_module(self.model.backbone.backbone)
        return self.model
"""

with open(os.path.join(out_dir, 'custom_nano_ti_lite.py'), 'w') as f:
    f.write(cfg)
