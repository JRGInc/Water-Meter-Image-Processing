#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:07:27
#   Description :
#
# ================================================================

import cv2
import numpy as np
import os
import tensorflow as tf
from common import data_ops, img_ops
from config.core import CoreCfg
from config.tensor import TensorCfg
from machine.yolo_v3 import yolo_v3, decode

if __name__ == '__main__':
    core_cfg = CoreCfg()
    img_path_dict = core_cfg.get(attrib='img_path_dict')

    tensor_cfg = TensorCfg(core_cfg=core_cfg)
    yolo_dict = tensor_cfg.get(attrib='yolo_dict')

    image_path = os.path.join(
        img_path_dict['orig'],
        'irotd_2019-04-16_0418_1000200_sg0001_a337.jpg'
    )
    classes = data_ops.read_class_names_yolo(yolo_dict['classes'])

    input_layer = tf.keras.layers.Input([yolo_dict['input_size'], yolo_dict['input_size'], 3])
    feature_maps = yolo_v3(input_layer)

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = img_ops.image_preprocess_yolo(
        np.copy(original_image),
        [yolo_dict['input_size'],
         yolo_dict['input_size']]
    )
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.load_weights(yolo_dict['wgts'])
    model.summary()

    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = img_ops.postprocess_boxes_yolo(
        pred_bbox,
        original_image_size,
        yolo_dict['input_size'],
        yolo_dict['score_thresh']
    )
    bboxes = img_ops.nms_yolo(bboxes, yolo_dict['iou_thresh'], method='nms')
    print(bboxes)

    image = img_ops.draw_bbox_yolo(original_image, bboxes, classes)
    # image = Image.fromarray(image)
    # image.show()
