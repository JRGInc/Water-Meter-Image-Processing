#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-07-19 10:29:34
#   Description :
#
# ================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from common import data_ops, img_ops
from config.core import CoreCfg
from config.tensor import TensorCfg
from machine.yolo_v3 import create_yolo_v3, decode

if __name__ == '__main__':
    core_cfg = CoreCfg()
    img_path_dict = core_cfg.get(attrib='img_path_dict')

    tensor_cfg = TensorCfg(core_cfg=core_cfg)
    yolo_dict = tensor_cfg.get(attrib='yolo_dict')

    NUM_CLASS = len(data_ops.read_class_names_yolo(yolo_dict['classes']))
    CLASSES = data_ops.read_class_names_yolo(yolo_dict['classes'])

    predicted_dir_path = './results/test/predicted/'
    ground_truth_dir_path = './results/test/ground-truth/'
    if os.path.exists(predicted_dir_path):
        shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path):
        shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(yolo_dict['rslts']):
        shutil.rmtree(yolo_dict['rslts'])

    os.mkdir(yolo_dict['rslts'])
    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)

    # Build Model
    input_layer = tf.keras.layers.Input([yolo_dict['input_size'], yolo_dict['input_size'], 3])
    feature_maps = create_yolo_v3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.load_weights(yolo_dict['wgts'])

    for img_orig_name in sorted(os.listdir(img_path_dict['orig'])):
        img_orig_name_base = img_orig_name.split(sep='.')[0]
        img_orig_url = os.path.join(
            img_path_dict['orig'],
            img_orig_name
        )
        image = cv2.imread(img_orig_url)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print('=> predict result of %s:' % img_orig_url)
        predict_result_path = os.path.join(predicted_dir_path, str(img_orig_name_base) + '.txt')
        # Predict Process
        image_size = image.shape[:2]
        image_data = img_ops.image_preprocess_yolo(np.copy(image), [yolo_dict['input_size'], yolo_dict['input_size']])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = img_ops.postprocess_boxes_yolo(
            pred_bbox,
            image_size,
            yolo_dict['input_size'],
            yolo_dict['score_thresh']
        )
        bboxes = img_ops.nms_yolo(bboxes, yolo_dict['iou_thresh'], method='nms')

        if yolo_dict['rslts'] is not None:
            classes = data_ops.read_class_names_yolo(yolo_dict['classes'])
            image = img_ops.draw_bbox_yolo(image, bboxes, classes)
            cv2.imwrite(yolo_dict['rslts'] + img_orig_name, image)

        with open(predict_result_path, 'w') as f:
            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = CLASSES[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
