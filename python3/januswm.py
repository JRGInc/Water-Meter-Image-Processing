#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import logging
import os
import tensorflow as tf
import time
from common import img_ops
from config.core import CoreCfg
from config.tensor import TensorCfg
from machine.tensor import Tensor
from machine import yolo_v3

if __name__ == '__main__':
    logfile = 'januswm-capture'
    logger = logging.getLogger(logfile)

    timea = time.time()

    # Error values dictionary
    err_vals_dict = {
        'build_yolo': True,
        'detection': True,
        'img_bbox': True,
        'img_angle': True,
        'img_rotd': True,
        'img_digw': True,
        'img_digs': True,
        'img_olay': True,
        'prediction': True,
    }

    # Load configuration settings
    core_cfg = CoreCfg()
    img_path_dict = core_cfg.get(attrib='img_path_dict')

    tensor_cfg = TensorCfg(core_cfg=core_cfg)
    dig_dict = tensor_cfg.get(attrib='dig_dict')
    yolo_dict = tensor_cfg.get(attrib='yolo_dict')
    incept_dict = tensor_cfg.get(attrib='incept_dict')

    tensor = Tensor(core_cfg=core_cfg)
    err_vals_dict['build_yolo'], yolo_model, yolo_classes = tensor.build_yolo_model()

    bboxes = None
    if not err_vals_dict['build_yolo']:
        for img_orig_name in sorted(os.listdir(img_path_dict['orig'])):

            img_orig_url = os.path.join(
                img_path_dict['orig'],
                img_orig_name
            )
            img_bbox_url = os.path.join(
                img_path_dict['bbox'],
                'bbox_' + img_orig_name[6::]
            )
            img_grotd_url = os.path.join(
                img_path_dict['grotd'],
                'grotd_' + img_orig_name[6::]
            )
            img_frotd_url = os.path.join(
                img_path_dict['frotd'],
                'frotd_' + img_orig_name[6::]
            )
            print(img_frotd_url)

            err_vals_dict['detection'], img_orig, bbox_dict = tensor.detect_yolo(
                model=yolo_model,
                img_orig_url=img_orig_url,
                img_bbox_url=img_bbox_url,
                classes=yolo_classes
            )
            print('Detection error: {0}'.format(err_vals_dict['detection']))

            img_ang_list = []
            if not err_vals_dict['detection']:
                err_vals_dict['img_angle'], img_ang_list = img_ops.find_angles(
                    img_orig_url=img_orig_url,
                    bbox_dict=bbox_dict,
                )
                print('Angle error: {0}'.format(err_vals_dict['img_angle']))
                print(img_ang_list)

            img_rotd = None
            if not err_vals_dict['img_angle']:
                err_vals_dict['img_rotd'], img_rotd = img_ops.rotate(
                    img_orig_url=img_orig_url,
                    img_grotd_url=img_grotd_url,
                    img_frotd_url=img_frotd_url,
                    img_ang_list=img_ang_list,
                )
                print('Rotation error: {0}'.format(err_vals_dict['img_rotd']))
    #
    # # Crop to individual digits if no gray-scale error
    # img_digw = None
    # if not err_vals_dict['img_rotd']:
    #
    #     # Crop close to digit window, leave some space for differences in zoom
    #     img_digw, err_vals_dict['img_digw'] = img_ops.crop_rect(
    #         img_rotd=img_rotd,
    #         img_rect_url=img_url_dict['rect'],
    #         img_digw_url=img_url_dict['digw'],
    #         tf_dict=tf_dict
    #     )
    #     print('Digit window error: {0}'.format(err_vals_dict['img_digw']))
    #
    # if not err_vals_dict['img_digw']:
    #     err_vals_dict['img_digs'] = img_ops.crop_digits(
    #         img_digw=img_digw,
    #         img_digw_url=img_url_dict['digw'],
    #         img_path_dict=img_path_dict,
    #         tf_dict=tf_dict,
    #         mode_str='pred',
    #     )
    #     print('Crop digits error: {0}'.format(err_vals_dict['img_digs']))
    #
    # img_olay_text = 'Date & Time: ' + \
    #     img_orig_dtg.split('_')[0] + \
    #     ' ' + img_orig_dtg.split('_')[1]
    #
    #
    # # Execute TensorFlow prediction
    # timeb = time.time()
    #
    # # Only import this library if predictions are enabled and
    # # image is successfully converted to numpy array
    #
    # err_vals_dict['pred_vals'], pred_list, img_olay_text_values = tensor.predict_inception(
    #     img_seq=img_seq,
    #     img_digw_url=img_url_dict['digw'],
    #     img_orig_dtg=img_orig_dtg,
    #     tf_dict=tf_dict
    # )
    # img_olay_text = img_olay_text + img_olay_text_values
    # print('Prediction time elapsed: {0} sec'.format(time.time() - timeb))
    #
    # # Overlay image with date-time stamp and value if
    # # no TensorFlow error.
    # # if not err_vals_dict['pred_vals']:
    # else:
    #     img_olay_text = img_olay_text + '           Prediction Not Enabled'
    #
    # if not err_vals_dict['img_digw']:
    #     err_vals_dict['img_olay'] = img_ops.overlay(
    #         img_digw_url=img_url_dict['digw'],
    #         img_olay_url=img_url_dict['olay'],
    #         img_olay_text=img_olay_text
    #     )
    #     print('Overlay error: {0}'.format(err_vals_dict['img_olay']))

    print('Total processing time elapsed: {0} sec'.format(time.time() - timea))
