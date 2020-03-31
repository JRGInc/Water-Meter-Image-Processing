#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import cv2
import logging
import os
import time
from common import img_ops
from config.core import CoreCfg
from config.tensor import TensorCfg
from machine.tensor import Tensor

if __name__ == '__main__':
    logfile = 'januswm-capture'
    logger = logging.getLogger(logfile)

    timea = time.time()

    # Error values dictionary
    err_vals_dict = {
        'shape': True,
        'build_yolo': True,
        'build_incept': True,
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
    err_vals_dict['build_incept'], incept_model = tensor.build_incept_model()

    bboxes = None
    if not err_vals_dict['build_yolo']:
        img_orig_names = iter(sorted(os.listdir(img_path_dict['orig'])))
        for img_orig_name in img_orig_names:

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
            img_rect_url = os.path.join(
                img_path_dict['rect'],
                'rect_' + img_orig_name[6::]
            )
            img_digw_url = os.path.join(
                img_path_dict['digw'],
                'digw_' + img_orig_name[6::]
            )
            img_inv_url = os.path.join(
                img_path_dict['inv'],
                'inv_' + img_orig_name[6::]
            )
            img_olay_url = os.path.join(
                img_path_dict['olay'],
                'olay_' + img_orig_name[6::]
            )

            if os.path.isfile(path=img_orig_url):
                img_save = True
                img_orig = cv2.imread(filename=img_orig_url)
                print(img_orig.shape)
                img_orig_shape = img_orig.shape

                if (img_orig.shape[0] == 2464) and (img_orig.shape[1] == 3280):
                    pass
                elif (img_orig.shape[0] == 1536) and (img_orig.shape[1] == 1536):
                    pass
                else:
                    continue

                err_vals_dict['detection'], bbox_dict = tensor.detect_yolo(
                    model=yolo_model,
                    img_orig=img_orig,
                    img_orig_shape=img_orig_shape,
                    img_save=img_save,
                    img_bbox_url=img_bbox_url,
                    classes=yolo_classes
                )
                print('Detection error: {0}'.format(err_vals_dict['detection']))

                img_ang_list = []
                if not err_vals_dict['detection']:
                    err_vals_dict['img_angle'], img_ang_list = img_ops.find_angles(
                        bbox_dict=bbox_dict,
                    )
                    print('Angle error: {0}'.format(err_vals_dict['img_angle']))
                    print(img_ang_list)

                img_rotd = None
                if not err_vals_dict['img_angle']:
                    err_vals_dict['img_rotd'], img_rotd = img_ops.rotate(
                        img_orig=img_orig,
                        img_save=img_save,
                        img_grotd_url=img_grotd_url,
                        img_frotd_url=img_frotd_url,
                        img_orig_shape=img_orig_shape,
                        img_ang_list=img_ang_list,
                    )
                    print('Rotation error: {0}'.format(err_vals_dict['img_rotd']))

                # Crop to individual digits if no gray-scale error
                img_digw = None
                if not err_vals_dict['img_rotd'] and img_rotd is not None:

                    # Crop close to digit window, leave some space for differences in zoom
                    err_vals_dict['img_digw'], img_digw = img_ops.crop_rect(
                        img_rotd=img_rotd,
                        img_save=img_save,
                        img_rect_url=img_rect_url,
                        img_digw_url=img_digw_url
                    )
                    print('Digit window error: {0}'.format(err_vals_dict['img_digw']))

                img_digs = None
                if not err_vals_dict['img_digw'] and img_digw is not None:
                    err_vals_dict['img_digs'], img_digs = img_ops.crop_digits(
                        img_digw=img_digw,
                        img_save=img_save,
                        img_digw_url=img_digw_url,
                        img_inv_url=img_inv_url,
                        img_path_dict=img_path_dict
                    )
                    print('Crop digits error: {0}'.format(err_vals_dict['img_digs']))

                img_orig_dtg = str(img_orig_name).split('_')[1] + ' ' + \
                    str(img_orig_name).split('_')[2]
                img_olay_text = 'Date & Time: ' + img_orig_dtg
                if not err_vals_dict['img_digs'] and img_digs is not None:

                    # Execute TensorFlow prediction
                    timeb = time.time()

                    # Only import this library if predictions are enabled and
                    # image is successfully converted to numpy array

                    err_vals_dict['pred_vals'], pred_list, img_olay_text_values = tensor.predict_inception(
                        model=incept_model,
                        img_digs=img_digs,
                        img_orig_dtg=img_orig_dtg,
                        incept_dict=incept_dict
                    )
                    img_olay_text = img_olay_text + img_olay_text_values
                    print('Prediction time elapsed: {0} sec'.format(time.time() - timeb))

                    # Overlay image with date-time stamp and value if
                    # no TensorFlow error.
                    if not err_vals_dict['pred_vals'] and img_digw is not None:
                        err_vals_dict['img_olay'] = img_ops.overlay(
                            img_orig_shape=img_orig_shape,
                            img_digw=img_digw,
                            img_olay_url=img_olay_url,
                            img_olay_text=img_olay_text
                        )
                        print('Overlay error: {0}'.format(err_vals_dict['img_olay']))

            else:
                img_ang_err = True
                log = 'OS failed to locate image {0} to rotate.'. \
                    format(img_orig_url)
                logger.error(msg=log)
                print(log)

    print('Total processing time elapsed: {0} sec'.format(time.time() - timea))
