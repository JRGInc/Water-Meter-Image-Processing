#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import cv2
import logging
import os
from common import img_ops
from config.core import CoreCfg
from machine.tensor import Tensor

if __name__ == '__main__':

    # Setup logging
    logfile = 'januswm'
    logger = logging.getLogger(logfile)

    # Error values dictionary
    err_dict = {
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

    # Setup TensorFlow models
    tensor = Tensor(core_cfg=core_cfg)
    err_dict['build_yolo'], yolo_model, yolo_classes = tensor.build_yolo_model()
    print('Build YOLO error: {0}'.format(err_dict['build_yolo']))
    err_dict['build_incept'], incept_model = tensor.build_incept_model()
    print('Build Inception error: {0}'.format(err_dict['build_incept']))

    # Proceed only if model setup successful
    if not err_dict['build_yolo'] and not err_dict['build_incept']:

        # Create sorted list of image names from source directory for iteration
        img_orig_names = iter(sorted(os.listdir(img_path_dict['orig'])))
        for img_orig_name in img_orig_names:

            # Build image urls from image name
            img_orig_url = os.path.join(
                img_path_dict['orig'],
                img_orig_name
            )
            img_bbox_url = os.path.join(
                img_path_dict['bbox'],
                'bbox_' + img_orig_name[5::]
            )
            img_grotd_url = os.path.join(
                img_path_dict['grotd'],
                'grotd_' + img_orig_name[5::]
            )
            img_frotd_url = os.path.join(
                img_path_dict['frotd'],
                'frotd_' + img_orig_name[5::]
            )
            img_rect_url = os.path.join(
                img_path_dict['rect'],
                'rect_' + img_orig_name[5::]
            )
            img_digw_url = os.path.join(
                img_path_dict['digw'],
                'digw_' + img_orig_name[5::]
            )
            img_inv_url = os.path.join(
                img_path_dict['inv'],
                'inv_' + img_orig_name[5::]
            )
            img_olay_url = os.path.join(
                img_path_dict['olay'],
                'olay_' + img_orig_name[5::]
            )

            # Check to determine that file is legitimate
            if os.path.isfile(path=img_orig_url):

                # Should intermediate images be saved?
                img_save = False

                # Read image into OpenCV format
                img_orig = cv2.imread(filename=img_orig_url)

                # Get shape, height = element 0, width = element 1
                img_orig_shape = img_orig.shape

                # If meets criteria, continue execution, if not skip and iterate
                # to next image.
                if (img_orig.shape[0] == 2464) and (img_orig.shape[1] == 3280):
                    pass
                elif (img_orig.shape[0] == 1536) and (img_orig.shape[1] == 1536):
                    pass
                else:
                    continue

                # Perform object detection on image
                err_dict['detection'], bbox_dict = tensor.detect_yolo(
                    model=yolo_model,
                    img_orig=img_orig,
                    img_orig_shape=img_orig_shape,
                    img_save=img_save,
                    img_bbox_url=img_bbox_url,
                    classes=yolo_classes
                )
                print('Detection error: {0}'.format(err_dict['detection']))

                # If object detection does not produce errors, used detected
                # objects to find angle of rotation
                img_ang_list = None
                if not err_dict['detection'] and bbox_dict is not None:
                    err_dict['img_angle'], img_ang_list = img_ops.find_angles(
                        bbox_dict=bbox_dict,
                    )
                    print('Angle error: {0}'.format(err_dict['img_angle']))
                    print(img_ang_list)

                # If finding angles does not produce errors, use angle
                # to rotate image
                img_rotd = None
                if not err_dict['img_angle'] and img_ang_list is not None:
                    err_dict['img_rotd'], img_rotd = img_ops.rotate(
                        img_orig=img_orig,
                        img_save=img_save,
                        img_grotd_url=img_grotd_url,
                        img_frotd_url=img_frotd_url,
                        img_orig_shape=img_orig_shape,
                        img_ang_list=img_ang_list,
                    )
                    print('Rotation error: {0}'.format(err_dict['img_rotd']))

                # If rotation does not produce errors, crop digit window from
                # rotated image
                img_digw = None
                if not err_dict['img_rotd'] and img_rotd is not None:
                    err_dict['img_digw'], img_digw = img_ops.crop_rect(
                        img_rotd=img_rotd,
                        img_save=img_save,
                        img_rect_url=img_rect_url,
                        img_digw_url=img_digw_url
                    )
                    print('Digit window error: {0}'.format(err_dict['img_digw']))

                # If cropping digit window does not produce errors, crop
                # individual digits from digit window
                img_digs = None
                if not err_dict['img_digw'] and img_digw is not None:
                    err_dict['img_digs'], img_digs = img_ops.crop_digits(
                        img_digw=img_digw,
                        img_save=img_save,
                        img_inv_url=img_inv_url,
                        img_path_dict=img_path_dict
                    )
                    print('Crop digits error: {0}'.format(err_dict['img_digs']))

                # Create text for overlaid image
                img_orig_dtg = str(img_orig_name).split('_')[1] + ' ' + \
                    str(img_orig_name).split('_')[2]
                img_olay_text = 'Date & Time: ' + img_orig_dtg

                # If cropping digits does not produce errors, perform
                # TensorFlow predictions on digits
                if not err_dict['img_digs'] and img_digs is not None:

                    #
                    err_dict['prediction'], pred_list, olay_text = tensor.predict_inception(
                        model=incept_model,
                        img_digs=img_digs,
                        img_orig_dtg=img_orig_dtg
                    )
                    img_olay_text = img_olay_text + olay_text

                # Overlay image with date-time stamp and value if
                # no TensorFlow error.
                if not err_dict['prediction'] and img_digw is not None:
                    err_dict['img_olay'] = img_ops.overlay(
                        img_orig_shape=img_orig_shape,
                        img_digw=img_digw,
                        img_olay_url=img_olay_url,
                        img_olay_text=img_olay_text
                    )
                    print('Overlay error: {0}'.format(err_dict['img_olay']))

            # If file is illegitimate, log error, move to next file
            else:
                img_ang_err = True
                log = 'OS failed to locate image {0} to rotate.'. \
                    format(img_orig_url)
                logger.error(msg=log)
                print(log)
