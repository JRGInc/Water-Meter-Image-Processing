__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import cv2
import logging
import numpy as np
import os
import tensorflow as tf
import time as ttime
from common import data_ops, img_ops
from config.tensor import TensorCfg
from machine import inception_v4, yolo_v3
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import concatenate


class Tensor:
    def __init__(
            self,
            core_cfg,
    ):
        logfile = 'januswm'
        self.logger = logging.getLogger(logfile)

        print(tf.__version__)
        tf.keras.backend.clear_session()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(
                    gpu,
                    True
                )

        np.set_printoptions(
            threshold=np.inf,
            precision=None
        )

        self.core_cfg = core_cfg
        self.img_path_dict = core_cfg.get(attrib='img_path_dict')
        self.tensor_cfg = TensorCfg(core_cfg=core_cfg)

    def build_yolo_model(
        self
    ) -> (bool, any, dict):
        build_err = False
        model = None

        yolo_dict = self.tensor_cfg.get(attrib='yolo_dict')

        classes = {}
        with open(yolo_dict['classes'], 'r') as data:
            for ID, name in enumerate(data):
                classes[ID] = name.strip('\n')
        print(classes)

        with open(yolo_dict['anchors']) as f:
            anchors = f.readline()
        anchors = np.array(
            anchors.split(','),
            dtype=np.float32
        )
        anchors = np.reshape(
            a=anchors,
            newshape=(3, 3, 2)
        )

        try:
            # Build Model
            input_layer = tf.keras.layers.Input(shape=[
                yolo_dict['input_size'],
                yolo_dict['input_size'],
                3
            ])
            feature_maps = yolo_v3.create_yolo_v3(
                input_layer=input_layer,
                classes=classes
            )

            bbox_tensors = []
            for index, fm in enumerate(feature_maps):
                bbox_tensor = yolo_v3.decode(
                    conv_output=fm,
                    classes=classes,
                    anchors=anchors,
                    strides=np.array(yolo_dict['strides']),
                    i=index
                )
                bbox_tensors.append(bbox_tensor)

            model = tf.keras.Model(input_layer, bbox_tensors)
            model.load_weights(filepath=yolo_dict['wgts'])

        except Exception as exc:
            build_err = True
            log = 'Failed to build YOLO model.'
            self.logger.error(msg=log)
            self.logger.error(msg=exc)
            print(log)
            print(exc)

        return build_err, model, classes

    def build_incept_model(
            self
    ) -> (bool, any):
        build_err = False
        model = None

        incept_dict = self.tensor_cfg.get(attrib='incept_dict')

        try:
            model = inception_v4.create_inception_v4(
                incept_dict=incept_dict
            )
            model.load_weights(
                filepath=incept_dict['wgts'],
                by_name=False
            )
            # model = tf.keras.models.load_model(incept_dict['mdl'])
            model.summary()

        except Exception as exc:
            build_err = True
            log = 'Failed to detect objects from image data.'
            self.logger.error(msg=log)
            self.logger.error(msg=exc)
            print(log)
            print(exc)

        return build_err, model

    def detect_yolo(
            self,
            model,
            img_orig,
            img_orig_shape,
            img_save: bool,
            img_bbox_url: str,
            classes
    ) -> (bool, list):

        detect_err = False

        yolo_dict = self.tensor_cfg.get(attrib='yolo_dict')

        sorted_bbox = {}
        try:
            img_inverse = cv2.cvtColor(
                src=img_orig,
                code=cv2.COLOR_BGR2RGB
            )

            # Predict Process
            preprocess_err, image_data = img_ops.preprocess_yolo(
                img_orig=np.copy(img_inverse),
                target_size=[
                    yolo_dict['input_size'],
                    yolo_dict['input_size']
                ]
            )
            image_data = image_data[np.newaxis, ...].astype(dtype=np.float32)
            pred_bbox = model.predict(x=image_data)
            pred_bbox = [
                tf.reshape(
                    tensor=x,
                    shape=(-1, tf.shape(input=x)[-1])
                ) for x in pred_bbox
            ]
            pred_bbox = concatenate(
                inputs=pred_bbox,
                axis=0
            )
            postprocess_err, bboxes, best_bboxes = img_ops.postprocess_boxes_yolo(
                pred_bbox=pred_bbox,
                org_img_shape=img_orig_shape[0:2],
                input_size=yolo_dict['input_size'],
                score_threshold=yolo_dict['score_thresh']
            )

            if yolo_dict['rslts'] is not None:
                # image = cv2.cvtColor(
                #     src=image,
                #     code=cv2.COLOR_RGB2BGR
                # )
                draw_bbox_err, img_bbox = img_ops.draw_bbox_yolo(
                    img_orig=img_orig.copy(),
                    img_orig_shape=img_orig_shape,
                    bboxes=best_bboxes,
                    classes=classes
                )

                if img_save:
                    cv2.imwrite(
                        filename=img_bbox_url,
                        img=img_bbox
                    )

            for bbox in best_bboxes:
                coor = np.array(
                    bbox[:4],
                    dtype=np.int32
                )
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(
                    str,
                    coor
                ))
                sorted_bbox[class_name] = [
                    float(score),
                    int(xmin),
                    int(ymin),
                    int(xmax),
                    int(ymax)
                ]

        except Exception as exc:
            detect_err = True
            log = 'Failed to detect objects from image data.'
            self.logger.error(msg=log)
            self.logger.error(msg=exc)
            print(log)
            print(exc)

        return detect_err, sorted_bbox

    def predict_inception(
            self,
            model,
            img_digs,
            img_orig_dtg: str,
            incept_dict: dict
    ) -> [bool, list]:
        """
        Generates and returns predictions for each digit using TensorFlow

        :param model
        :param img_digs
        :param img_orig_dtg: str
        :param incept_dict: dict

        :return pred_err: bool
        :return pred_list: list
        """
        logfile = 'januswm'
        logger = logging.getLogger(logfile)

        timea = ttime.time()

        pred_err = False
        pred_list = [
            img_orig_dtg,
            '',
            'V'
        ]

        try:
            for digit in range(0, 6):

                img_pred_rgb = cv2.cvtColor(img_digs[digit], cv2.COLOR_BGR2RGB)
                img_pred_tensor = tf.convert_to_tensor(img_pred_rgb, dtype=tf.float32)
                img_pred_gray = tf.image.rgb_to_grayscale(img_pred_tensor)
                img_pred_resize = tf.image.resize_with_pad(
                    img_pred_gray,
                    incept_dict['img_tgt_width'],
                    incept_dict['img_tgt_height']
                )
                img_ds = tf.expand_dims(img_pred_resize, 0)

                predictions = model.predict(
                    x=img_ds,
                    batch_size=incept_dict['batch_size']
                )
                for element in predictions:
                    prediction = element.argmax(axis=0)
                    confidence = element[prediction]
                    print('Prediction is {0}, with probability {1}'.
                          format(prediction, confidence))

                    if confidence < 0.80:
                        pred_list[1] += 'R'
                        pred_list[2] = 'I'

                        log = 'Digit {0} prediction rejected due to low confidence.'. \
                            format(digit)
                        logger.info(msg=log)
                        print(log)

                    else:
                        if prediction < 10:
                            pred_list[1] += str(prediction)

                        elif (prediction >= 10) and (prediction < 20):
                            pred_list[1] += 'T'
                            pred_list[2] = 'I'

                            log = 'Digit {0} falls on transition boundary.'.\
                                format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif (prediction >= 20) and (prediction < 30):
                            pred_list[1] += 'N'
                            pred_list[2] = 'I'

                            log = 'Digit {0} is occluded by the needle.'.\
                                format(digit)
                            logger.info(msg=log)
                            print(log)

                    pred_list.append(prediction)
                    pred_list.append(confidence)

            if pred_list[2] == 'V':
                log = 'Prediction value has valid digits.'
                logger.info(msg=log)
                print(log)

            else:
                log = 'Prediction value has one or more invalid digits.'
                logger.info(msg=log)
                print(log)

            log = 'Successfully predicted digit values from image data.'
            logger.info(msg=log)
            print(log)
            print(pred_list)

        except Exception as exc:
            pred_err = True
            log = 'Failed to predict digit values from image data.'
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

        if not pred_err:
            img_olay_text = '          Value: '
            for digit in range(5, -1, -1):
                if digit > 0:
                    img_olay_text += str(pred_list[1][5 - digit]) + '-'
                else:
                    img_olay_text += str(pred_list[1][5 - digit])

            if pred_list[2] == 'V':
                img_olay_text += '     (valid)'
            elif pred_list[2] == 'I':
                img_olay_text += '     (invalid)'

        else:
            img_olay_text = '           Value: Prediction Error'

        print('Total prediction time elapsed: {0} sec'.format(ttime.time() - timea))

        return pred_err, pred_list, img_olay_text
