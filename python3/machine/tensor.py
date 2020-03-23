__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import cv2
import logging
import numpy as np
import os
import tensorflow as tf
import time as ttime
from common import data_ops, file_ops, img_ops
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

    def detect_yolo(
        self,
        model,
        img_orig_url,
        img_bbox_url,
        classes
    ) -> (bool, list):

        detect_err = False
        yolo_dict = self.tensor_cfg.get(attrib='yolo_dict')

        sorted_bbox = {}
        img_orig = None

        try:
            img_orig = cv2.imread(filename=img_orig_url)
            img_inverse = cv2.cvtColor(
                src=img_orig,
                code=cv2.COLOR_BGR2RGB
            )

            # Predict Process
            image_size = img_inverse.shape[:2]
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
                img_orig_url=img_orig_url,
                pred_bbox=pred_bbox,
                org_img_shape=image_size,
                input_size=yolo_dict['input_size'],
                score_threshold=yolo_dict['score_thresh']
            )

            if yolo_dict['rslts'] is not None:
                # image = cv2.cvtColor(
                #     src=image,
                #     code=cv2.COLOR_RGB2BGR
                # )
                draw_bbox_err, img_bbox = img_ops.draw_bbox_yolo(img_orig, best_bboxes, classes)
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

        return detect_err, img_orig, sorted_bbox

    def predict_inception(
        self,
        img_orig_dtg: str,
        img_digw_url: str,
        img_seq: str,
        tf_dict: dict
    ) -> [bool, list]:
        """
        Generates and returns predictions for each digit using TensorFlow

        :param img_orig_dtg: str
        :param img_digw_url: str
        :param img_seq: str
        :param tf_dict: dict

        :return pred_err: bool
        :return pred_list: list
        """
        logfile = 'januswm-test'
        logger = logging.getLogger(logfile)

        timea = ttime.time()

        pred_err = False

        data_path_dict = self.core_cfg.get('data_path_dict')
        cfg_url_dict = self.core_cfg.get('cfg_url_dict')
        img_path_dict = self.core_cfg.get('img_path_dict')

        predictions = [
            [0, None],
            [0, None],
            [0, None],
            [0, None],
            [0, None],
            [0, None],
        ]
        pred_list = [
            img_orig_dtg,
            '',
            'V'
        ]

        try:
            mdl_url_str = '/opt/Janus/WM/model/tflite/mdl_{0}.tflite'.format(self.core_cfg.get(attrib='train_version'))
            interpreter = tf.lite.Interpreter(model_path=mdl_url_str)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            for digit in range(5, -1, -1):
                timeb = ttime.time()
                img_pdig_url = os.path.join(
                    img_path_dict['pred'],
                    'pred' + '_d' + str(5 - digit) + os.path.basename(img_digw_url)[4::]
                )
                if os.path.isfile(path=img_pdig_url):
                    data_err, img_pdig = data_ops.image_to_prediction_inception(
                        img_pdig_url=img_pdig_url,
                        tf_dict=tf_dict,
                        core_cfg=self.core_cfg
                    )
                    # print(img_pdig)
                    # print(img_pdig.shape)

                    if not data_err:
                        interpreter.set_tensor(input_details[0]['index'], img_pdig)
                        interpreter.invoke()
                        pred_outputs = interpreter.get_tensor(output_details[0]['index'])

                        for element in pred_outputs:
                            # print(index)
                            predictions[5 - digit][0] = element.argmax(axis=0)
                            predictions[5 - digit][1] = element[predictions[digit][0]]
                            print('Prediction is {0}, with probability {1}'.
                                  format(predictions[digit][0], predictions[digit][1]))

                        if predictions[5 - digit][0] >= 10:
                            pred_list[2] = 'I'
                        else:
                            pred_list[1] = str(predictions[5 - digit][0]) + pred_list[1]

                        if predictions[5 - digit][0] == 10:
                            pred_list[1] += 'A'

                            log = 'Prediction value for digit {0} is between 0 and 1.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 11:
                            pred_list[1] += 'B'

                            log = 'Prediction value for digit {0} is between 1 and 2.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 12:
                            pred_list[1] += 'C'

                            log = 'Prediction value for digit {0} is between 2 and 3.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 13:
                            pred_list[1] += 'D'

                            log = 'Prediction value for digit {0} is between 3 and 4.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 14:
                            pred_list[1] += 'E'

                            log = 'Prediction value for digit {0} is between 4 and 5.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 15:
                            pred_list[1] += 'F'

                            log = 'Prediction value for digit {0} is between 5 and 6.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 16:
                            pred_list[1] += 'G'

                            log = 'Prediction value for digit {0} is between 6 and 7.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 17:
                            pred_list[1] += 'H'

                            log = 'Prediction value for digit {0} is between 7 and 8.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 18:
                            pred_list[1] += 'I'

                            log = 'Prediction value for digit {0} is between 8 and 9.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 19:
                            pred_list[1] += 'J'

                            log = 'Prediction value for digit {0} is between 9 and 0.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 20:
                            pred_list[1] += 'K'

                            log = 'Prediction value for digit {0} is occluded over 0.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 21:
                            pred_list[1] += 'L'

                            log = 'Prediction value for digit {0} is occluded over 1.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 22:
                            pred_list[1] += 'M'

                            log = 'Prediction value for digit {0} is occluded over 2.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 23:
                            pred_list[1] += 'N'

                            log = 'Prediction value for digit {0} is occluded over 3.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 24:
                            pred_list[1] += 'O'

                            log = 'Prediction value for digit {0} is occluded over 4.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 25:
                            pred_list[1] += 'P'

                            log = 'Prediction value for digit {0} is occluded over 5.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 26:
                            pred_list[1] += 'Q'

                            log = 'Prediction value for digit {0} is occluded over 6.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 27:
                            pred_list[1] += 'R'

                            log = 'Prediction value for digit {0} is occluded over 7.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 28:
                            pred_list[1] += 'S'

                            log = 'Prediction value for digit {0} is occluded over 8.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        elif predictions[5 - digit][0] == 29:
                            pred_list[1] += 'T'

                            log = 'Prediction value for digit {0} is occluded over 9.'.format(digit)
                            logger.info(msg=log)
                            print(log)

                        pred_list.append(predictions[5 - digit][0])
                        pred_list.append(predictions[5 - digit][1])

                    else:
                        pred_err = True
                        log = 'Error loading data from {0}, could not make prediction.'.format(img_pdig_url)
                        logger.error(msg=log)
                        print(log)

                        pred_list[1] += 'Z'
                        pred_list[2] = 'I'

                else:
                    pred_err = True
                    log = 'Could not locate file {0} from which to make predictions.'.format(img_pdig_url)
                    logger.error(msg=log)
                    print(log)

                    pred_list[1] += 'Z'
                    pred_list[2] = 'I'

                print('Digit prediction time elapsed: {0} sec'.format(ttime.time() - timeb))

            if pred_list[2] == 'V':
                log = 'Prediction value has valid digits.'
                logger.info(msg=log)
                print(log)

            else:
                log = 'Prediction value has one or more invalid digits.'
                logger.info(msg=log)
                print(log)

            # Save prediction
            last_url_str = os.path.join(
                data_path_dict['last'],
                'last_' + img_orig_dtg + '_' + img_seq + '.txt'
            )
            copy_err = file_ops.copy_file(
                data_orig_url=cfg_url_dict['last'],
                data_dest_url=last_url_str
            )
            if not copy_err:
                file_ops.f_request(
                    file_cmd='file_csv_writelist',
                    file_name=last_url_str,
                    data_file_in=pred_list
                )

            hist_url_str = os.path.join(
                data_path_dict['hist'],
                'hist_' + img_orig_dtg.split('_')[0] + '.txt'
            )
            if not os.path.isfile(path=hist_url_str):
                copy_err = file_ops.copy_file(
                    data_orig_url=cfg_url_dict['hist'],
                    data_dest_url=hist_url_str
                )
                if not copy_err:
                    open(hist_url_str, 'w').close()

            # Dump every prediction into file for history
            file_ops.f_request(
                file_cmd='file_csv_appendlist',
                file_name=hist_url_str,
                data_file_in=pred_list
            )

            log = 'Successfully predicted digit values from numpy array.'
            logger.info(msg=log)

            # img_ops.remove_images(img_path=img_path_dict['pred'])
            # log = 'Deleted prediction images from disk.'
            # logger.info(msg=log)
            # print(log)

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
