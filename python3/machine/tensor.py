__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import colorsys
import cv2
import logging
import numpy as np
import random
import tensorflow as tf
import time as ttime
from config.tensor import TensorCfg
from machine import inception_v4, yolo_v3
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import concatenate


class Tensor:
    def __init__(
            self,
            core_cfg,
    ):
        """
        Initializes Tensor object

        :param core_cfg
        """
        logfile = 'januswm'
        self.logger = logging.getLogger(logfile)

        # Print version, clear session, and get GPUs
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

        # Set configuration variables
        self.core_cfg = core_cfg
        self.img_path_dict = core_cfg.get(attrib='img_path_dict')
        tensor_cfg = TensorCfg(core_cfg=core_cfg)
        self.yolo_dict = tensor_cfg.get(attrib='yolo_dict')
        self.incept_dict = tensor_cfg.get(attrib='incept_dict')

    def build_yolo_model(
        self
    ) -> (bool, any, dict):
        """
        Builds and returns YOLO v3 model with classes

        :return build_err: bool
        :return model
        :return classes: dict
        """
        build_err = False
        model = None

        # Get class names from file
        classes = {}
        with open(self.yolo_dict['classes'], 'r') as data:
            for ID, name in enumerate(data):
                classes[ID] = name.strip('\n')
        print(classes)

        # Get anchors from file
        with open(self.yolo_dict['anchors']) as f:
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
            # Get YOLO feature maps from input layer
            data_in = tf.keras.layers.Input(shape=[
                self.yolo_dict['input_size'],
                self.yolo_dict['input_size'],
                3
            ])
            feature_maps = yolo_v3.create_yolo_v3(
                data_in=data_in,
                classes=classes
            )

            # Get boundary box tensors from YOLO feature maps
            bbox_tensors = []
            for index, feature_map in enumerate(feature_maps):
                strides = np.array(self.yolo_dict['strides'])
                bbox_tensor = yolo_v3.decode(
                    data_in=feature_map,
                    classes=classes,
                    anchors=anchors[index],
                    strides=strides[index]
                )
                bbox_tensors.append(bbox_tensor)

            # Build YOLO model from input layer and empty boundary
            # box tensors, then load weights
            model = tf.keras.Model(data_in, bbox_tensors)
            model.load_weights(filepath=self.yolo_dict['wgts'])

        except Exception as exc:
            build_err = True
            log = 'Failed to build YOLO model.'
            self.logger.error(msg=log)
            self.logger.error(msg=exc)
            print(log)

        return build_err, model, classes

    def build_incept_model(
            self
    ) -> (bool, any):
        """
        Builds and returns Inception v4 model

        :return build_err: bool
        :return model
        """
        build_err = False
        model = None

        try:
            # Build Inception model then load weights
            model = inception_v4.create_inception_v4(
                incept_dict=self.incept_dict
            )
            model.load_weights(
                filepath=self.incept_dict['wgts'],
                by_name=False
            )
            # model.summary()

        except Exception as exc:
            build_err = True
            log = 'Failed to detect objects from image data.'
            self.logger.error(msg=log)
            self.logger.error(msg=exc)
            print(log)
            print(exc)

        return build_err, model

    def preprocess_yolo(
            self,
            img_orig
    ):
        """
        Resizes image in preparation for object detection

        :param img_orig

        :return preprocess_err: bool
        :return img_padded
        """
        preprocess_err = False
        img_padded = None

        # Get input sizes
        input_h = self.yolo_dict['input_size']
        input_w = self.yolo_dict['input_size']

        # Get image sizes
        img_h, img_w, _ = img_orig.shape

        # Determine scale of image vs input by taking
        # the minimum scale of either the width or height.
        # However all standard image sizes have shorter
        # heights vs widths.
        img_scale = min(input_w / img_w, input_h / img_h)
        # NOTE: square images of 1536x1536 and 4:3 images
        # of 3280x2464 work with YOLOv3, but not 16:9
        # images of 1920x1080. The 4:3 ratio is much closer
        # to square, may be the reason why 1920x1080 does
        # not work.

        # Calculate new sizes for processing image.
        new_w, new_h = int(img_scale * img_w), int(img_scale * img_h)
        diff_w, diff_h = (input_w - new_w) // 2, (input_h - new_h) // 2

        try:
            # Resize image
            img_resized = cv2.resize(
                src=img_orig,
                dsize=(
                    new_w,
                    new_h
                )
            )

            # Create image with padded values, same size as input sizes
            img_padded = np.full(
                shape=[
                    input_h,
                    input_w,
                    3
                ],
                fill_value=128.0
            )

            # Copy resized image into image with padded values
            img_padded[diff_h:new_h + diff_h, diff_w:new_w + diff_w, :] = img_resized

            # Normalize values with range 0 to 255.
            img_padded = img_padded / 255.

        except Exception as exc:
            preprocess_err = True
            log = 'Failed to preprocess image {0}.'.format(img_orig)
            self.logger.error(msg=log)
            self.logger.error(msg=exc)
            print(log)
            print(exc)

        return preprocess_err, img_padded

    def postprocess_boxes_yolo(
            self,
            pred_bbox,
            img_shape
    ):
        """
        Locates bounding boxes with best prediction confidence

        :param pred_bbox
        :param img_shape

        :return preprocess_err: bool
        :return img_padded
        """
        postprocess_err = False
        bboxes = None
        best_bboxes = []

        valid_scale = [0, np.inf]

        # Setup up prediction variables
        obj_bbox = np.array(pred_bbox)
        obj_xywh = obj_bbox[:, 0:4]
        obj_conf = obj_bbox[:, 4]
        obj_class = obj_bbox[:, 5:]

        try:
            # Get prediction coordinate center points
            obj_coords = np.concatenate(
                [
                    obj_xywh[:, :2] - obj_xywh[:, 2:] * 0.5,
                    obj_xywh[:, :2] + obj_xywh[:, 2:] * 0.5
                ],
                axis=-1
            )

            # Resize prediction box
            img_h, img_w = img_shape
            resize_ratio = min(
                self.yolo_dict['input_size'] / img_w,
                self.yolo_dict['input_size'] / img_h
            )

            diff_w = (self.yolo_dict['input_size'] - resize_ratio * img_w) / 2
            diff_h = (self.yolo_dict['input_size'] - resize_ratio * img_h) / 2

            obj_coords[:, 0::2] = 1.0 * (obj_coords[:, 0::2] - diff_w) / resize_ratio
            obj_coords[:, 1::2] = 1.0 * (obj_coords[:, 1::2] - diff_h) / resize_ratio

            # Clip prediction boxes that are out of range
            obj_coords = np.concatenate(
                [
                    np.maximum(
                        obj_coords[:, :2],
                        [0, 0]
                    ),
                    np.minimum(
                        obj_coords[:, 2:],
                        [img_w - 1, img_h - 1]
                    )
                ],
                axis=-1
            )
            invalid_mask = np.logical_or(
                (obj_coords[:, 0] > obj_coords[:, 2]),
                (obj_coords[:, 1] > obj_coords[:, 3])
            )
            obj_coords[invalid_mask] = 0

            # Discard invalid prediction boxes
            bboxes_scale = np.sqrt(np.multiply.reduce(
                obj_coords[:, 2:4] - obj_coords[:, 0:2],
                axis=-1
            ))
            scale_mask = np.logical_and(
                (valid_scale[0] < bboxes_scale),
                (bboxes_scale < valid_scale[1])
            )

            # Discard boxes with low scores
            classes = np.argmax(
                a=obj_class,
                axis=-1
            )
            scores = obj_conf * obj_class[
                np.arange(start=len(obj_coords)),
                classes
            ]
            score_mask = scores > self.yolo_dict['score_thresh']
            mask = np.logical_and(
                scale_mask,
                score_mask
            )
            coords = obj_coords[mask]
            scores = scores[mask]
            classes = classes[mask]
            bboxes = np.concatenate(
                [
                    coords,
                    scores[:, np.newaxis],
                    classes[:, np.newaxis]
                ],
                axis=-1
            )

            classes_in_img = list(set(bboxes[:, 5]))

            # Iterate through boundary boxes and find boxes
            # with highest confidence in each class
            for cls in classes_in_img:
                cls_mask = (bboxes[:, 5] == cls)
                cls_bboxes = bboxes[cls_mask]

                if len(cls_bboxes) > 0:
                    max_ind = np.argmax(a=cls_bboxes[:, 4])
                    best_bbox = cls_bboxes[max_ind]
                    best_bboxes.append(best_bbox)

        except Exception as exc:
            postprocess_err = True
            log = 'Failed to postprocess boxes for image.'
            self.logger.error(msg=log)
            self.logger.error(msg=exc)
            print(log)
            print(exc)

        return postprocess_err, bboxes, best_bboxes

    def draw_bbox_yolo(
            self,
            img_orig,
            img_orig_shape,
            bboxes,
            classes=None,
    ) -> (bool, any):
        """
        Draws boundary boxes for detected objects

        :param img_orig
        :param img_orig_shape
        :param bboxes
        :param classes

        :return draw_bbox_err: bool
        :return img_orig
        """
        draw_bbox_err = False
        num_classes = len(classes)

        try:

            # Setup colors, one for each class
            hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(map(
                lambda x: (
                    int(x[0] * 255),
                    int(x[1] * 255),
                    int(x[2] * 255)
                ),
                colors
            ))

            random.seed(0)
            random.shuffle(colors)
            random.seed(None)

            # Draw boundary boxes and add label to original image
            for i, bbox in enumerate(bboxes):
                coords = np.array(
                    bbox[:4],
                    dtype=np.int32
                )
                font_scale = 0.5
                score = bbox[4]
                class_ind = int(bbox[5])
                bbox_color = colors[class_ind]
                bbox_thick = int(0.6 * (img_orig_shape[0] + img_orig_shape[1]) / 600)
                c1 = (coords[0], coords[1])
                c2 = (coords[2], coords[3])
                cv2.rectangle(
                    img=img_orig,
                    pt1=c1,
                    pt2=c2,
                    color=bbox_color,
                    thickness=bbox_thick
                )

                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
                t_size = cv2.getTextSize(
                    text=bbox_mess,
                    fontFace=0,
                    fontScale=font_scale,
                    thickness=bbox_thick // 2
                )[0]
                cv2.rectangle(
                    img=img_orig,
                    pt1=c1,
                    pt2=(c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                    color=bbox_color,
                    thickness=-1
                )  # filled

                cv2.putText(
                    img=img_orig,
                    text=bbox_mess,
                    org=(c1[0], c1[1] - 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=(0, 0, 0),
                    thickness=bbox_thick // 2,
                    lineType=cv2.LINE_AA
                )

        except Exception as exc:
            draw_bbox_err = True
            log = 'Failed to draw boundary boxes for image {0}.'.format(img_orig)
            self.logger.error(msg=log)
            self.logger.error(msg=exc)
            print(log)
            print(exc)

        return draw_bbox_err, img_orig

    def detect_yolo(
            self,
            model,
            img_orig,
            img_orig_shape,
            img_save: bool,
            img_bbox_url: str,
            classes
    ) -> (bool, dict):
        """
        Execute image object detection

        :param model
        :param img_orig
        :param img_orig_shape
        :param img_save
        :param img_bbox_url
        :param classes

        :return detect_err: bool
        :return sorted_bbox: dict
        """
        detect_err = False
        sorted_bbox = {}

        try:
            # Convert image from BGR to RGB
            img_inverse = cv2.cvtColor(
                src=img_orig,
                code=cv2.COLOR_BGR2RGB
            )

            # Resize and pad images as necessary to make them usable for model
            preprocess_err, img_data = self.preprocess_yolo(
                img_orig=np.copy(img_inverse)
            )
            img_data = img_data[np.newaxis, ...].astype(dtype=np.float32)

            # Run image data through prediction model and reshape
            pred_bbox = model.predict(x=img_data)
            pred_bbox = [
                tf.reshape(
                    tensor=x,
                    shape=(
                        -1,
                        tf.shape(input=x)[-1]
                    )
                ) for x in pred_bbox
            ]
            pred_bbox = concatenate(
                inputs=pred_bbox,
                axis=0
            )

            # Get boxes for each object that have highest confidence
            postprocess_err, bboxes, best_bboxes = self.postprocess_boxes_yolo(
                pred_bbox=pred_bbox,
                img_shape=img_orig_shape[0:2]
            )

            # Draw and save YOLO boundary box image
            if self.yolo_dict['rslts'] is not None:
                draw_bbox_err, img_bbox = self.draw_bbox_yolo(
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

            # Sort the boundary boxes
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

    def prepare_data_inception(
            self,
            img_dig
    ) -> (any, bool):
        """
        Prepares digit data for Inception v4 model

        :param img_dig

        :return data_err: bool
        :return img_data
        """
        data_err = False
        img_data = None

        try:
            # Convert digit image data to RGB format, load into
            # TensorFlow tensor, convert to gray-scale, resize to
            # 299x299, then expand dimensions from (299, 299, 1)
            # to (x, 299, 299, 1)
            img_rgb = cv2.cvtColor(
                img_dig,
                cv2.COLOR_BGR2RGB
            )
            img_tensor = tf.convert_to_tensor(
                img_rgb,
                dtype=tf.float32
            )
            img_gray = tf.image.rgb_to_grayscale(img_tensor)
            img_resized = tf.image.resize_with_pad(
                img_gray,
                self.incept_dict['img_tgt_width'],
                self.incept_dict['img_tgt_height']
            )
            img_data = tf.expand_dims(
                img_resized,
                0
            )

        except Exception as exc:
            data_err = True
            log = 'Failed to convert digit data to tensor.'
            self.logger.error(msg=log)
            self.logger.error(msg=exc)
            print(log)
            print(exc)

        return data_err, img_data

    def predict_inception(
            self,
            model,
            img_digs,
            img_orig_dtg: str
    ) -> [bool, list]:
        """
        Generates and returns predictions for each digit using TensorFlow

        :param model
        :param img_digs
        :param img_orig_dtg: str

        :return pred_err: bool
        :return pred_list: list
        """
        timea = ttime.time()

        pred_err = False
        pred_list = [
            img_orig_dtg,
            '',
            'I'
        ]

        try:

            # Iterate through all six digits
            for digit in range(0, 6):

                # Prepare digit data for Inception v4 model
                data_err, img_data = self.prepare_data_inception(
                        img_dig=img_digs[digit]
                )

                if not data_err:

                    # Execute prediction algorithm on digit
                    predictions = model.predict(
                        x=img_data,
                        batch_size=self.incept_dict['batch_size']
                    )

                    # Iterate through each prediction, build prediction
                    # list after processing each digit's prediction
                    for element in predictions:
                        prediction = element.argmax(axis=0)
                        confidence = element[prediction]
                        log = 'Digit {0} prediction is {1}, with confidence {2}'.\
                            format(digit, prediction, confidence)
                        self.logger.info(msg=log)
                        print(log)

                        # Test prediction against confidence threshold
                        if confidence < self.incept_dict['confidence']:
                            pred_list[1] += 'R'
                            pred_list[2] = 'I'

                            log = 'Digit {0} prediction rejected due to low confidence.'. \
                                format(digit)
                            self.logger.info(msg=log)
                            print(log)

                        else:
                            # Prediction falls into range of normal
                            # values, 0 through 9
                            if prediction < 10:
                                pred_list[1] += str(prediction)
                                pred_list[2] = 'V'

                            # Prediction falls into transition zones
                            # between values
                            elif (prediction >= 10) and (prediction < 20):
                                pred_list[1] += 'T'
                                pred_list[2] = 'I'

                                log = 'Digit {0} falls on transition boundary.'.\
                                    format(digit)
                                self.logger.info(msg=log)
                                print(log)

                            # Prediction shows occluded by meter needle
                            elif (prediction >= 20) and (prediction < 30):
                                pred_list[1] += 'N'
                                pred_list[2] = 'I'

                                log = 'Digit {0} is occluded by the needle.'.\
                                    format(digit)
                                self.logger.info(msg=log)
                                print(log)

                        pred_list.append(prediction)
                        pred_list.append(confidence)

                    if pred_list[2] == 'V':
                        log = 'Prediction value has valid digits.'
                        self.logger.info(msg=log)
                        print(log)

                    else:
                        log = 'Prediction value has one or more invalid digits.'
                        self.logger.info(msg=log)
                        print(log)

                    log = 'Successfully predicted digit values from image data.'
                    self.logger.info(msg=log)
                    print(log)

            print(pred_list)

        except Exception as exc:
            pred_err = True
            log = 'Failed to predict digit values from image data.'
            self.logger.error(msg=log)
            self.logger.error(msg=exc)
            print(log)
            print(exc)

        # Build text with which to overlay image
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
