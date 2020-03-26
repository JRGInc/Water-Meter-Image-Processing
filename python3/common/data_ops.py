__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import logging
import numpy as np
import os
import pathlib
import random
import shutil
import tensorflow as tf
from common.file_ops import copy_file


logfile = 'januswm'
logger = logging.getLogger(logfile)


def sift_digits(
    orig_path: str,
    sift_path: str
) -> bool:
    """
    Sifts digit images and separates them into folders according to digit position

    :param orig_path: str
    :param sift_path: str

    :return sift_err: bool
    """
    sift_err = False

    try:
        for img_orig_name in os.listdir(orig_path):
            img_orig_url = os.path.join(
                orig_path,
                img_orig_name
            )
            img_orig_core_name = str(img_orig_name.split(sep='.')[0])

            digit = img_orig_core_name.split(sep='_')[1]
            sift_dest_path = os.path.join(
                sift_path,
                digit
            )
            img_orig_name = img_orig_core_name[5:] + '.jpg'
            img_dest_url = os.path.join(
                sift_dest_path,
                img_orig_name
            )

            copy_err = copy_file(
                data_orig_url=img_orig_url,
                data_dest_url=img_dest_url
            )

            if copy_err:
                sift_err = True
                log = 'Failed to sift digits in path {0}.'.format(orig_path)
                logger.error(msg=log)
                print(log)
                break

    except Exception as exc:
        sift_err = True
        log = 'Failed to sift digits in path {0}.'.format(orig_path)
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return sift_err


class SplitDatasetInception:
    def __init__(
        self,
        train_imgs_dir,
        valid_imgs_dir,
        valid_ratio,
        show_progress=False
    ):
        self.train_imgs_dir = train_imgs_dir
        self.valid_imgs_dir = valid_imgs_dir

        self.valid_ratio = valid_ratio

        self.train_file_path = []
        self.valid_file_path = []

        self.index_label_dict = {}

        self.show_progress = show_progress

        if not os.path.exists(self.valid_imgs_dir):
            os.mkdir(self.valid_imgs_dir)

    def __get_label_names(self):
        label_names = []
        for img in os.listdir(self.train_imgs_dir):
            img_path = os.path.join(self.train_imgs_dir, img)
            if os.path.isdir(img_path):
                label_names.append(img)
        return label_names

    def __get_all_file_path(self):
        all_file_path = []
        index = 0
        for file_type in self.__get_label_names():
            self.index_label_dict[index] = file_type
            index += 1
            type_file_path = os.path.join(
                self.train_imgs_dir,
                file_type
            )
            file_path = []
            for file in os.listdir(type_file_path):
                single_file_path = os.path.join(
                    type_file_path,
                    file
                )
                file_path.append(single_file_path)
            all_file_path.append(file_path)
        return all_file_path

    @staticmethod
    def __copy_files(
        type_path,
        type_saved_dir
    ):
        for item in type_path:
            src_path_list = item[1]
            dst_path = type_saved_dir + "%s/" % (item[0])
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            for src_path in src_path_list:
                shutil.copy(src_path, dst_path)

    def __split_dataset(self):
        all_file_paths = self.__get_all_file_path()
        for index in range(len(all_file_paths)):
            file_path_list = all_file_paths[index]
            train_num = len(file_path_list)
            random.shuffle(file_path_list)

            valid_num = int(train_num * self.valid_ratio)

            self.valid_file_path.append([
                self.index_label_dict[index],
                file_path_list[(train_num - valid_num):]
            ])

    def start_splitting(self):
        self.__split_dataset()
        self.__copy_files(
            type_path=self.valid_file_path,
            type_saved_dir=self.valid_imgs_dir
        )


def prepare_data_inception(
    mode: str,
    incept_dict: dict,
    img_root_dir: str = '',
    img_url: str = '',
    img_cache_dir: str = ''
):

    img_count = 0
    img_list_ds = None
    img_ds = None

    if mode == 'train':
        img_dir = pathlib.Path(img_root_dir)
        img_count = len(list(img_dir.glob('*/*.jpg')))
        img_list_ds = tf.data.Dataset.list_files(str(img_dir / '*/*'))

    elif (mode == 'test') or (mode == 'pred'):
        img_count = 1
        img_list_ds = tf.data.Dataset.list_files(img_url)

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return tf.strings.to_number(
            parts[-2],
            tf.dtypes.int32
        )

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(
            img,
            channels=incept_dict['nbr_channels']
        )
        # resize the image to the desired size.
        img = tf.image.resize_with_pad(
            img,
            incept_dict['img_tgt_width'],
            incept_dict['img_tgt_height']
        )
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        return tf.image.convert_image_dtype(
            img,
            tf.float32
        )

    def process_path(file_path):
        if (mode == 'train') or (mode == 'test'):
            label = get_label(file_path)
        else:
            label = tf.strings.to_number(
                '0',
                tf.dtypes.int32
            )

        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    img_labeled_ds = img_list_ds.map(
        process_path,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    def prepare(
        ds,
        batch_size: int,
        cache=None,
        shuffle_buffer_size=200
    ):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(
            batch_size=batch_size,
            drop_remainder=True
        )

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    if mode == 'train':
        img_ds = prepare(
            ds=img_labeled_ds,
            cache=img_cache_dir,
            batch_size=incept_dict['batch_size']
        )
    elif (mode == 'test') or (mode == 'pred'):
        img_ds = img_labeled_ds.batch(batch_size=incept_dict['batch_size'])

    return img_ds, img_count


def image_to_prediction_inception(
    img_pdig_url: str,
    tf_dict: dict
) -> [bool, any]:
    """
    Loads cropped digits into numpy array for Tensor Flow prediction

    :param img_pdig_url: str
    :param tf_dict: dict

    :return data_err: bool
    :return img_pdig: tensor
    """
    data_err = False
    img_pdig = None

    try:
        img_pdig = tf.io.read_file(img_pdig_url)
        img_pdig = tf.image.decode_jpeg(
            img_pdig,
            channels=tf_dict['nbr_channels']
        )
        img_pdig = tf.image.resize_with_pad(
            img_pdig,
            tf_dict['img_tgt_width'],
            tf_dict['img_tgt_height']
        )
        img_pdig = tf.expand_dims(img_pdig, 0)

        log = 'Successfully loaded digit from {0} into tensor.'. \
            format(img_pdig_url)
        logger.info(msg=log)
        print(log)

    except Exception as exc:
        data_err = True
        log = 'Failed to load digits into numpy array.'
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

    return data_err, img_pdig


def load_weights_yolo(model, weights_file):
    """
    I agree that this code is very ugly, but I don’t know any better way of doing it.
    """
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    bn_weights = None
    bn_layer = None
    conv_bias = None

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()
