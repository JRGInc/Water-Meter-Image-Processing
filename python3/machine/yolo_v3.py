#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolo_v3.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:47:10
#   Description :
#
# ================================================================

import tensorflow as tf

# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import BatchNormalization
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import concatenate
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Conv2D
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import ZeroPadding2D


class YoloBN(BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """

    def call(
            self,
            inputs,
            training=False
    ):
        if not training:
            training = tf.constant(value=False)
        training = tf.math.logical_and(
            x=training,
            y=self.trainable
        )
        return super().call(
            inputs=inputs,
            training=training
        )


def convolutional(
        input_layer,
        filters_shape,
        downsample=False,
        activate=True,
        bn=True
):
    if downsample:
        input_layer = ZeroPadding2D(padding=(
            (1, 0),
            (1, 0)
        ))(inputs=input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(
        filters=filters_shape[-1],
        kernel_size=filters_shape[0],
        strides=strides,
        padding=padding,
        use_bias=not bn,
        kernel_regularizer=tf.keras.regularizers.l2(l=0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(value=0.)
    )(inputs=input_layer)

    if bn:
        conv = YoloBN()(inputs=conv)
    if activate:
        conv = tf.nn.leaky_relu(
            features=conv,
            alpha=0.1
        )

    return conv


def residual_block(
        input_layer,
        input_channel,
        filter_num1,
        filter_num2
):
    short_cut = input_layer
    conv = convolutional(
        input_layer=input_layer,
        filters_shape=(1, 1, input_channel, filter_num1)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(3, 3, filter_num1, filter_num2)
    )

    residual_output = short_cut + conv
    return residual_output


def upsample(input_layer):
    return tf.image.resize(
        images=input_layer,
        size=(
            input_layer.shape[1] * 2,
            input_layer.shape[2] * 2
        ),
        method='nearest'
    )


def darknet53(input_data):
    input_data = convolutional(
        input_layer=input_data,
        filters_shape=(3, 3, 3, 32)
    )
    input_data = convolutional(
        input_layer=input_data,
        filters_shape=(3, 3, 32, 64),
        downsample=True
    )

    for i in range(1):
        input_data = residual_block(
            input_layer=input_data,
            input_channel=64,
            filter_num1=32,
            filter_num2=64
        )

    input_data = convolutional(
        input_layer=input_data,
        filters_shape=(3, 3, 64, 128),
        downsample=True
    )

    for i in range(2):
        input_data = residual_block(
            input_layer=input_data,
            input_channel=128,
            filter_num1=64,
            filter_num2=128
        )

    input_data = convolutional(
        input_layer=input_data,
        filters_shape=(3, 3, 128, 256),
        downsample=True
    )

    for i in range(8):
        input_data = residual_block(
            input_layer=input_data,
            input_channel=256,
            filter_num1=128,
            filter_num2=256
        )

    route_1 = input_data
    input_data = convolutional(
        input_layer=input_data,
        filters_shape=(3, 3, 256, 512),
        downsample=True
    )

    for i in range(8):
        input_data = residual_block(
            input_layer=input_data,
            input_channel=512,
            filter_num1=256,
            filter_num2=512
        )

    route_2 = input_data
    input_data = convolutional(
        input_layer=input_data,
        filters_shape=(3, 3, 512, 1024),
        downsample=True
    )

    for i in range(4):
        input_data = residual_block(
            input_layer=input_data,
            input_channel=1024,
            filter_num1=512,
            filter_num2=1024
        )

    return route_1, route_2, input_data


def create_yolo_v3(
        input_layer,
        classes
):
    nbr_classes = len(classes)
    route_1, route_2, conv = darknet53(input_data=input_layer)

    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 1024, 512)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(3, 3, 512, 1024)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 1024, 512)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(3, 3, 512, 1024)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 1024, 512)
    )

    conv_lobj_branch = convolutional(
        input_layer=conv,
        filters_shape=(3, 3, 512, 1024)
    )
    conv_lbbox = convolutional(
        input_layer=conv_lobj_branch,
        filters_shape=(1, 1, 1024, 3 * (nbr_classes + 5)),
        activate=False,
        bn=False
    )

    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 512, 256)
    )
    conv = upsample(conv)

    conv = concatenate(
        inputs=[
            conv,
            route_2
        ],
        axis=-1
    )

    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 768, 256)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(3, 3, 256, 512)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 512, 256)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(3, 3, 256, 512)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 512, 256)
    )

    conv_mobj_branch = convolutional(
        input_layer=conv,
        filters_shape=(3, 3, 256, 512)
    )
    conv_mbbox = convolutional(
        input_layer=conv_mobj_branch,
        filters_shape=(1, 1, 512, 3 * (nbr_classes + 5)),
        activate=False,
        bn=False
    )

    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 256, 128)
    )
    conv = upsample(conv)

    conv = concatenate(
        inputs=[
            conv,
            route_1
        ],
        axis=-1
    )

    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 384, 128)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(3, 3, 128, 256)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 256, 128)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(3, 3, 128, 256)
    )
    conv = convolutional(
        input_layer=conv,
        filters_shape=(1, 1, 256, 128)
    )

    conv_sobj_branch = convolutional(
        input_layer=conv,
        filters_shape=(3, 3, 128, 256)
    )
    conv_sbbox = convolutional(
        input_layer=conv_sobj_branch,
        filters_shape=(1, 1, 256, 3 * (nbr_classes + 5)),
        activate=False,
        bn=False
    )

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(
        conv_output,
        classes,
        anchors,
        strides,
        i=0
):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """
    nbr_classes = len(classes)
    conv_shape = tf.shape(input=conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(
        tensor=conv_output,
        shape=(
            batch_size,
            output_size,
            output_size,
            3,
            5 + nbr_classes
        )
    )

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    y = tf.tile(
        input=tf.range(
            start=0,
            limit=output_size,
            dtype=tf.int32
        )[:, tf.newaxis],
        multiples=[1, output_size]
    )
    x = tf.tile(
        input=tf.range(
            start=0,
            limit=output_size,
            dtype=tf.int32
        )[tf.newaxis, :],
        multiples=[output_size, 1]
    )

    xy_grid = concatenate(
        inputs=[
            x[:, :, tf.newaxis],
            y[:, :, tf.newaxis]
        ],
        axis=-1
    )
    xy_grid = tf.tile(
        input=xy_grid[tf.newaxis, :, :, tf.newaxis, :],
        multiples=[batch_size, 1, 1, 3, 1]
    )
    xy_grid = tf.cast(
        x=xy_grid,
        dtype=tf.float32
    )

    pred_xy = (tf.math.sigmoid(x=conv_raw_dxdy) + xy_grid) * strides[i]
    pred_wh = (tf.math.exp(x=conv_raw_dwdh) * anchors[i]) * strides[i]
    pred_xywh = concatenate(
        inputs=[
            pred_xy,
            pred_wh
        ],
        axis=-1
    )

    pred_conf = tf.math.sigmoid(x=conv_raw_conf)
    pred_prob = tf.math.sigmoid(x=conv_raw_prob)

    return concatenate(
        inputs=[
            pred_xywh,
            pred_conf,
            pred_prob
        ],
        axis=-1
    )
