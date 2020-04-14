__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import tensorflow as tf

# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import BatchNormalization
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import concatenate
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Conv2D
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import ZeroPadding2D


class YoloBatchNorm(BatchNormalization):
    """
    Subclass TensorFlow BatchNormalization for some simple modifications
    """
    def call(
            self,
            inputs,
            training: bool = False
    ) -> any:
        """
        Call for customized BatchNormalization

        :param inputs
        :param training: bool

        :return x
        """
        # Convert training parameter to TensorFlow variable
        if training:
            training = tf.constant(value=True)
        else:
            training = tf.constant(value=False)

        # Set training parameter in TensorFlow BatchNormalization
        # only if the user requires training, and the layer.trainable
        # flag is set.
        """
        "Frozen state" and "inference mode" are two separate concepts.
        `layer.trainable = False` is to freeze the layer, so the layer will use
        stored moving `var` and `mean` in the "inference mode", and both `gama`
        and `beta` will not be updated !
        """
        training = tf.math.logical_and(
            x=training,
            y=self.trainable
        )

        # Calls and returns TensorFlow BatchNormalized data
        data_out = super().call(
            inputs=inputs,
            training=training
        )
        return data_out


def conv_block(
        data_in,
        nb_filter: int,
        size: tuple,
        downsample: bool = False,
        activate: bool = True,
        batch_norm: bool = True
) -> any:
    """
    Builds common convolution block and returns convoluted data

    :param data_in
    :param nb_filter: int
    :param size: tuple
    :param downsample: bool
    :param activate: bool
    :param batch_norm: bool

    :return conv_data
    """
    if downsample:
        data_in = ZeroPadding2D(padding=(
            (1, 0),
            (1, 0)
        ))(inputs=data_in)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv_data = Conv2D(
        filters=nb_filter,
        kernel_size=size,
        strides=strides,
        padding=padding,
        use_bias=not batch_norm,
        kernel_regularizer=tf.keras.regularizers.l2(l=0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(value=0.)
    )(inputs=data_in)

    if batch_norm:
        conv_data = YoloBatchNorm()(inputs=conv_data)
    if activate:
        conv_data = tf.nn.leaky_relu(
            features=conv_data,
            alpha=0.1
        )

    return conv_data


def res_block(
        data_in,
        filter1: int,
        filter2: int
) -> any:
    """
    Builds common residual block and returns convoluted data

    :param data_in
    :param filter1: int
    :param filter2: int

    :return data_out
    """
    conv_data = conv_block(
        data_in=data_in,
        nb_filter=filter1,
        size=(1, 1)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=filter2,
        size=(3, 3)
    )

    data_out = data_in + conv_data
    return data_out


def darknet53(
        data_in
) -> list:
    """
    Builds Darknet53 block and returns a list of convoluted data

    :param data_in

    :return [conv_data, conv_data_r1, conv_data_r2]
    """
    # Transit route 0 through neural network
    conv_data_r0 = conv_block(
        data_in=data_in,
        nb_filter=32,
        size=(3, 3)
    )
    conv_data_r0 = conv_block(
        data_in=conv_data_r0,
        nb_filter=64,
        size=(3, 3),
        downsample=True
    )

    for i in range(1):
        conv_data_r0 = res_block(
            data_in=conv_data_r0,
            filter1=32,
            filter2=64
        )

    conv_data_r0 = conv_block(
        data_in=conv_data_r0,
        nb_filter=128,
        size=(3, 3),
        downsample=True
    )

    for i in range(2):
        conv_data_r0 = res_block(
            data_in=conv_data_r0,
            filter1=64,
            filter2=128
        )

    conv_data_r0 = conv_block(
        data_in=conv_data_r0,
        nb_filter=256,
        size=(3, 3),
        downsample=True
    )

    for i in range(8):
        conv_data_r0 = res_block(
            data_in=conv_data_r0,
            filter1=128,
            filter2=256
        )

    # Transit route 1 through neural network
    conv_data_r1 = conv_data_r0
    conv_data_r0 = conv_block(
        data_in=conv_data_r0,
        nb_filter=512,
        size=(3, 3),
        downsample=True
    )

    for i in range(8):
        conv_data_r0 = res_block(
            data_in=conv_data_r0,
            filter1=256,
            filter2=512
        )

    # Transit route 2 through neural network
    conv_data_r2 = conv_data_r0
    conv_data_r0 = conv_block(
        data_in=conv_data_r0,
        nb_filter=1024,
        size=(3, 3),
        downsample=True
    )

    for i in range(4):
        conv_data_r0 = res_block(
            data_in=conv_data_r0,
            filter1=512,
            filter2=1024
        )

    return [conv_data_r0, conv_data_r1, conv_data_r2]


def create_yolo_v3(
        data_in,
        classes: dict
) -> list:
    """
    Builds empty Yolo v3 model and returns convolutions of small,
    medium, and large boundary boxes

    :param data_in
    :param classes: dict

    :return [conv_sbbox, conv_mbbox, conv_lbbox]
    """
    yolo_filter = 3 * (len(classes) + 5)
    darknet_data = darknet53(data_in=data_in)

    conv_data = conv_block(
        data_in=darknet_data[0],
        nb_filter=512,
        size=(1, 1)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=1024,
        size=(3, 3)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=512,
        size=(1, 1)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=1024,
        size=(3, 3)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=512,
        size=(1, 1)
    )

    conv_lobj_branch = conv_block(
        data_in=conv_data,
        nb_filter=1024,
        size=(3, 3)
    )
    conv_lbbox = conv_block(
        data_in=conv_lobj_branch,
        nb_filter=yolo_filter,
        size=(1, 1),
        activate=False,
        batch_norm=False
    )

    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=256,
        size=(1, 1)
    )
    conv_data = tf.image.resize(
        images=conv_data,
        size=(
            conv_data.shape[1] * 2,
            conv_data.shape[2] * 2
        ),
        method='nearest'
    )

    conv_data = concatenate(
        inputs=[
            conv_data,
            darknet_data[2]
        ],
        axis=-1
    )

    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=256,
        size=(1, 1)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=512,
        size=(3, 3)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=256,
        size=(1, 1)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=512,
        size=(3, 3)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=256,
        size=(1, 1)
    )

    conv_mobj_branch = conv_block(
        data_in=conv_data,
        nb_filter=512,
        size=(3, 3)
    )
    conv_mbbox = conv_block(
        data_in=conv_mobj_branch,
        nb_filter=yolo_filter,
        size=(1, 1),
        activate=False,
        batch_norm=False
    )

    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=128,
        size=(1, 1)
    )
    conv_data = tf.image.resize(
        images=conv_data,
        size=(
            conv_data.shape[1] * 2,
            conv_data.shape[2] * 2
        ),
        method='nearest'
    )

    conv_data = concatenate(
        inputs=[
            conv_data,
            darknet_data[1]
        ],
        axis=-1
    )

    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=128,
        size=(1, 1)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=256,
        size=(3, 3)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=128,
        size=(1, 1)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=256,
        size=(3, 3)
    )
    conv_data = conv_block(
        data_in=conv_data,
        nb_filter=128,
        size=(1, 1)
    )

    conv_sobj_branch = conv_block(
        data_in=conv_data,
        nb_filter=256,
        size=(3, 3)
    )
    conv_sbbox = conv_block(
        data_in=conv_sobj_branch,
        nb_filter=yolo_filter,
        size=(1, 1),
        activate=False,
        batch_norm=False
    )

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(
        data_in,
        classes: dict,
        anchors,
        strides
) -> any:
    """
    Decodes empty YOLO convoluted boundary boxes to produce empty
    human-readable data to feed into YOLO v3 model

    :param data_in
    :param classes: dict
    :param anchors
    :param strides

    return tensor with (x, y, w, h), confidence, class)
    """
    conv_shape = tf.shape(input=data_in)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_data = tf.reshape(
        tensor=data_in,
        shape=(
            batch_size,
            output_size,
            output_size,
            3,
            5 + len(classes)
        )
    )

    conv_raw_dxdy = conv_data[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_data[:, :, :, :, 2:4]
    conv_raw_conf = conv_data[:, :, :, :, 4:5]
    conv_raw_class = conv_data[:, :, :, :, 5:]

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

    pred_xy = (tf.math.sigmoid(x=conv_raw_dxdy) + xy_grid) * strides
    pred_wh = (tf.math.exp(x=conv_raw_dwdh) * anchors) * strides
    pred_xywh = concatenate(
        inputs=[
            pred_xy,
            pred_wh
        ],
        axis=-1
    )

    pred_conf = tf.math.sigmoid(x=conv_raw_conf)
    pred_class = tf.math.sigmoid(x=conv_raw_class)

    data_out = concatenate(
        inputs=[
            pred_xywh,
            pred_conf,
            pred_class
        ],
        axis=-1
    )
    return data_out
