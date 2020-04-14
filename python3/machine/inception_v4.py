__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import tensorflow as tf

# noinspection PyUnresolvedReferences
from tensorflow.keras import Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Activation
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import AveragePooling2D
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import BatchNormalization
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import concatenate
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Conv2D
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Dense
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Dropout
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Flatten
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Input
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import MaxPooling2D


def conv_block(
    data_in,
    nb_filter: int,
    size: tuple,
    padding: str = 'same',
    strides: int = 1,
    channel_axis: int = -1
) -> any:
    """
    Builds common convolution block and returns convoluted data

    :param data_in
    :param nb_filter: int
    :param size: tuple
    :param padding: str
    :param strides: int
    :param channel_axis: int

    :return x
    """
    conv_data = Conv2D(
        filters=nb_filter,
        kernel_size=size,
        strides=strides,
        padding=padding
    )(inputs=data_in)
    conv_data = BatchNormalization(
        axis=channel_axis
    )(inputs=conv_data)
    conv_data = Activation(
        activation=tf.nn.relu
    )(inputs=conv_data)

    return conv_data


def inception_stem(
    data_in,
    channel_axis: int = -1
) -> any:
    """
    Build Inception stem

    :param data_in
    :param channel_axis: int

    :return conv_data
    """
    # Use original data to transit route 0 through neural network
    conv_data_r0 = conv_block(
        data_in=data_in,
        nb_filter=32,
        size=(3, 3),
        strides=2,
        padding='valid'
    )
    conv_data_r0 = conv_block(
        data_in=conv_data_r0,
        nb_filter=32,
        size=(3, 3),
        padding='valid'
    )
    conv_data_r0 = conv_block(
        data_in=conv_data_r0,
        nb_filter=64,
        size=(3, 3)
    )

    # Use route 0 data to transit route 1 through neural network
    conv_data_r1 = MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='valid'
    )(inputs=conv_data_r0)

    # Use route 0 data to transit route 2 through neural network
    conv_data_r2 = conv_block(
        data_in=conv_data_r0,
        nb_filter=96,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    # Merge route 1 and route 2 back into route 0
    conv_data_r0 = concatenate(
        inputs=[
            conv_data_r1,
            conv_data_r2
        ],
        axis=channel_axis
    )

    # Use route 0 data to transit route 1 through neural network
    conv_data_r1 = conv_block(
        data_in=conv_data_r0,
        nb_filter=64,
        size=(1, 1)
    )
    conv_data_r1 = conv_block(
        data_in=conv_data_r1,
        nb_filter=96,
        size=(3, 3),
        padding='valid'
    )

    # Use route 0 data to transit route 2 through neural network
    conv_data_r2 = conv_block(
        data_in=conv_data_r0,
        nb_filter=64,
        size=(1, 1)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=64,
        size=(1, 7)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=64,
        size=(7, 1)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=96,
        size=(3, 3),
        padding='valid'
    )

    # Merge route 1 and 2 back into route 0
    conv_data_r0 = concatenate(
        inputs=[
            conv_data_r1,
            conv_data_r2
        ],
        axis=channel_axis
    )

    # Use route 0 data to transit route 1 through neural network
    conv_data_r1 = conv_block(
        data_in=conv_data_r0,
        nb_filter=192,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    # Use route 0 data to transit route 2 through neural network
    conv_data_r2 = MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='valid'
    )(inputs=conv_data_r0)

    # Merge route 1 and route 2 back into route 0
    conv_data_r0 = concatenate(
        inputs=[
            conv_data_r1,
            conv_data_r2
        ],
        axis=channel_axis
    )

    return conv_data_r0


def inception_a(
    data_in,
    channel_axis: int = -1
) -> any:
    """
    Build Inception A block

    :param data_in
    :param channel_axis: int

    :return conv_data
    """
    # Use original data to transit route 1 through neural network
    conv_data_r1 = conv_block(
        data_in=data_in,
        nb_filter=96,
        size=(1, 1)
    )

    # Use original data to transit route 2 through neural network
    conv_data_r2 = conv_block(
        data_in=data_in,
        nb_filter=64,
        size=(1, 1)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=96,
        size=(3, 3)
    )

    # Use original data to transit route 3 through neural network
    conv_data_r3 = conv_block(
        data_in=data_in,
        nb_filter=64,
        size=(1, 1)
    )
    conv_data_r3 = conv_block(
        data_in=conv_data_r3,
        nb_filter=96,
        size=(3, 3)
    )
    conv_data_r3 = conv_block(
        data_in=conv_data_r3,
        nb_filter=96,
        size=(3, 3)
    )

    # Use original data to transit route 4 through neural network
    conv_data_r4 = AveragePooling2D(
        pool_size=3,
        strides=1,
        padding='same'
    )(inputs=data_in)
    conv_data_r4 = conv_block(
        data_in=conv_data_r4,
        nb_filter=96,
        size=(1, 1)
    )

    # Merge routes 1, 2, 3, and 4 into route 0
    conv_data_r0 = concatenate(
        inputs=[
            conv_data_r1,
            conv_data_r2,
            conv_data_r3,
            conv_data_r4
        ],
        axis=channel_axis
    )

    return conv_data_r0


def inception_b(
    data_in,
    channel_axis: int = -1
) -> any:
    """
    Build Inception B block

    :param data_in
    :param channel_axis: int

    :return x
    """
    # Use original data to transit route 1 through neural network
    conv_data_r1 = conv_block(
        data_in=data_in,
        nb_filter=384,
        size=(1, 1)
    )

    # Use original data to transit route 2 through neural network
    conv_data_r2 = conv_block(
        data_in=data_in,
        nb_filter=192,
        size=(1, 1)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=224,
        size=(1, 7)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=256,
        size=(7, 1)
    )

    # Use original data to transit route 3 through neural network
    conv_data_r3 = conv_block(
        data_in=data_in,
        nb_filter=192,
        size=(1, 1)
    )
    conv_data_r3 = conv_block(
        data_in=conv_data_r3,
        nb_filter=192,
        size=(7, 1)
    )
    conv_data_r3 = conv_block(
        data_in=conv_data_r3,
        nb_filter=224,
        size=(1, 7)
    )
    conv_data_r3 = conv_block(
        data_in=conv_data_r3,
        nb_filter=224,
        size=(7, 1)
    )
    conv_data_r3 = conv_block(
        data_in=conv_data_r3,
        nb_filter=256,
        size=(1, 7)
    )

    # Use original data to transit route 4 through neural network
    conv_data_r4 = AveragePooling2D(
        pool_size=3,
        strides=1,
        padding='same'
    )(inputs=data_in)
    conv_data_r4 = conv_block(
        data_in=conv_data_r4,
        nb_filter=128,
        size=(1, 1)
    )

    # Merge routes 1, 2, 3, and 4 into route 0
    conv_data_r0 = concatenate(
        inputs=[
            conv_data_r1,
            conv_data_r2,
            conv_data_r3,
            conv_data_r4
        ],
        axis=channel_axis
    )

    return conv_data_r0


def inception_c(
    data_in,
    channel_axis: int = -1
) -> any:
    """
    Build Inception C block

    :param data_in
    :param channel_axis: int

    :return x
    """
    # Use original data to transit route 1 through neural network
    conv_data_r1 = conv_block(
        data_in=data_in,
        nb_filter=256,
        size=(1, 1)
    )

    # Use original data to transit route 2 through neural network
    conv_data_r2 = conv_block(
        data_in=data_in,
        nb_filter=384,
        size=(1, 1)
    )

    # Use route 2 data to transit route 2.1 through neural network
    conv_data_r2_1 = conv_block(
        data_in=conv_data_r2,
        nb_filter=256,
        size=(1, 3)
    )

    # Use route 2 data to transit route 2.2 through neural network
    conv_data_r2_2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=256,
        size=(3, 1)
    )

    # Merge routes 2.1 and 2.2 back into route 2
    conv_data_r2 = concatenate(
        inputs=[
            conv_data_r2_1,
            conv_data_r2_2
        ],
        axis=channel_axis
    )

    # Use original data to transit route 3 through neural network
    conv_data_r3 = conv_block(
        data_in=data_in,
        nb_filter=384,
        size=(1, 1)
    )
    conv_data_r3 = conv_block(
        data_in=conv_data_r3,
        nb_filter=448,
        size=(3, 1)
    )
    conv_data_r3 = conv_block(
        data_in=conv_data_r3,
        nb_filter=512,
        size=(1, 3)
    )

    # Use route 3 data to transit route 3.1 through neural network
    conv_data_r3_1 = conv_block(
        data_in=conv_data_r3,
        nb_filter=256,
        size=(1, 3)
    )

    # Use route 3 data to transit route 3.2 through neural network
    conv_data_r3_2 = conv_block(
        data_in=conv_data_r3,
        nb_filter=256,
        size=(3, 1)
    )

    # Merge routes 3.1 and 3.2 back into route 3
    conv_data_r3 = concatenate(
        inputs=[
            conv_data_r3_1,
            conv_data_r3_2
        ],
        axis=channel_axis
    )

    # Use original data to transit route 4 through neural network
    conv_data_r4 = AveragePooling2D(
        pool_size=3,
        strides=1,
        padding='same'
    )(inputs=data_in)
    conv_data_r4 = conv_block(
        data_in=conv_data_r4,
        nb_filter=256,
        size=(1, 1)
    )

    # Merge routes 1, 2, 3, and 4 into route 0
    conv_data_r0 = concatenate(
        inputs=[
            conv_data_r1,
            conv_data_r2,
            conv_data_r3,
            conv_data_r4
        ],
        axis=channel_axis
    )

    return conv_data_r0


def reduction_a(
    data_in,
    channel_axis: int = -1
) -> any:
    """
    Build Reduction A block

    :param data_in
    :param channel_axis: int

    :return x
    """
    # Use original data to transit route 1 through neural network
    conv_data_r1 = conv_block(
        data_in=data_in,
        nb_filter=384,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    # Use original data to transit route 2 through neural network
    conv_data_r2 = conv_block(
        data_in=data_in,
        nb_filter=192,
        size=(1, 1)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=224,
        size=(3, 3)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=256,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    # Use original data to transit route 3 through neural network
    conv_data_r3 = MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='valid'
    )(inputs=data_in)

    # Merge routes 1, 2, and 3 into route 0
    conv_data_r0 = concatenate(
        inputs=[
            conv_data_r1,
            conv_data_r2,
            conv_data_r3
        ], axis=channel_axis
    )

    return conv_data_r0


def reduction_b(
    data_in,
    channel_axis: int = -1
) -> any:
    """
    Build Reduction B block

    :param data_in
    :param channel_axis: int

    :return x
    """
    # Use original data to transit route 1 through neural network
    conv_data_r1 = conv_block(
        data_in=data_in,
        nb_filter=192,
        size=(1, 1)
    )
    conv_data_r1 = conv_block(
        data_in=conv_data_r1,
        nb_filter=192,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    # Use original data to transit route 2 through neural network
    conv_data_r2 = conv_block(
        data_in=data_in,
        nb_filter=256,
        size=(1, 1)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=256,
        size=(1, 7)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=320,
        size=(7, 1)
    )
    conv_data_r2 = conv_block(
        data_in=conv_data_r2,
        nb_filter=320,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    # Use original data to transit route 3 through neural network
    conv_data_r3 = MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='valid'
    )(inputs=data_in)

    # Merge routes 1, 2, and 3 into route 0
    conv_data = concatenate(
        inputs=[
            conv_data_r1,
            conv_data_r2,
            conv_data_r3
        ],
        axis=channel_axis
    )
    return conv_data


def create_inception_v4(
    incept_dict: dict
):
    """
    Creates a inception v4 network

    :param incept_dict: dict

    :return: Keras Model with 1 input and 1 output
    """

    init = Input(batch_shape=(
        incept_dict['batch_size'],
        incept_dict['img_tgt_width'],
        incept_dict['img_tgt_height'],
        incept_dict['nbr_channels']
    ))

    channel_axis = -1

    # Input Shape is (x, 299, 299, 1)
    # training = False
    conv_data = inception_stem(
        data_in=init,
        channel_axis=channel_axis
    )

    # 4 x Inception A
    for i in range(4):
        conv_data = inception_a(
            data_in=conv_data,
            channel_axis=channel_axis
        )

    # Reduction A
    conv_data = reduction_a(
        data_in=conv_data,
        channel_axis=channel_axis
    )

    # 7 x Inception B
    for i in range(7):
        conv_data = inception_b(
            data_in=conv_data,
            channel_axis=channel_axis
        )

    # Reduction B
    conv_data = reduction_b(
        data_in=conv_data,
        channel_axis=channel_axis
    )

    # 3 x Inception C
    for i in range(3):
        conv_data = inception_c(
            data_in=conv_data,
            channel_axis=channel_axis
        )

    # Average Pooling
    conv_data = AveragePooling2D(
        pool_size=8
    )(inputs=conv_data)

    # Dropout
    conv_data = Dropout(
        rate=0.2
    )(inputs=conv_data)
    conv_data = Flatten()(inputs=conv_data)

    # Output
    out = Dense(
        units=incept_dict['nbr_classes'],
        activation=tf.nn.softmax
    )(inputs=conv_data)

    model = Model(
        inputs=init,
        outputs=out,
        name='Inception-v4'
    )

    return model
