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
):
    x = Conv2D(
        filters=nb_filter,
        kernel_size=size,
        strides=strides,
        padding=padding
    )(data_in)
    x = BatchNormalization(
        axis=channel_axis
    )(x)
    x = Activation(
        activation=tf.nn.relu
    )(x)

    return x


def inception_stem(
    data_in,
    channel_axis: int = -1
):
    x = conv_block(
        data_in=data_in,
        nb_filter=32,
        size=(3, 3),
        strides=2,
        padding='valid'
    )
    x = conv_block(
        data_in=x,
        nb_filter=32,
        size=(3, 3),
        padding='valid'
    )
    x = conv_block(
        data_in=x,
        nb_filter=64,
        size=(3, 3)
    )

    x1 = MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='valid'
    )(x)
    x2 = conv_block(
        data_in=x,
        nb_filter=96,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    x = concatenate(
        inputs=[
            x1,
            x2
        ],
        axis=channel_axis
    )

    x1 = conv_block(
        data_in=x,
        nb_filter=64,
        size=(1, 1)
    )
    x1 = conv_block(
        data_in=x1,
        nb_filter=96,
        size=(3, 3),
        padding='valid'
    )

    x2 = conv_block(
        data_in=x,
        nb_filter=64,
        size=(1, 1)
    )
    x2 = conv_block(
        data_in=x2,
        nb_filter=64,
        size=(1, 7)
    )
    x2 = conv_block(
        data_in=x2,
        nb_filter=64,
        size=(7, 1)
    )
    x2 = conv_block(
        data_in=x2,
        nb_filter=96,
        size=(3, 3),
        padding='valid'
    )

    x = concatenate(
        inputs=[
            x1,
            x2
        ],
        axis=channel_axis
    )

    x1 = conv_block(
        data_in=x,
        nb_filter=192,
        size=(3, 3),
        strides=2,
        padding='valid'
    )
    x2 = MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='valid'
    )(x)

    x = concatenate(
        inputs=[
            x1,
            x2
        ],
        axis=channel_axis
    )

    return x


def inception_a(
    data_in,
    channel_axis: int = -1
):
    a1 = conv_block(
        data_in=data_in,
        nb_filter=96,
        size=(1, 1)
    )

    a2 = conv_block(
        data_in=data_in,
        nb_filter=64,
        size=(1, 1)
    )
    a2 = conv_block(
        data_in=a2,
        nb_filter=96,
        size=(3, 3)
    )

    a3 = conv_block(
        data_in=data_in,
        nb_filter=64,
        size=(1, 1)
    )
    a3 = conv_block(
        data_in=a3,
        nb_filter=96,
        size=(3, 3)
    )
    a3 = conv_block(
        data_in=a3,
        nb_filter=96,
        size=(3, 3)
    )

    a4 = AveragePooling2D(
        pool_size=3,
        strides=1,
        padding='same'
    )(data_in)
    a4 = conv_block(
        data_in=a4,
        nb_filter=96,
        size=(1, 1)
    )

    m = concatenate(
        inputs=[
            a1,
            a2,
            a3,
            a4
        ],
        axis=channel_axis
    )

    return m


def inception_b(
    data_in,
    channel_axis: int = -1
):
    b1 = conv_block(
        data_in=data_in,
        nb_filter=384,
        size=(1, 1)
    )

    b2 = conv_block(
        data_in=data_in,
        nb_filter=192,
        size=(1, 1)
    )
    b2 = conv_block(
        data_in=b2,
        nb_filter=224,
        size=(1, 7)
    )
    b2 = conv_block(
        data_in=b2,
        nb_filter=256,
        size=(7, 1)
    )

    b3 = conv_block(
        data_in=data_in,
        nb_filter=192,
        size=(1, 1)
    )
    b3 = conv_block(
        data_in=b3,
        nb_filter=192,
        size=(7, 1)
    )
    b3 = conv_block(
        data_in=b3,
        nb_filter=224,
        size=(1, 7)
    )
    b3 = conv_block(
        data_in=b3,
        nb_filter=224,
        size=(7, 1)
    )
    b3 = conv_block(
        data_in=b3,
        nb_filter=256,
        size=(1, 7)
    )

    b4 = AveragePooling2D(
        pool_size=3,
        strides=1,
        padding='same'
    )(data_in)
    b4 = conv_block(
        data_in=b4,
        nb_filter=128,
        size=(1, 1)
    )

    m = concatenate(
        inputs=[
            b1,
            b2,
            b3,
            b4
        ],
        axis=channel_axis
    )

    return m


def inception_c(
    data_in,
    channel_axis: int = -1
):
    c1 = conv_block(
        data_in=data_in,
        nb_filter=256,
        size=(1, 1)
    )

    c2 = conv_block(
        data_in=data_in,
        nb_filter=384,
        size=(1, 1)
    )
    c2_1 = conv_block(
        data_in=c2,
        nb_filter=256,
        size=(1, 3)
    )
    c2_2 = conv_block(
        data_in=c2,
        nb_filter=256,
        size=(3, 1)
    )
    c2 = concatenate(
        inputs=[
            c2_1,
            c2_2
        ],
        axis=channel_axis
    )

    c3 = conv_block(
        data_in=data_in,
        nb_filter=384,
        size=(1, 1)
    )
    c3 = conv_block(
        data_in=c3,
        nb_filter=448,
        size=(3, 1)
    )
    c3 = conv_block(
        data_in=c3,
        nb_filter=512,
        size=(1, 3)
    )
    c3_1 = conv_block(
        data_in=c3,
        nb_filter=256,
        size=(1, 3)
    )
    c3_2 = conv_block(
        data_in=c3,
        nb_filter=256,
        size=(3, 1)
    )
    c3 = concatenate(
        inputs=[
            c3_1,
            c3_2
        ],
        axis=channel_axis
    )

    c4 = AveragePooling2D(
        pool_size=3,
        strides=1,
        padding='same'
    )(data_in)
    c4 = conv_block(
        data_in=c4,
        nb_filter=256,
        size=(1, 1)
    )

    m = concatenate(
        inputs=[
            c1,
            c2,
            c3,
            c4
        ],
        axis=channel_axis
    )

    return m


def reduction_a(
    data_in,
    channel_axis: int = -1
):
    r1 = conv_block(
        data_in=data_in,
        nb_filter=384,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    r2 = conv_block(
        data_in=data_in,
        nb_filter=192,
        size=(1, 1)
    )
    r2 = conv_block(
        data_in=r2,
        nb_filter=224,
        size=(3, 3)
    )
    r2 = conv_block(
        data_in=r2,
        nb_filter=256,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    r3 = MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='valid'
    )(data_in)

    m = concatenate(
        inputs=[
            r1,
            r2,
            r3
        ], axis=channel_axis
    )

    return m


def reduction_b(
    data_in,
    channel_axis: int = -1
):
    r1 = conv_block(
        data_in=data_in,
        nb_filter=192,
        size=(1, 1)
    )
    r1 = conv_block(
        data_in=r1,
        nb_filter=192,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    r2 = conv_block(
        data_in=data_in,
        nb_filter=256,
        size=(1, 1)
    )
    r2 = conv_block(
        data_in=r2,
        nb_filter=256,
        size=(1, 7)
    )
    r2 = conv_block(
        data_in=r2,
        nb_filter=320,
        size=(7, 1)
    )
    r2 = conv_block(
        data_in=r2,
        nb_filter=320,
        size=(3, 3),
        strides=2,
        padding='valid'
    )

    r3 = MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='valid'
    )(data_in)

    m = concatenate(
        inputs=[
            r1,
            r2,
            r3
        ],
        axis=channel_axis
    )
    return m


def create_inception_v4(
    tf_dict: dict
):
    """
    Creates a inception v4 network

    :param tf_dict: dict

    :return: Keras Model with 1 input and 1 output
    """

    init = Input(batch_shape=(
        tf_dict['batch_size'],
        tf_dict['img_tgt_width'],
        tf_dict['img_tgt_height'],
        tf_dict['nbr_channels']
    ))

    channel_axis = -1

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    # training = False
    x = inception_stem(
        data_in=init,
        channel_axis=channel_axis
    )

    # 4 x Inception A
    for i in range(4):
        x = inception_a(
            data_in=x,
            channel_axis=channel_axis
        )

    # Reduction A
    x = reduction_a(
        data_in=x,
        channel_axis=channel_axis
    )

    # 7 x Inception B
    for i in range(7):
        x = inception_b(
            data_in=x,
            channel_axis=channel_axis
        )

    # Reduction B
    x = reduction_b(
        data_in=x,
        channel_axis=channel_axis
    )

    # 3 x Inception C
    for i in range(3):
        x = inception_c(
            data_in=x,
            channel_axis=channel_axis
        )

    # Average Pooling
    x = AveragePooling2D(
        pool_size=8
    )(x)

    # Dropout
    x = Dropout(
        rate=0.2
    )(x)
    x = Flatten()(x)

    # Output
    out = Dense(
        units=tf_dict['nbr_classes'],
        activation=tf.nn.softmax
    )(x)

    model = Model(
        inputs=init,
        outputs=out,
        name='Inception-v4'
    )

    return model
