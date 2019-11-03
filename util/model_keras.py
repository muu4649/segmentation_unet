import tensorflow as tf
import keras
layers = keras.layers
K = tf.keras.backend


def original_loss(y_true, y_pred, smooth=1):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))


def build_model(output_class_num, size=None, l2_reg=None):
    input_shape = (None, None, 3) if size is None else size + (3, )
    inputs = layers.Input(input_shape)
    conv1_1 = base_conv(inputs, filters=32, l2_reg_scale=l2_reg)

    conv1_2 = base_conv(conv1_1, filters=32, l2_reg_scale=l2_reg)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1_2)
    
    # 1/2, 1/2, 64
    conv2_1 = base_conv(pool1, filters=64, l2_reg_scale=l2_reg)
    conv2_2 = base_conv(conv2_1, filters=64, l2_reg_scale=l2_reg)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2_2)
    
    # 1/4, 1/4, 128
    conv3_1 = base_conv(pool2, filters=128, l2_reg_scale=l2_reg)
    conv3_2 = base_conv(conv3_1, filters=128, l2_reg_scale=l2_reg)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3_2)
    
    # 1/8, 1/8, 256
    conv4_1 = base_conv(pool3, filters=256, l2_reg_scale=l2_reg)
    conv4_2 = base_conv(conv4_1, filters=256, l2_reg_scale=l2_reg)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4_2)

    # 1/16, 1/16, 512
    conv5_1 = base_conv(pool4, filters=512, l2_reg_scale=l2_reg)
    conv5_2 = base_conv(conv5_1, filters=512, l2_reg_scale=l2_reg)
    concated1 = layers.Concatenate()([conv_transpose(conv5_2, filters=512, l2_reg_scale=l2_reg), conv4_2])

    conv_up1_1 = base_conv(concated1, filters=256, l2_reg_scale=l2_reg)
    conv_up1_2 = base_conv(conv_up1_1, filters=256, l2_reg_scale=l2_reg)
    concated2 = layers.Concatenate()([conv_transpose(conv_up1_2, filters=256, l2_reg_scale=l2_reg), conv3_2])

    conv_up2_1 = base_conv(concated2, filters=128, l2_reg_scale=l2_reg)
    conv_up2_2 = base_conv(conv_up2_1, filters=128, l2_reg_scale=l2_reg)
    concated3 = layers.Concatenate()([conv_transpose(conv_up2_2, filters=128, l2_reg_scale=l2_reg), conv2_2])

    conv_up3_1 = base_conv(concated3, filters=64, l2_reg_scale=l2_reg)
    conv_up3_2 = base_conv(conv_up3_1, filters=64, l2_reg_scale=l2_reg)
    concated4 = layers.Concatenate()([conv_transpose(conv_up3_2, filters=64, l2_reg_scale=l2_reg), conv1_2])

    conv_up4_1 = base_conv(concated4, filters=32, l2_reg_scale=l2_reg)
    conv_up4_2 = base_conv(conv_up4_1, filters=32, l2_reg_scale=l2_reg)
    outputs = base_conv(conv_up4_2, filters=output_class_num, kernel_size=(1, 1))

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def base_conv(inputs, filters, kernel_size=(3, 3), activation='relu', l2_reg_scale=None):
    reg = keras.regularizers.l2(l2_reg_scale) if l2_reg_scale is not None else None
    conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same',
                         kernel_regularizer=reg)(inputs)
    conv = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True)(conv)
    return conv


def conv_transpose(inputs, filters, l2_reg_scale=None):
    reg = keras.regularizers.l2(l2_reg_scale) if l2_reg_scale is not None else None
    conv = layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), activation='relu',
                                  padding='same', kernel_regularizer=reg)(inputs)
    return conv


def cross_entropy(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    # return -K.sum(y_true * K.log(y_pred + 1e-7), axis=-1)


def test():
    model = build_model(output_class_num=100, size=(64, 64), l2_reg=0.0001)
    model.summary()


if __name__ == '__main__':
    test()
