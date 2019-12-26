import tensorflow as tf
import keras
from keras.layers import *
from keras.regularizers import l2
from keras.models import *
from util.BilinearUpSampling import *
from sklearn.metrics import roc_auc_score
layers = keras.layers
K = tf.keras.backend

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def original_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

def original_accuracy(y_true, y_pred):#pixel wise accuracy 正解画素の比率
    correct_prediction = tf.equal(tf.argmax(y_pred, 3), tf.argmax(y_true, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def auc(y_true, y_pred):
    auc = tf.metrics.auc(tf.argmax(y_true), tf.argmax(y_pred))[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

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



def FCN_Vgg16_32(classes ,size=None ,weight_decay=0., batch_momentum=0.9, batch_shape=None):
    input_shape = (None, None, 3) if size is None else size + (3, )
    inputs = layers.Input(input_shape)


    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)
    model = Model(img_input, x)
    return model

















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
    model = build_model(output_class_num=100, size=(512, 512), l2_reg=0.0001)
    model.summary()


if __name__ == '__main__':
    test()
