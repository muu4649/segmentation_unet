import colorsys
import cv2
import numpy as np
import tensorflow as tf
from util import model
from util.loader import Loader, DataSet


def gen_fixed_colors(num_of_classes):
    hsv_tuples = [(x / num_of_classes, 1., 1.) for x in range(num_of_classes)]
    cs = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    cs = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), cs))
    np.random.seed(10103)
    np.random.shuffle(cs)
    np.random.seed(None)
    return np.array(cs)


colors = gen_fixed_colors(DataSet.length_category())

model_unet = model.UNet(l2_reg=0.0001).model

sess = tf.Session()
# tf.global_variables_initializer()

saver = tf.train.Saver()
saver.restore(sess, './models/weights_epoch_15_loss_0.940_val_loss_1.151_')

image_path_list = ['data_set/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg']
input_shape = (128, 128)

for image in Loader.image_generator(image_path_list, input_shape, antialias=True):
    image = image[np.newaxis, ...]
    dummy = np.zeros((1, ) + input_shape + (DataSet.length_category(), ), np.float32)
    print('dummy', dummy.shape)
    output = sess.run(model_unet.outputs, feed_dict={model_unet.inputs: image,
                                                     model_unet.is_training: False})
    print(output.shape)
    output = np.argmax(output[0], axis=-1)
    print(output[:10, :10])
    print(colors[0])
    output_img = colors[output]

    cv2.imwrite('test_img.png', output_img)
    print(output.shape)



