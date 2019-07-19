import argparse
import colorsys
import cv2
import numpy as np
import tensorflow as tf
from util.loader import DataSet, Loader
from util.model_keras import cross_entropy


def gen_fixed_colors(num_of_classes):
    hsv_tuples = [(x / num_of_classes, 1., 1.) for x in range(num_of_classes)]
    cs = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    cs = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), cs))
    np.random.seed(10103)
    np.random.shuffle(cs)
    np.random.seed(None)
    return np.array(cs)


colors = gen_fixed_colors(DataSet.length_category())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file_path', type=str)
    args = parser.parse_args()

    image_path_list = ['data_set/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg']
    input_shape = (128, 128)
    model = tf.keras.models.load_model(args.model_file_path, custom_objects={'cross_entropy': cross_entropy})
    for image in Loader.image_generator(image_path_list, input_shape):
        image = image[np.newaxis, ...]
        output = model.predict(image)
        output = np.argmax(output, axis=-1)

        output = colors[output]
        cv2.imwrite('test.png', output[0])


if __name__ == '__main__':
    main()
