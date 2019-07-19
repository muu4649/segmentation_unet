import argparse
import random
import tensorflow as tf

from util import loader as ld
from util.model_keras import build_model, cross_entropy
from util import repoter as rp


def load_dataset(train_rate):
    loader = ld.Loader(dir_original="data_set/VOCdevkit/VOC2012/JPEGImages",
                       dir_segmented="data_set/VOCdevkit/VOC2012/SegmentationClass")
    return loader.load_train_test(train_rate=train_rate, shuffle=False)


class SequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator, batch_size, is_augment):
        self.data_gen = generator
        self.batch_size = batch_size
        self.is_augment = is_augment
        self.iterator = self.generator()

    def __len__(self):
        return self.data_gen.length // self.batch_size

    def __getitem__(self, item):
        return next(self.iterator)

    def generator(self):
        while True:
            for batch in self.data_gen(self.batch_size, self.is_augment):
                yield batch.images_original, batch.images_segmented


def train(parser):
    # 訓練とテストデータを読み込みます
    # Load train and test datas
    train_gen, test = load_dataset(train_rate=parser.trainrate)
    valid = train_gen.perm(0, 30)
    test = test.perm(0, 150)

    epochs = parser.epoch
    batch_size = parser.batchsize
    is_augment = parser.augmentation

    model = build_model(output_class_num=ld.DataSet.length_category(), l2_reg=parser.l2reg)
    model.compile(
        loss=cross_entropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.001)
    )

    train_sequence = SequenceGenerator(train_gen, batch_size=batch_size, is_augment=is_augment)

    model.fit_generator(
        generator=train_sequence,
        epochs=epochs,
        callbacks=tf.keras.callbacks.ModelCheckpoint
    )


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int, default=250, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.85, help='Training rate')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float, default=0.0001, help='L2 regularization')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
