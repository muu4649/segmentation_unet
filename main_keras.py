import argparse
import tensorflow as tf

from util import loader as ld
from util.model_keras import build_model, cross_entropy


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


def train(args):
    # 訓練とテストデータを読み込みます
    # Load train and test datas
    train_gen, test_gen = load_dataset(train_rate=args.trainrate)

    epochs = args.epoch
    batch_size = args.batchsize
    is_augment = args.augmentation

    model = build_model(output_class_num=ld.DataSet.length_category(), l2_reg=args.l2reg)
    model.compile(
        loss=cross_entropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.001)
    )

    train_sequence = SequenceGenerator(train_gen, batch_size=batch_size, is_augment=is_augment)
    test_sequence = SequenceGenerator(test_gen, batch_size=batch_size, is_augment=False)

    model.fit_generator(
        generator=train_sequence,
        validation_data=test_sequence,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='models')]
    )


def get_parser():
    args = argparse.ArgumentParser(
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

    return args


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
