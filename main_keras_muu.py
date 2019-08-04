import argparse
import tensorflow as tf
import keras
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from util import loader as ld
from util.model_keras import build_model, cross_entropy
from keras import losses
from util import repoter as rp
import random
import numpy as np

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

class SavePredictionCallback(keras.callbacks.Callback):
     def __init__(self, trains,tests,reporters, **kwargs):
         super().__init__(**kwargs)
         self.trains = trains
         self.tests = tests
         self.reporter=reporters

     def on_epoch_end(self, epoch, logs={}):
         if epoch % 5== 0:
             print("validation start")
             #idx_train = random.randrange(160)
             #idx_test = random.randrange(40)
             idx_train = 1#2番目の画像
             idx_test = 1
             print(self.trains.images_original.shape)

             outputs_train = self.model.predict(self.trains.images_original[idx_train].reshape(1,64,64,3), verbose=0)
             outputs_test = self.model.predict(self.tests.images_original[idx_test].reshape(1,64,64,3), verbose=0)
        
             train_set = [self.trains.images_original[idx_train], outputs_train[0], self.trains.images_segmented[idx_train]]
             test_set = [self.tests.images_original[idx_test], outputs_test[0], self.tests.images_segmented[idx_test]]
             self.reporter.save_image_from_ndarray(train_set, test_set, self.trains.palette, epoch,
                                                 index_void=len(ld.DataSet.CATEGORY)-1)

def train(args):
    # 訓練とテストデータを読み込みます
    # Load train and test datas
    train_gen, test_gen = load_dataset(train_rate=args.trainrate)
    trainx=train_gen.images_original
    trainy=train_gen.images_segmented
    testx=test_gen.images_original
    testy=test_gen.images_segmented
    print(trainx.shape)
    print(testx.shape)
        # Create Reporter Object
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "test"])
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    epochs = args.epoch
    batch_size = args.batchsize
    is_augment = args.augmentation

    model = build_model(output_class_num=ld.DataSet.length_category(), l2_reg=args.l2reg)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(lr=0.0005),
        metrics=['accuracy']
    )

    train_sequence = SequenceGenerator(train_gen, batch_size=batch_size, is_augment=is_augment)
    test_sequence = SequenceGenerator(test_gen, batch_size=batch_size, is_augment=False)
    
    callbacks_list =[
            
        SavePredictionCallback(train_gen,test_gen,reporter),
        ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5, min_lr=1e-15, verbose=1, mode='auto',cooldown=0),
        ModelCheckpoint(filepath='./model_{epoch:02d}_{val_loss:.2f}.h5',monitor='acc', save_best_only=True, verbose=1, mode='auto')
        ]

   # model.fit_generator(
   #     generator=train_sequence,
   #     validation_data=test_sequence,
   #     epochs=epochs,
   #     callbacks=callbacks_list
   # )
    model.summary()
    history= model.fit(
            x=[trainx],y=[trainy],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([testx],[testy]),
            callbacks=callbacks_list
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
    parser.add_argument('-b', '--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.85, help='Training rate')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Data augmentation flaga')
    parser.add_argument('-r', '--l2reg', type=float, default=0.0001, help='L2 regularization')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
