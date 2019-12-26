import argparse
import tensorflow as tf
import keras
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras import losses
import random
import numpy as np
from keras.optimizers import Adam
from PIL import Image
from keras.models import load_model
import glob
import os
import cv2  #OpenCVのインポート

class DataSet(object):
    CATEGORY = (
        "1",
        "2",
        "3"
    )

def generate_paths(dir_original, dir_segmented):
    paths_original = glob.glob(dir_original + "/pred.jpg") 
    paths_segmented = glob.glob(dir_segmented + "/segment.png")
    filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
    paths_original = list(map(lambda filename: filename , paths_original))
    #paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))

    return paths_original, paths_segmented

def crop_to_square(image):
    size = min(image.size)
    left, upper = (image.width - size) // 2, (image.height - size) // 2
    right, bottom = (image.width + size) // 2, (image.height + size) // 2
    return image.crop((left, upper, right, bottom))





def image_generator(file_paths, init_size=None, normalization=True, antialias=False):
    for file_path in file_paths:
        print(file_path)
        if file_path.endswith(".png") or file_path.endswith(".jpg"):
                # open a image
            image = Image.open(file_path)
                # to square
            image = crop_to_square(image)
                # resize by init_size
            if init_size is not None and init_size != image.size:
                if antialias:
                    image = image.resize(init_size, Image.ANTIALIAS)
                else:
                    image = image.resize(init_size)
                # delete alpha channel
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = np.asarray(image)
            if normalization:
                image = image / 255.0
            yield image

def extract_images(paths_original, paths_segmented, init_size, one_hot):
     images_original, images_segmented = [], []
     # Load images from directory_path using generator
     print("Loading original images", end="", flush=True)
     for image in image_generator(paths_original, init_size, antialias=True):
         images_original.append(image)
         if len(images_original) % 200 == 0:
              print(".", end="", flush=True)
     print(" Completed", flush=True)
     print("Loading segmented images", end="", flush=True)
     for image in image_generator(paths_segmented, init_size, normalization=False):
         images_segmented.append(image)
         if len(images_segmented) % 200 == 0:   
             print(".", end="", flush=True)
     print(" Completed")
     #assert len(images_original) == len(images_segmented)

        # Cast to ndarray
     images_original = np.asarray(images_original, dtype=np.float32)
     images_segmented = np.asarray(images_segmented, dtype=np.uint8)

        # Change indices which correspond to "void" from 255
     images_segmented = np.where(images_segmented == 255, len(DataSet.CATEGORY)-1, images_segmented)

        # One hot encoding using identity matrix.
     if one_hot:
         print("Casting to one-hot encoding... ", end="", flush=True)
         identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
         images_segmented = identity[images_segmented]
         print("Done") 
     else:
         pass

     return images_original, images_segmented

def save_image_from_ndarray(self,test_set, palette, index_void=None):
    assert len(train_set) == len(test_set) == 3
    test_image = Reporter.get_imageset(test_set[0], test_set[1], test_set[2], palette, index_void)
    self.save_image(test_image)

def cast_to_pil(ndarray, palette, index_void=None):
    res = np.argmax(ndarray, axis=2)
    if index_void is not None:
        res = np.where(res == index_void, 0, res)
    image = Image.fromarray(np.uint8(res), mode="P")
    image.putpalette(palette)
    return image

def count_pits():
    fname="./predict_data.png" #開く画像ファイル名
    threshold=30 #二値化閾値

    img_color= cv2.imread(fname) #画像を読み出しオブジェクトimg_colorに代入
    img_gray = cv2.imread(fname,cv2.IMREAD_GRAYSCALE) #画像をグレースケールで読み出しオブジェクトimg_grayに代入

    ret, img_binary= cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY) #オブジェクトimg_grayを閾値threshold(127)で二値化しimg_binaryに代入
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #img_binaryを輪郭抽出
    cv2.drawContours(img_color, contours, -1, (0,0,255), 2) #抽出した輪郭を赤色でimg_colorに重ね書き
    print("///////////////////////////")
    print("\n")
    print("Pits numbers:",len(contours)) #抽出した輪郭の個数を表示する
    print("\n")
    print("///////////////////////////")
    #cv2.imshow("contours",img_color) #別ウィンドウを開き(ウィンドウ名 "contours")オブジェクトimg_colorを表示
    #cv2.waitKey(0) #キー入力待ち
    #cv2.destroyAllWindows() #ウインドウを閉じる



def predict():
    # 訓練とテストデータを読み込みます
    # Load train and test datas
    dir_original ="./image" 
    dir_segmented ="./segmentation" 
    init_size=(512, 512)
    one_hot=True
    
    paths_original, paths_segmented = generate_paths(dir_original, dir_segmented)
    images_original, images_segmented = extract_images(paths_original, paths_segmented, init_size, one_hot)
    image_sample_palette = Image.open(paths_segmented[0])
    palette = image_sample_palette.getpalette()

    testx=images_original
    testy=images_segmented

    model = load_model("./model_200_0.02.h5",compile=False)
    #model.summary()
    outputs_test = model.predict(testx, verbose=0).reshape(512,512,3)
    print(outputs_test.shape)
    image_out = cast_to_pil(outputs_test, palette, index_void=len(DataSet.CATEGORY)-1)
    image_out = image_out.convert("RGB")
    
    image_out.save("./predict_data.png")
    count_pits() 



if __name__ == '__main__':
    predict()
