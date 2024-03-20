import numpy as np
import os
import cv2
import math
from numba import jit
import random
import argparse
from pathlib import Path


# only use the image including the labeled instance objects for training
def load_annotations(annot_path):
    print(annot_path)
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if
                       len(line.strip().split()[1:]) != 0]
    return annotations


@jit()
def AddHaz_loop(img_f, center, size, beta, A):
    (row, col, chs) = img_f.shape

    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f

# print('*****************Add haze offline***************************')
def parse_annotation(annotation, img_dir):

    line = annotation.split()
    image_path = line[0]
    annots = line[1:]
    annots = [ann.replace(",", " ") for ann in annots]
    yolo_annot = "\n".join(annots)
    # print(image_path)
    img_name = image_path.split('/')[-1]
    # print(img_name)
    image_name = img_name.split('.')[0]
    # print(image_name)
    image_name_index = img_name.split('.')[1]
    # print(image_name_index)

#'/data/vdd/liuwenyu/data_vocfog/train/JPEGImages/'
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = cv2.imread(image_path)

    for i in range(10):
        img_f = image/255
        (row, col, chs) = image.shape
        A = 0.5  
        # beta = 0.08  
        beta = 0.01 * i + 0.05
        size = math.sqrt(max(row, col)) 
        center = (row // 2, col // 2)  
        # foggy_image = AddHaz_loop(img_f, center, size, beta, A)
        # img_f = np.clip(foggy_image*255, 0, 255)
        # img_f = img_f.astype(np.uint8)
        image_name_converted = image_name + '_' + ("%.2f"%beta)
        img_name = img_dir + image_name_converted + '.' + image_name_index
        image_name_converted += ".txt"
        label_path = Path(img_dir).parents[0] / "labels" / image_name_converted
        #img_name = '/data/vdd/liuwenyu/data_vocfog/val/JPEGImages/' + image_name \
        #   + '_' + ("%.2f"%beta) + '.' + image_name_index
        print(label_path)
        with open(label_path, "w") as f:
            f.write(yolo_annot)

        # cv2.imwrite(img_name, img_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument("--data_path", default="/home/soheil/Sync/ia-yolo/data/dataset/voc_norm_train.txt")
    parser.add_argument('--train_path', dest='train_path', type=str,
                        default='./data/dataset_fog/voc_norm_train.txt',
                        help='folder of the training data')
    parser.add_argument('--val_path', dest='val_path', type=str,
                        default='./data/dataset_fog/voc_norm_test.txt',
                        help='folder of the training data') 
    parser.add_argument('--test_path', dest='test_path', type=str,
                        default='./data/dataset_fog/voc_norm_test.txt',
                        help='folder of the training data') 
    parser.add_argument('--vocfog_traindata_dir', dest='vocfog_traindata_dir',
                        default='/home/soheil/data/data_vocfog/train/images/',
                        help='the dir contains ten levels synthetic foggy images')
    parser.add_argument('--vocfog_testdata_dir', dest='vocfog_testdata_dir',
                        default='/home/soheil/data/data_vocfog/test/images/',
                        help='the dir contains ten levels synthetic foggy images')
    flags = parser.parse_args()

    train_label_dir = Path(os.path.dirname(flags.vocfog_traindata_dir)).parents[0] / "labels"
    test_label_dir = Path(os.path.dirname(flags.vocfog_testdata_dir)).parents[0] / "labels"

    if not os.path.exists(flags.vocfog_traindata_dir):
        os.makedirs(flags.vocfog_traindata_dir)

    if not os.path.exists(flags.vocfog_testdata_dir):
        os.makedirs(flags.vocfog_testdata_dir)

    if not os.path.exists(train_label_dir):
        os.makedirs(train_label_dir)

    if not os.path.exists(test_label_dir):
        os.makedirs(test_label_dir)


    train_an = load_annotations(flags.train_path)
    test_an = load_annotations(flags.test_path)
    #an = load_annotations('/home/liuwenyu.lwy/code/defog_yolov3/data/dataset/voc_norm_test.txt')
    train_ll = len(train_an)
    test_ll = len(test_an)
    print(train_ll)
    print(test_ll)
    train_img_dir = flags.vocfog_traindata_dir # '/data/vdd/liuwenyu/data_vocfog/train/JPEGImages/'
    test_img_dir = flags.vocfog_testdata_dir # '/data/vdd/liuwenyu/data_vocfog/train/JPEGImages/'
    for j in range(train_ll):
        parse_annotation(train_an[j], train_img_dir)

    for j in range(test_ll):
        parse_annotation(test_an[j], test_img_dir)
