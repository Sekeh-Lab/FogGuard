import os
import argparse
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
import shutil
import glob
import shutil

def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=False):
    """
    @brief      creates the annotation file list, containing the filenames alongside
    the annotations in one line

    @param      datapath: path to the data directory
    @param      data_type: train or test?
    @param      anno_path: path to the annotation file


    @return     the number of files added to the annotation list
    """
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    # classes = ['person', 'car', 'bus', 'bicycle',  'motorbike']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                if obj.find('name').text.lower().strip() in ['person', 'car', 'bus', 'bicycle',  'motorbike']:
                    class_ind = classes.index(obj.find('name').text.lower().strip())
                    xmin = bbox.find('xmin').text.strip()
                    xmax = bbox.find('xmax').text.strip()
                    ymin = bbox.find('ymin').text.strip()
                    ymax = bbox.find('ymax').text.strip()
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)

def convert_voc_annotation_to_yolo(data_path, data_type, anno_path, use_difficult_bbox=False):

    # classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    #            'train', 'tvmonitor']
    classes = ['person', 'car', 'bus', 'bicycle',  'motorbike']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                if obj.find('name').text.lower().strip() in ['person', 'car', 'bus', 'bicycle',  'motorbike']:
                    class_ind = classes.index(obj.find('name').text.lower().strip())
                    xmin = int(bbox.find('xmin').text.strip())
                    xmax = int(bbox.find('xmax').text.strip())
                    ymin = int(bbox.find('ymin').text.strip())
                    ymax = int(bbox.find('ymax').text.strip())
                    x, y, w, h = xml_to_yolo_bbox([xmin, ymin, xmax, ymax],
                                                  width, height)
                    # annotation += ' ' + ','.join([xmin, ymin, xmax, ymax,
                    # str(class_ind)])
                    annotation += f" {class_ind},{x:.6f},{y:.6f},{w:.6f},{h:.6f}"
                    # f.write(f"{class_ind} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


def create_coco_format_voc(data_src, data_type, data_dest, use_difficult_bbox=False):

    # classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    #            'train', 'tvmonitor']
    
    classes = ['person', 'car', 'bus', 'bicycle',  'motorbike']
    img_inds_file = os.path.join(data_src, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    # with open(anno_path, 'a') as f:
    for image_ind in image_inds:
        image_path = os.path.join(data_src, 'JPEGImages', image_ind + '.jpg')
        # save the file in new diretory
        image_dst = os.path.join(data_dest, data_type, 'images',
                                 image_ind + '.jpg')
        shutil.copyfile(image_path, image_dst)
        annotation = image_path
        anno_path = os.path.join(data_dest, data_type, 'labels', image_ind + '.txt')
        label_path = os.path.join(data_src, 'Annotations', image_ind + '.xml')
        root = ET.parse(label_path).getroot()
        objects = root.findall('object')
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        yolo_annot = ""
        for obj in objects:
            difficult = obj.find('difficult').text.strip()
            if (not use_difficult_bbox) and(int(difficult) == 1):
                continue
            bbox = obj.find('bndbox')
            if obj.find('name').text.lower().strip() in classes:
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = int(bbox.find('xmin').text.strip())
                xmax = int(bbox.find('xmax').text.strip())
                ymin = int(bbox.find('ymin').text.strip())
                ymax = int(bbox.find('ymax').text.strip())
                x, y, w, h = xml_to_yolo_bbox([xmin, ymin, xmax, ymax],
                                                width, height)
                # annotation += ' ' + ','.join([xmin, ymin, xmax, ymax,
                # str(class_ind)])
                yolo_annot += f"{class_ind} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                # f.write(f"{class_ind} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        # print(yolo_annot)
        open(anno_path, 'a').write(yolo_annot)
    return len(image_inds)


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]

def convert_coco_json_to_yolo_txt(output_path, json_file):

    path = make_folders(output_path)

    with open(json_file) as f:
        json_data = json.load(f)

    # write _darknet.labels, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "_darknet.labels")
    with open(label_file, "w") as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            f.write(f"{category_name}\n")

    for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"]
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Converting COCO Json to YOLO txt finished!")


def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]

def make_folders(path="output"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


if __name__ == '__main__':
    # for foggy conditions
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/lwy/work/code/tensorflow-yolov3/data/VOC/")
    parser.add_argument("--data_dst", default="/home/soheil/data/vocfog")
    parser.add_argument("--train_annotation", default="../data/dataset_fog/voc_norm_train.txt")
    parser.add_argument("--test_annotation",  default="../data/dataset_fog/voc_norm_test.txt")

    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    # if os.path.exists(flags.val_annotation):os.remove(flags.val_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)


    data_type = 'test'
    image_dst = os.path.join(flags.data_dst, data_type, 'images')
    label_dst = os.path.join(flags.data_dst, data_type, 'labels-20')
    make_folders(image_dst)
    make_folders(label_dst)
    # num1 = convert_voc_annotation_to_yolo(os.path.join(flags.data_path,
    #                                                    'train/VOCdevkit/VOC2007'),
    #                                       'trainval', flags.train_annotation,
    #                                       False)

    # num2 = convert_voc_annotation_to_yolo(os.path.join(flags.data_path,
    #                                            'train/VOCdevkit/VOC2012'),
    #                               'trainval', flags.train_annotation, False)

    # num3 = convert_voc_annotation_to_yolo(os.path.join(flags.data_path,
    #                                            'test/VOCdevkit/VOC2007'),
    #                               'test', flags.test_annotation, False)

    num4 = create_coco_format_voc(os.path.join(flags.data_path, 'test/VOCdevkit/VOC2007'),
                                  'test', flags.data_dst)

    # print('=> The number of image for train is: %d\tThe number of image for val
    # is:%d\tThe number of image for test is:%d' %(num1, num2, num3))
    # create_coco_format_voc(args.data_path, data_type, anno_path)



