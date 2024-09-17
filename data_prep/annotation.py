import os
import argparse
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
import shutil
# import glob

def convert_voc_annotation(data_root, data_type, anno_path, classes,
                           use_difficult_bbox=False):
    """
    @brief      creates the annotation file list, containing the filenames alongside
    the annotations in one line

    @param      datapath: path to the data directory
    @param      data_type: train or test?
    @param      anno_path: path to the annotation file

    @return     the number of files added to the annotation list
    """
    img_inds_file = os.path.join(data_root, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_root, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_root, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                if obj.find('name').text.lower().strip() in classes:
                    class_ind = classes.index(obj.find('name').text.lower().strip())
                    xmin = bbox.find('xmin').text.strip()
                    xmax = bbox.find('xmax').text.strip()
                    ymin = bbox.find('ymin').text.strip()
                    ymax = bbox.find('ymax').text.strip()
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


def convert_xml_annotation_to_txt(data_root, data_type, image_inds, annot_file,
                                  classes, use_difficult_bbox=False):
    """
    @brief      Create YOLO format annotation, one .txt per image and one
    object per line.

    @details    detailed description

    @param      param

    @return     return type
    """
    image_dirs = []
    for image_ind in image_inds:
        image_path = os.path.join(data_root, 'images', image_ind + '.jpg')
        anno_path = os.path.join(data_root, 'labels-' + str(len(classes)),
                                 image_ind + '.txt')
        label_path = os.path.join(data_root, 'Annotations', image_ind + '.xml')
        root = ET.parse(label_path).getroot()
        objects = root.findall('object')
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        yolo_annot = ""
        for obj in objects:
            difficult = obj.find('difficult').text.strip()
            if (not use_difficult_bbox) and (int(difficult) == 1):
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
                # annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
                # annotation += f" {class_ind},{x:.6f},{y:.6f},{w:.6f},{h:.6f}"
                yolo_annot += f"{class_ind} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                # f.write(f"{class_ind} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        # print(yolo_annot)
        # f.write(annotation + "\n")
        # only create the txt file if there is an object in the image
        # if yolo_annot != "":
        open(anno_path, 'w').write(yolo_annot)
        image_dirs.append(image_path)
    open(annot_file, 'w').write("\n".join(image_dirs))
    print("Number of images: ", len(image_inds))


def create_coco_format_voc(data_src, data_type, data_dest, classes, use_difficult_bbox=False):
    """ Create a single .txt annotation file for all the images
    """
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
    # converts xmin, ymin, xmax, ymax to x, y, w, h
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


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        names = fp.read().splitlines()

    print("Classes: ", names)
    return names


def voc_image_inds(data_root, data_type):
    img_inds_file = os.path.join(data_root, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]
    return image_inds


def rtts_image_inds(data_root, data_type):
    img_inds_file = os.path.join(data_root, "config", data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip().split("/")[-1][:-4] for line in txt]
    return image_inds


if __name__ == '__main__':

    # classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    #            'train', 'tvmonitor']

    # classes = ['person', 'car', 'bus', 'bicycle',  'motorbike']
    # for foggy conditions
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_root",
                        default="/home/soheil/data/voc/VOCdevkit/VOC2012",
                        help="The path where the .xml files are located")

    parser.add_argument("--annot_file", default="../data/config/train.txt",
                        help="The image directories file, containing all the image names")

    parser.add_argument("--data_type", default="trainval",
                        choices=["trainval", "train", "test", "valid"],
                        help="trainval for training and test for evaluation.")

    parser.add_argument("--class_names", default="../config/voc-5.names",
                        help="List of items in the dataset")

    parser.add_argument("--dataset_name", default="voc", choices=["voc", "rtts"],
                        help="Name of dataset")

    flags = parser.parse_args()
    classes = load_classes(flags.class_names)

    if flags.dataset_name == "voc":
        image_inds = voc_image_inds(flags.labels_root, flags.data_type)

    elif flags.dataset_name == "rtts":
        image_inds = rtts_image_inds(flags.labels_root, flags.data_type)
        flags.labels_root = os.path.join(flags.labels_root, flags.data_type)

    else:
        exit("Bad dataset")

    label_dst = os.path.join(flags.labels_root, f'labels-{len(classes)}')
    make_folders(label_dst)
    convert_xml_annotation_to_txt(flags.labels_root, flags.data_type,
                                  image_inds, flags.annot_file, classes)
