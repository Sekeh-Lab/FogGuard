import torch.nn.functional as F
import torch
import glob
import math
import random
# from numba import jit
from tqdm import tqdm
import os
import pathlib
import pandas as pd
from torchvision.io import read_image
import warnings
import numpy as np
from collections import defaultdict
import torch.utils.data
import torchvision

import PIL.Image
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from utils.utils import worker_seed_set
import matplotlib.pyplot as plt
from torchvision import datasets, models

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None,
                 path=None, fog_level=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_dir_list = [os.path.join(img_dir, item.name)
                             for item in pathlib.Path(img_dir).glob("*.jpg")
                             if not item.is_dir()][:100]
        self.transform = transform
        self.target_transform = target_transform
        self.image_list = []
        self.fog_level = fog_level
        self.load_images()


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        images = self.image_list[idx]
        trainsformed_image_list = []
        # image = Image.open(self.img_dir_list[idx]).convert('RGB')
        # image = read_image(self.img_dir_list[idx])
        if self.transform:
            for image in images:
                trainsformed_image_list.append(self.transform(image))
        if self.target_transform:
            label = self.target_transform(label)

        return tuple(trainsformed_image_list)


    def load_images(self):
        image_list = []
        ds_dir = pathlib.Path(__file__).parents[1] / 'data/torch-dataset.pt'
        if os.path.exists(ds_dir):
            image_list = torch.load(ds_dir)
        else:
            print("Start loading images")
            for img in tqdm(self.img_dir_list):
                image = Image.open(img).convert('RGB')

                if self.fog_level:
                    image_numpy = np.array(image)
                    
                    foggy_image_list = self.foggify_image(image_numpy)
                    image_list.append(tuple([image] + foggy_image_list))
                else:
                    image_list.append((image))

            torch.save(image_list, ds_dir)
        self.image_list = image_list


    def foggify_image(self, image):
        image_list = []
        (row, col, chs) = image.shape
        A = 0.5  
        size = math.sqrt(max(row, col)) 
        center = (row // 2, col // 2)  
        for i in range(self.fog_level):
            beta = 0.01 * i
            hazy_img_numpy = self.add_haz(image, center, size, beta, A)
            image_list.append(Image.fromarray(hazy_img_numpy))
        return image_list


    # @jit
    def add_haz(self, img_f, center, size, beta, A):
        (row, col, chs) = img_f.shape

        x, y = np.meshgrid(np.linspace(0, row, row, dtype=int),
                           np.linspace(0, col, col, dtype=int))
        d = -1 / 255 * np.sqrt(x ** 2 + y ** 2) # + size
        d = np.tile(d, (3, 1, 1)).T
        trans = np.exp(-d * beta)

        A = 255
        hazy = img_f * trans + A * (1 - trans)
        hazy = np.array(hazy, dtype=np.uint8)

        return hazy

        # for j in range(row):
        #     for l in range(col):
        #         d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
        #         td = math.exp(-beta * d)
        #         img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
        # return img_f


    def gen_haze(self, img, depth_img):
        depth_img_3c = np.zeros_like(img)
        depth_img_3c[:,:,0] = depth_img
        depth_img_3c[:,:,1] = depth_img
        depth_img_3c[:,:,2] = depth_img

        beta = random.randint(100,150)/100
        norm_depth_img = depth_img_3c / 255
        trans = np.exp(-norm_depth_img * beta)

        A = 255
        hazy = img * trans + A * (1 - trans)
        hazy = np.array(hazy, dtype=np.uint8)

        return hazy


def show(sample):

    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()

def load_example_coco_detection_dataset(**kwargs):
    torchvision.disable_beta_transforms_warning()
    root = pathlib.Path("/home/soheil/data") / "coco"
    return datasets.CocoDetection(str(root / "images" / "train2014"),
                                  str(root / "annotations" / "instances_train2014.json"),
                                  **kwargs)


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def _create_data_loader(img_path, batch_size, img_size, n_cpu, transform,
                        multiscale_training=False, shuffle=True):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True,
                 transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)


    def load_annotations(self, annot_path):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        print('###################the total image:', len(annotations))
        return annotations

