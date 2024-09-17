# import torch.nn.functional as F
import torchvision.transforms.functional as F
import torch
import glob
import shutil
import math
import random
import os
from tqdm import tqdm, trange
import xml.etree.ElementTree as ET
import pathlib
from pathlib import Path
import pandas as pd
import warnings
import numpy as np
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, WeightedRandomSampler, SequentialSampler
# import torch.utils.data
import torchvision
import PIL.Image
from PIL import Image
from PIL import ImageFile
# import transforms
import util.transforms as utransforms
from torchvision import transforms
# import utils as utils
import util.utils as utils
from util.logger import Logger
# from logger import Logger

ImageFile.LOAD_TRUNCATED_IMAGES = True

class yolo_dataset(Dataset):
    def __init__(self, data_config, data_type, transform=None, image_size=416,
                 batch_size=16, num_samples=None):
        self.classes = utils.load_classes(data_config["names"])
        self.data_type = data_type
        self.image_dir_path = data_config[data_type]
        self.transform = transform
        self.img_size = image_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.multiscale = False
        self.batch_count = 0
        self.name = data_config["eval"]
        with open(self.image_dir_path, 'r') as f:
            self.img_files = f.readlines()

        self.create_folders()
        if copy_file:
            self.create_files()

    def create_folders(self):
        image_dst = os.path.join(self.data_dst, self.data_type, 'images')
        label_dst = os.path.join(self.data_dst, self.data_type, 'labels')
        self.make_folders(image_dst)
        self.make_folders(label_dst)
        
    def make_folders(self, path):
        """Create directory if not exists"""
        if not os.path.exists(path):
            # shutil.rmtree(path)
            os.makedirs(path)

    def replace_folders(self, path):
        """Replace the directory if exists"""
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def __len__(self):
        return len(self.image_inds)

    def create_files(self, use_difficult_bbox=False):
        image_list = ""
        for image_ind in tqdm(self.image_inds):
            image_path = os.path.join(self.data_src, image_ind[0], 'JPEGImages',
                                      image_ind[1] + '.jpg')
            # save the file in new diretory
            image_dst = os.path.join(self.data_dst, self.data_type, 'images',
                                    image_ind[1] + '.jpg')
            image_list += image_dst + "\n"
            shutil.copyfile(image_path, image_dst)
            annotation = image_path
            anno_path = os.path.join(self.data_dst, self.data_type, 'labels',
                                     image_ind[1] + '.txt')
            label_path = os.path.join(self.data_src, image_ind[0], 'Annotations',
                                      image_ind[1] + '.xml')
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
                if obj.find('name').text.lower().strip() in self.classes:
                    class_ind = self.classes.index(obj.find('name').text.lower().strip())
                    xmin = int(bbox.find('xmin').text.strip())
                    xmax = int(bbox.find('xmax').text.strip())
                    ymin = int(bbox.find('ymin').text.strip())
                    ymax = int(bbox.find('ymax').text.strip())
                    x, y, w, h = self.xml_to_yolo_bbox([xmin, ymin, xmax, ymax],
                                                    width, height)
                    # annotation += ' ' + ','.join([xmin, ymin, xmax, ymax,
                    # str(class_ind)])
                    yolo_annot += f"{class_ind} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                    # f.write(f"{class_ind} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            # print(yolo_annot)
            open(anno_path, 'w').write(yolo_annot)
        open(os.path.join(self.data_dst, self.data_type, "list"), 'w').write(image_list)

    def create_dataloader(self):
        dataset = ListDataset(
            list_path=os.path.join(self.data_dst, self.data_type, "list"),
            img_size=self.img_size,
            multiscale=False,
            transform=self.transform,
            num_samples=self.num_samples
        )

        # sampler = BatchSampler(RandomSampler(dataset, replacement=True),
        #                          self.batch_size, drop_last=True)
        sampler = RandomSampler(dataset, replacement=False,
                                num_samples=self.num_samples)

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            # shuffle=True,
            # num_workers=n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=utils.worker_seed_set
        )

        return dataloader

    def xml_to_yolo_bbox(self, bbox, w, h):
        # xmin, ymin, xmax, ymax
        x_center = ((bbox[2] + bbox[0]) / 2) / w
        y_center = ((bbox[3] + bbox[1]) / 2) / h
        width = (bbox[2] - bbox[0]) / w
        height = (bbox[3] - bbox[1]) / h
        return [x_center, y_center, width, height]


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None,
                 path=None, fog_level=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_dir_list = [os.path.join(img_dir, item.name)
                             for item in pathlib.Path(img_dir).glob("*.jpg")
                             if not item.is_dir()]
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
        ds_dir = pathlib.Path(__file__).parents[2] / 'data/torch-dataset.pt'
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
        A = 0.5  
        for i in range(self.fog_level):
            beta = 0.01 * i
            hazy_img_numpy = self.add_haz(image, beta)
            image_list.append(Image.fromarray(hazy_img_numpy))
        return image_list


    # @jit
    def add_haz(self, img_f, beta):
        img_f = img_f / 255
        (row, col, chs) = img_f.shape
        center = (row // 2, col // 2)  
        size = math.sqrt(max(row, col)) 

        x, y = np.meshgrid(np.linspace(0, row, row, dtype=int),
                        np.linspace(0, col, col, dtype=int))
        d = -0.04 * np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) + size
        d = np.tile(d, (3, 1, 1)).T
        trans = np.exp(-d * beta)

        # A = 255
        A = 0.5
        hazy = img_f * trans + A * (1 - trans)
        # hazy = np.array(hazy, dtype=np.uint8)

        return hazy

    # def add_haz_0(img_f, beta):
    #     (row, col, chs) = img_f.shape
    #     center = (row // 2, col // 2)  
    #     size = math.sqrt(max(row, col)) 
    #     A = 0.5  

    #     for j in range(row):
    #         for l in range(col):
    #             d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
    #             td = math.exp(-beta * d)
    #             img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    #     return img_f


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
                 transform=None, num_samples=None):
        with open(list_path, "r") as file:
            # if num_samples:
            #     self.img_files = file.readlines()[:num_samples]
            # else:
            self.img_files = file.readlines()
        if num_samples is None:
            # self.img_files = self.img_files[:num_samples]
            self.num_samples = len(self.img_files)

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = f"labels-{len(self.classes)}".join(image_dir.rsplit("JPEGImages", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'JPEGImages'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

    def __getitem__(self, index):

        #  Image
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

            # load depth channel
            depth_parent = Path(Path(img_path).parents[1], "depth")
            image_name = img_path.split("/")[-1][:-4]
            depth_fn = Path(depth_parent, image_name + '.pt')
            # depth = np.load(depth_fn)
            # Image.fromarray(np.array(Image.open(depth_fn)).astype('uint16'))
            depth = torch.load(depth_fn)

        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        #  Label
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)

        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        #  Transform
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        # return img_path, img, bb_targets
        return img_path, img, depth, bb_targets

    def collate_fn(self, batch, eval_flag=False):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        # paths, imgs, bb_targets = list(zip(*batch))
        paths, imgs, depth, bb_targets = list(zip(*batch))

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

        # return paths, imgs, bb_targets
        depth = torch.stack([self.transform_depth(img) for img in depth])
        return paths, imgs, depth, bb_targets


    def __len__(self):
        return len(self.img_files)
      
    def make_folders(self, path):
        """Create directory if not exists"""
        if not os.path.exists(path):
            # shutil.rmtree(path)
            os.makedirs(path)

    def replace_folders(self, path):
        """Replace the directory if exists"""
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def create_dataloader(self):
        # sampler = BatchSampler(RandomSampler(dataset, replacement=True),
        #                          self.batch_size, drop_last=True)
        if self.data_type == "train":
            train_ratio = 0.8
            train_size = int(train_ratio * len(self))
            valid_size = len(self) - train_size
            train_ds, valid_ds = torch.utils.data.random_split(self,
                                                            [train_size, valid_size])

            if self.name == 'voc':
                train_sampler = RandomSampler(train_ds, replacement=True, 
                                            num_samples=int(train_ratio *\
                                                            self.num_samples))

                valid_sampler = RandomSampler(valid_ds, replacement=True,
                                            num_samples=int((1 - train_ratio) *\
                                                            self.num_samples))

            elif self.name == 'voc-rtts':
                train_sampler = RandomSampler(train_ds, replacement=True,
                                            num_samples=int((1 - train_ratio) *\
                                                            self.num_samples))
                                        # num_samples=self.num_samples)
                valid_sampler = RandomSampler(valid_ds, replacement=True,
                                            num_samples=int((1 - train_ratio) *\
                                                            self.num_samples))
                                            # num_samples=self.num_samples)
                # if self.data_type == "test":
                #     rtts_frag = 432.0 / 5385.0
                # else:
                #     rtts_frag = 3890.0 / 20442.0

                # rtts_rate = .05
                # weights = [rtts_rate] * int(rtts_frag * len(self)) +\
                #             [1 - rtts_rate] * int((1 - rtts_frag) * len(self))
                # sampler = WeightedRandomSampler(weights, self.num_samples)

            else:
                sampler = SequentialSampler(self)

            train_dataloader = DataLoader(
                train_ds,
                sampler=train_sampler,
                batch_size=self.batch_size,
                # shuffle=self.shuffle,
                # num_workers=n_cpu,
                pin_memory=False,
                collate_fn=self.collate_fn,
                worker_init_fn=utils.worker_seed_set
            )
            val_dataloader = DataLoader(
                valid_ds,
                sampler=valid_sampler,
                batch_size=self.batch_size,
                pin_memory=False,
                collate_fn=self.collate_fn,
                worker_init_fn=utils.worker_seed_set
            )
            # else:
            #     dataloader = DataLoader(
            #         self,
            #         batch_size=self.batch_size,
            #         shuffle=False,
            #         # num_workers=n_cpu,
            #         pin_memory=True,
            #         collate_fn=self.collate_fn,
            #         worker_init_fn=utils.worker_seed_set
            #     )

            return train_dataloader, val_dataloader



        elif self.data_type == "test":
            test_sampler = RandomSampler(self, replacement=False)
            test_dataloader = DataLoader(
                self,
                sampler=test_sampler,
                batch_size=self.batch_size,
                pin_memory=False,
                collate_fn=self.collate_fn,
                worker_init_fn=utils.worker_seed_set
            )
            return test_dataloader

        else:
            exit("Wrong dataset type")
 
    def transform_depth(self, img):
        d_transform=transforms.Compose([
            SquarePadSize(self.img_size),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # img_tensor = F.interpolate(img_tensor.unsqueeze(0),
        #                            size=(self.img_size,self.img_size),
        #                            mode="nearest").squeeze(0)
        img_trs = d_transform(img)
        return img_trs
 
class SquarePadSize:
    def __init__(self, img_size):
        self.img_size = img_size
        "docstring"
        
    def __call__(self, image):
        # print('before: ', image.min(), image.max())
        s = image.shape
        max_wh = np.max([s[-1], s[-2]])
        hp = int((max_wh - s[-1]) / 2)
        vp = int((max_wh - s[-2]) / 2)
        padding = (hp, vp, hp, vp)
        image = F.pad(image, padding, fill=0, padding_mode='constant')
        # padded = torch.nn.functional.pad(image, padding, 0, mode='constant')
        image = torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0),
                                            size=(self.img_size,self.img_size),
                                            mode="bicubic").squeeze().squeeze()
        image = torch.clamp(image, 0.)
        # print("after:", image.min(), image.max())
        return image
        # max_wh = max(image.size)
        # p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        # p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        # padding = (p_left, p_top, p_right, p_bottom)
        # return F.pad(image, padding, 0, 'constant')

def pad_to_square(img, pad_value=0):
    # c, h, w = img.shape
    h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def _create_data_loader(img_path, batch_size, img_size, n_cpu, transform,
                        num_samples=None, multiscale_training=False,
                        shuffle=True):
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
        transform=transform,
        num_samples=num_samples
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=utils.worker_seed_set)
    return dataloader


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




def main():
    args = utils.get_parsed_args()
    logger = Logger(args.logdir)  # Tensorboard logger
    data_config = utils.parse_data_config(args.data)
    class_names = utils.load_classes(data_config["names"])
    device = utils.get_device(args)

    batch_size = 4
    image_size = 256
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Transforms images to a PyTorch Tensor
    transform = transforms.simple_transform(image_size, norm_mean, norm_std)

    # voc train dataset
    train_ds = VOC(data_config["train"], data_config["dest"], "train",
                   class_names, transform, image_size, batch_size).create_dataloader()
    test_ds = VOC(data_config["test"], data_config["dest"], "test",
                  class_names, transform, image_size, batch_size).create_dataloader()


if __name__ == '__main__':
    main()
