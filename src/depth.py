import cv2
import torch
import math
import glob
import urllib.request
import os
from util.parse_config import parse_data_config
import argparse

# import util.transforms as utransforms
# from torchvision import utils
# import util.datasets as ds
from tqdm import tqdm
# import numpy as np
# from PIL import Image
# from os.path import expanduser
from pathlib import Path
# home = expanduser("~")
home = str(Path.home())

# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_config",
                    default="../config/voc-5.data",
                    help=".data file containing the config of dataset")

parser.add_argument("--data_type",
                    default="train", choices=["train", "test"])

parser.add_argument("--directory",
                    default="/home/soheil/data/VOC/train/VOCdevkit/VOC2012/images")

flags = parser.parse_args()

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


batch_size = 32
image_size = 416

data_config = parse_data_config(flags.data_config)
# simple_transform = utransforms.simple_transform(image_size)

# train_ds = ds.yolo_dataset(data_config, "train", simple_transform, image_size=image_size,
#                             batch_size=batch_size,
#                             # num_samples,
#                             )
# train_dl, valid_dl = train_ds.create_dataloader()

# for _, imgs, targets in train_dl:
#     imgs = transform(imgs).to(device)
#     with torch.no_grad():
#         prediction = midas(imgs)

# input_batch = transform(img).to(device)
# images_list_dir = data_config['test']
# images_list_dir = data_config['train']
images_list_dir = home + data_config[flags.data_type]
print("Image list file", images_list_dir)
with open(images_list_dir, 'r') as f:
    img_files = f.readlines()

# img_files = glob.glob(flags.directory + '/*.jpg')

psum, psum_sq, max_p, min_p = 0.0, 0.0, 0.0, 0.0

midas.to(device)
midas.eval()
for filename in tqdm(img_files):
    filename = filename.split()[0]
    # img = np.array(Image.open(filename).convert('RGB'), dtype=np.uint8)
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(image)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_parent = Path(Path(filename).parents[1], "depth")
    if not os.path.exists(depth_parent): os.makedirs(depth_parent)

    image_name = filename.split("/")[-1][:-4]
    # depth_fn = Path(depth_parent, image_name + ".pt")
    depth_fn = Path(depth_parent, image_name + ".png")

    # save the depth image
    # torch.save(prediction, depth_fn)
    cv2.imwrite(str(depth_fn), prediction.cpu().numpy()) # cv2.IMREAD_UNCHANGED

    # utils.save_image(prediction, depth_fn)
    # prediction = prediction.cpu().numpy() # .astype('uint16')
    psum += prediction.sum() / (prediction.shape[0] * prediction.shape[1])
    psum_sq += (prediction ** 2).sum() / (prediction.shape[0] * prediction.shape[1])

    max_p = max(max_p, prediction.max())
    min_p = min(min_p, prediction.min())


print(psum)
print(psum_sq)
count = len(img_files)
# mean and STD
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean ** 2)
total_std = math.sqrt(total_var)
print(total_mean)
print(total_std)
print(max_p)
print(min_p)

# output = prediction.cpu().numpy()

# plt.imshow(output)
# plt.show()
