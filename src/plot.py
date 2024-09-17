import os
import torch
import cv2
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import util.utils as utils
import util.transforms as utransforms
import util.augmentations as aug
from util.logger import Logger
import util.datasets as ds
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes
import constants as C
import imgaug.augmenters as iaa

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs, filename):
    if not isinstance(imgs, list):
        imgs = [imgs]

    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    img_path = C.OUTPUT_DIR + filename + '.png'
    print(img_path)
    plt.savefig(img_path)


def plot_image(image, pred_boxes, image_dir, class_labels):
    # Getting the color map from matplotlib
    colour_map = plt.get_cmap("tab20b")
    # Getting 20 different colors from the color map for 20 different classes
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))]

    # Reading the image with OpenCV
    img = np.array(image)
    # Getting the height and width of the image
    h, w, _ = img.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Add image to plot
    ax.imshow(img)

    # Plotting the bounding boxes and labels over the image
    for box in pred_boxes:
        # Get the class from the box
        class_pred = box[5]
        # Get the center x and y coordinates
        box = box[:4]
        # Get the upper left corner coordinates
        # upper_left_x = box[0] - box[2] / 2
        # upper_left_y = box[1] - box[3] / 2

        # Create a Rectangle patch with the bounding box
        rect = patches.Rectangle(
            # (upper_left_x * w, upper_left_y * h),
            # box[2] * w,
            # box[3] * h,
            box[0:2],
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add class name to the patch
        plt.text(
            # upper_left_x * w,
            # upper_left_y * h,
            box[0],
            box[1],
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    img_path = image_dir + '-detected.png'
    print(img_path)
    plt.savefig(img_path)


def plot_batch(dataloader):
    img_dir, images, depth, targets = next(iter(dataloader))
    grid = make_grid(images)
    show(grid, "orig")

    images = images.permute([0, 2, 3, 1]) * 255
    images = images.numpy().astype(np.uint8)

    # rain
    rain = iaa.Sequential([
        iaa.Rain(speed=(0.1, 0.3)),
    ])
    rainy = torch.from_numpy(rain(images=images)).permute([0, 3, 1, 2])
    grid = make_grid(rainy)
    show(grid, "rainy")

    # snow
    snow = iaa.Sequential([
        iaa.Snowflakes(),
    ])
    snowy = torch.from_numpy(snow(images=images)).permute([0, 3, 1, 2])
    grid = make_grid(snowy)
    show(grid, "snowy")

    # fog
    fog = iaa.Sequential([
        iaa.Fog(),
    ])

    foggy = torch.from_numpy(fog(images=images)).permute([0, 3, 1, 2])
    grid = make_grid(foggy)
    show(grid, "foggy")


def main():
    args = utils.get_parsed_args()
    data_config = utils.parse_data_config(args.data)
    class_names = utils.load_classes(data_config["names"])
    device = utils.get_device(args)

    image_size = 416
    transform = utransforms.simple_transform(image_size)
    # transform = aug.WeatherAug(image_size)
    train_ds = ds.yolo_dataset(data_config, "train", transform, image_size,
                               batch_size=4,
                               )
    train_dl, valid_dl = train_ds.create_dataloader()

    plot_batch(train_dl)
    # transform = AUGMENTATION_TRANSFORMS
    # inv_transform =  transforms.inv_simple_transform(image_size, norm_mean, norm_std)

    # transform = utransforms.DEFAULT_TRANSFORMS
    # plot_image(imgs[0].permute(1,2,0).to("cpu"), outputs[0], image_path,
    #             class_names)


if __name__ == '__main__':
    main()
