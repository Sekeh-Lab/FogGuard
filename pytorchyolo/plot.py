import os
import torch
import cv2
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from torchvision import datasets, models
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import PIL.Image
from torchinfo import summary
import torch.optim as optim
from model import load_model, Darknet
from util.loss import compute_loss
import util.utils as utils
import util.transforms as utransforms
from test import _evaluate, evaluate_on_fog
from util.parse_config import parse_data_config, parse_model_config
from util.augmentations import AUGMENTATION_TRANSFORMS
from util.logger import Logger
import util.datasets as ds
from activation import Activations
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from teacher_student import fine_tune

from util.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info

cur_time = time.strftime("%H-%M-%S",time.localtime(time.time()))



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

  #   target_boxes = target_boxes[target_boxes[:,0] == 0.].to("cpu")
  # # Plotting the bounding boxes and labels over the image
  #   for box in target_boxes:
  #       # Get the class from the box
  #       class_pred = box[1]
  #       box = box[2:]

  #       # Create a Rectangle patch with the bounding box
  #       rect = patches.Rectangle(
  #           box[0:2],
  #           box[2] - box[0],
  #           box[3] - box[1],
  #           linewidth=2,
  #           edgecolor=colors[int(class_pred)],
  #           facecolor="none",
  #       )

  #       # Add the patch to the Axes
  #       ax.add_patch(rect)

  #       # Add class name to the patch
  #       plt.text(
  #           box[0],
  #           box[1],
  #           s="target-"+class_labels[int(class_pred)],
  #           color="white",
  #           verticalalignment="top",
  #           bbox={"color": colors[int(class_pred)], "pad": 0},
  #       )

  #   plt.text(
  #           0,
  #           0,
  #           s=f"map: {map_metric:.3f}",
  #           color="white",
  #           verticalalignment="top",
  #           bbox={"color": "green", "pad": 0},
  #   )
  #   # Display the plot
  


def main():
    args = utils.get_parsed_args()
    tb_logger = Logger(args.logdir)  # Tensorboard logger
    stio_logger = utils.setup_logger_dir(args)
    data_config = parse_data_config(args.data)
    class_names = utils.load_classes(data_config["names"])
    device = utils.get_device(args)

    model = load_model(args.model, device, args.t_pretrained_weights)
    batch_size = 1 # t_model.hyperparams['batch']
    num_samples = 5000

    image_size = model.hyperparams['height']

    transform = utransforms.simple_transform(image_size)
    # transform = AUGMENTATION_TRANSFORMS
    # inv_transform =  transforms.inv_simple_transform(image_size, norm_mean, norm_std)

    # transform = utransforms.DEFAULT_TRANSFORMS
    # train_ds = ds.yolo_dataset(data_config, "train", transform, image_size,
    #                            batch_size,
    #                            # num_samples,
    #                           )

    # train_dl = train_ds.create_dataloader()
    # train_dl, valid_dl = train_ds.create_dataloader()

    test_ds = ds.yolo_dataset(data_config, "test", transform, image_size,
                              batch_size,
                              # num_samples=8,
                              )

    test_dl = test_ds.create_dataloader()

    # laod rtts to only validate at the end
    rtts_config = parse_data_config('../config/rtts.data')
    rtts_ds = ds.yolo_dataset(rtts_config, "test", transform, image_size,
                              batch_size)
    rtts_dl = rtts_ds.create_dataloader()

    voc_config = parse_data_config('../config/voc-5.data')
    voc_ds = ds.yolo_dataset(voc_config, "test", transform, image_size,
                              batch_size)
    voc_dl = voc_ds.create_dataloader()

    model.eval()  # Set model to evaluation mode

    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    for img_dir, imgs, depth, targets in test_dl:
        # Extract labels
        imgs = imgs.to(device)
        targets = targets.to(device)
        labels += targets[:, 1].tolist()
        beta = torch.ones(batch_size) * .16
        beta = beta.to(device)
        hazy_batch = utils.add_real_haze_batch(imgs, depth, beta)

        plt.imshow(hazy_batch[0].cpu().permute(1, 2, 0))

        image_dir = "/".join(img_dir[0].split('/')[-3:-1])
        image_name = img_dir[0].split('/')[-1]
        parent_dir = f'output/figures/{image_dir}' # Path(image_dir[0].parents[1])
        if not os.path.exists(parent_dir): os.makedirs(parent_dir)
        image_path = f'{parent_dir}/{image_name}'

        plt.savefig(image_path + '-hazy.png')

        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= image_size

        with torch.no_grad():
            outputs = model(imgs).to(device)
            outputs = non_max_suppression(outputs, conf_thres=args.conf_thres,
                                          iou_thres=args.nms_thres)

        plot_image(imgs[0].permute(1,2,0).to("cpu"), outputs[0], image_path,
                   class_names)

if __name__ == '__main__':
    main()
