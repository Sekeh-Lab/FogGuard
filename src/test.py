#! /usr/bin/env python3

from __future__ import division

import argparse
from tqdm import tqdm, trange
import numpy as np

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import util.utils as utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import load_model
from util.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info
# from util.datasets import ListDataset
from util.transforms import DEFAULT_TRANSFORMS
from util.parse_config import parse_data_config


def evaluate_model_file(model_path, weights_path, img_path, class_names, batch_size=8, img_size=416,
                        n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    dataloader = _create_validation_data_loader(
        img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    metrics_output = _evaluate(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


def _evaluate(model, dataloader, device, class_names, img_size, iou_thres,
              conf_thres, nms_thres, verbose, epoch):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for _, imgs, depth, targets in dataloader:
        # Extract labels
        imgs = imgs.to(device)
        targets = targets.to(device)
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        # imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs).to(device)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres,
                                          iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None


    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels,
                                  labels)

    # print_eval_stats(metrics_output, class_names, verbose)

    # plot_image(imgs[0].permute(1,2,0).to("cpu"), outputs[0], targets, epoch,
    #            class_names, metrics_output[2].mean())

    return metrics_output


def evaluate_on_fog(model, dataloader, args, device, class_names, img_size,
                    epoch, lu_beta=[0, 16]):

    iou_thres=args.iou_thres
    conf_thres=args.conf_thres
    nms_thres=args.nms_thres
    verbose=args.verbose

    model.eval()  # Set model to evaluation mode

    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for _, imgs, depth, targets in dataloader:
        # Extract labels
        # hazy_batch = (utils.get_hazy_tensor(imgs)).to(device)

        imgs = imgs.to(device)
        beta = torch.randint(lu_beta[0], lu_beta[1], (imgs.shape[0], )) / 100.
        beta = beta.to(device)
        depth = depth.to(device)
        imgs = utils.add_real_haze_batch(imgs, depth, beta)

        targets = targets.to(device)
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        # imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs).to(device)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres,
                                          iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None


    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels,
                                  labels)

    # print_eval_stats(metrics_output, class_names, verbose)

    # plot_image(hazy_batch[0].permute(1,2,0).to("cpu"), outputs[0], targets, epoch,
    #            class_names, metrics_output[2].mean())

    # return metrics_output[2].mean()
    return metrics_output



def plot_image(image, pred_boxes, target_boxes, epoch, class_labels, map_metric):
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

    target_boxes = target_boxes[target_boxes[:,0] == 0.].to("cpu")
  # Plotting the bounding boxes and labels over the image
    for box in target_boxes:
        # Get the class from the box
        class_pred = box[1]
        box = box[2:]

        # Create a Rectangle patch with the bounding box
        rect = patches.Rectangle(
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
            box[0],
            box[1],
            s="target-"+class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.text(
            0,
            0,
            s=f"map: {map_metric:.3f}",
            color="white",
            verticalalignment="top",
            bbox={"color": "green", "pad": 0},
    )
    # Display the plot
    plt.savefig(f'output/figures/{epoch}.png')



def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Load configuration from data file
    data_config = parse_data_config(args.data)
    # Path to file containing all images for validation
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])  # List of class names

    precision, recall, AP, f1, ap_class = evaluate_model_file(
        args.model,
        args.weights,
        valid_path,
        class_names,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=True)


if __name__ == "__main__":
    run()
