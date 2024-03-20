import os
import torch
import cv2
import time
import numpy as np
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


def train_student_one_epoch(s_model, t_model, loader, optimizer, device, epoch,
                            logger, batch_size, stio_logger):

    progress_bar = tqdm(loader, leave=True)
    t_activations = Activations(t_model, loader, device, batch_size)
    s_activations = Activations(s_model, loader, device, batch_size)
    layers_idx = t_activations.get_layers_idx()
    layers_dim = t_activations.layers_dim
    num_layers = len(layers_dim)
    # act_keys = t_activations.get_act_keys()

    losses = utils.AverageMeter()  # loss
    lda = utils.AverageMeter()  # loss
    lbox = utils.AverageMeter()
    lobj = utils.AverageMeter()
    lcls = utils.AverageMeter()
    # outputs = []
    loss = 0
    beta = 0.005
    s_model.train()

    for train_epoch, (img_dir, images, targets) in enumerate(progress_bar):
        hazy_batch = utils.get_hazy_tensor(images)
        hazy_batch = hazy_batch.to(device)
        images = images.to(device)
        targets = targets.to(device)

        # Output of student
        s_outputs = s_model(images)
        t_output = t_model(images)

        # Calculating the loss function
        # diff_act = activations.get_batch_diff_activations(s_model, images,
        # hazy_batch)
        l = 5                   # last l layers
        diff_act = torch.zeros(l)
        clear_act = t_activations.get_activations(images, l)
        hazy_act = s_activations.get_activations(hazy_batch, l)
        for idx in range(len(clear_act)):
            diff_act[idx] = torch.mean(torch.Tensor([torch.sqrt(
                                torch.norm(clear_act[idx] - hazy_act[idx]))
                                for i in range(1, clear_act[idx].shape[0])]))

        loss, loss_components = compute_loss(s_outputs, targets, s_model)
        # sum_diff_act = torch.tensor(np.sum(diff_act) * beta).to(device)
        sum_diff_act = (torch.sum(diff_act) * beta).to(device)
        # stio_logger.debug(f"activation: {sum_diff_act.item():.3f}, other: {loss_components}")
        loss += sum_diff_act

        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.update(loss.detach().item())
        lda.update(sum_diff_act)
        lbox.update(loss_components[0])
        lobj.update(loss_components[1])
        lcls.update(loss_components[2])
        # outputs.append((epoch, image, reconstructed))
        logger_summary = [
            ("student/loss/total", loss.detach().item()),
            ("student/loss/diff_act", lda.avg), 
            ("student/loss/box", lbox.avg), 
            ("student/loss/obj", lobj.avg), 
            ("student/loss/cls", lcls.avg), 
            ]
        logger.list_of_scalars_summary(logger_summary,
                                       epoch * len(loader) + train_epoch)

        progress_bar.set_postfix(loss=losses.avg)

    return losses

def train_teacher_one_epoch(model, loader, optimizer, device, epoch, logger):
    # progress_bar = tqdm(loader, leave=True)
    model.train()  # Set model to training mode
    # outputs = []
    loss = 0.
    # losses = []
    losses = utils.AverageMeter()  # loss
    lbox = utils.AverageMeter()
    lobj = utils.AverageMeter()
    lcls = utils.AverageMeter()
    # for train_epoch, (img_dir, images, targets) in enumerate(progress_bar):
    for train_epoch, (img_dir, images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device)

        outputs = model(images)

        # Calculating the loss function
        loss, loss_components = compute_loss(outputs, targets, model)

        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.update(loss.detach().item())
        lbox.update(loss_components[0])
        lobj.update(loss_components[1])
        lcls.update(loss_components[2])

        # outputs.append((epoch, image, reconstructed))
        logger_summary = [
            # ("teacher/loss", loss.detach().item())
            ("teacher/loss/total", losses.avg), 
            ("teacher/loss/box", lbox.avg), 
            ("teacher/loss/obj", lobj.avg), 
            ("teacher/loss/cls", lcls.avg), 
            ]
        logger.list_of_scalars_summary(logger_summary,
                                       epoch * len(loader) + train_epoch)
        # progress_bar.set_postfix(loss=losses.avg)

    return losses.avg

def evaluate_model(model, valid_dl, args, device, class_names, logger, epoch):
    # Evaluate the model on the validation set
    metrics_output = _evaluate(
        model,
        valid_dl,
        device,
        class_names,
        img_size=model.hyperparams['height'],
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=args.verbose,
        epoch=epoch
    )

    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        evaluation_metrics = [
            # ("validation/precision", precision.mean()),
            # ("validation/recall", recall.mean()),
            ("validation/mAP", AP.mean()),
            # ("validation/f1", f1.mean())
        ]
        logger.list_of_scalars_summary(evaluation_metrics, epoch)

    return AP.mean()

def save_best(model, m_type, best_map, cur_map):
    print(f"current mAP: {cur_map:.3f}, best mAP: {best_map:.3f}")
    checkpoint_path = f"checkpoints/yolov3_ckpt_{m_type}_{cur_map:.2f}_map.pth"
    print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
    torch.save(model.state_dict(), checkpoint_path)
    return best_map


def main():
    args = utils.get_parsed_args()
    tb_logger = Logger(args.logdir)  # Tensorboard logger
    stio_logger = utils.setup_logger_dir(args)
    data_config = parse_data_config(args.data)
    class_names = utils.load_classes(data_config["names"])
    device = utils.get_device(args)

    t_model = load_model(args.model, device, args.t_pretrained_weights)
    s_model = load_model(args.model, device)
    batch_size = t_model.hyperparams['batch']
    num_samples = 100

    # t_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # t_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device, classes=10)  #
    
    image_size = t_model.hyperparams['height']
    transform = utransforms.simple_transform(image_size)
    # transform = AUGMENTATION_TRANSFORMS
    # inv_transform =  transforms.inv_simple_transform(image_size, norm_mean, norm_std)

    # transform = utransforms.DEFAULT_TRANSFORMS
    # summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))
    mini_batch_size = t_model.hyperparams['batch'] // t_model.hyperparams['subdivisions']

    train_ds = ds.yolo_dataset(data_config, "train", transform, image_size,
                              batch_size,
                              # num_samples,
                              shuffle=True
                              )
    # train_ds = ds.yolo(data_config["train"], data_config["dest"], "train",
    #                   class_names, transform, image_size, batch_size,
    #                   num_samples=3*num_samples,
    #                   # copy_file=True
    #                   )
    train_dl = train_ds.create_dataloader()

    valid_ds = ds.yolo_dataset(data_config, "test", transform, image_size,
                              batch_size,
                              # num_samples=1,
                              shuffle=False
                              )
    valid_dl = valid_ds.create_dataloader()

    optimizer = utils.set_optimizer(t_model)

    # hyperparams = parse_model_config(args.model)[0]
    # optimizer = optim.SGD(
    #     t_model.parameters(),
    #     lr=float(hyperparams['learning_rate']),
    #     weight_decay=float(hyperparams['decay']),
    #     momentum=float(hyperparams['momentum']))

    best_t_map, best_s_map = 0., 0.

    # print("---- Evaluating the teacher model ----")
    t_cur_map = evaluate_model(t_model, valid_dl, args, device,
                                class_names, tb_logger, epoch=0)

    # t_cur_map = evaluate_on_fog(t_model, valid_dl, args, device, class_names,
    #                             image_size, epoch=0)
    print("map on clear data", t_cur_map)

    progress_bar = tqdm(range(args.epochs), leave=True)
    for epoch in progress_bar:
        # print("---- Training Teacher Model ----")
        loss = train_teacher_one_epoch(t_model, train_dl, optimizer, device,
                                       epoch, tb_logger)
        # stio_logger.debug("---- Training Student Model ----")
        # loss = train_student_one_epoch(s_model, t_model, train_dl, optimizer,
        #                                device, epoch, tb_logger, batch_size,
        #                                stio_logger)

        # Evaluate
        if epoch % args.evaluation_interval == 0:
            # stio_logger.debug("---- Evaluating the student model ----")
            # s_cur_map = evaluate_model(s_model, valid_dl, args, device,
            #                            class_names, tb_logger, epoch)
            # print("---- Evaluating the teacher model ----")
            t_cur_map = evaluate_model(t_model, valid_dl, args, device,
                                        class_names, tb_logger, epoch)

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            if t_cur_map - best_t_map > 0.01:
                save_best(t_model, f"{len(class_names)}_teacher", best_t_map, t_cur_map)
                best_t_map = t_cur_map

            # if s_cur_map - best_s_map > 0.01:
            #     save_best(s_model, f"{len(class_names)}_student", best_s_map, s_cur_map)
            #     best_s_map = s_cur_map

        progress_bar.set_postfix({"loss":loss, "map":t_cur_map})


    # load the best model
    checkpoint_path = f"checkpoints/yolov3_ckpt_{len(class_names)}_teacher_{best_t_map:.2f}_map.pth"
    t_model = load_model(args.model, device, checkpoint_path)

    # evaluate it on the foggy dataset
    t_cur_map = evaluate_on_fog(t_model, valid_dl, args, device, class_names,
                                image_size, epoch=len(valid_dl))

    print("teacher mAP on foggy VOC:", best_t_map)
            
if __name__ == '__main__':
    main()
