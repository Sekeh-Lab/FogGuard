import os
import torch
# import cv2
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
# from util.parse_config import parse_data_config
from util.augmentations import AUGMENTATION_TRANSFORMS
from util.logger import Logger
import util.datasets as ds
from activation import Activations
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names

cur_time = time.strftime("%H-%M-%S",time.localtime(time.time()))


def train_student_one_epoch(s_model, t_model, loader, optimizer, device, epoch,
                            logger, batch_size, stio_logger):

    losses = utils.AverageMeter()
    lbox = utils.AverageMeter()
    lobj = utils.AverageMeter()
    lcls = utils.AverageMeter()

    loss = 0
    s_model.train()

    for train_epoch, (img_dir, images, depth, targets) in enumerate(loader):
        hazy_batch = (utils.get_hazy_tensor(images)).to(device)
        images = images.to(device)
        targets = targets.to(device)

        # Output of student
        s_outputs = s_model(images)
        # t_output = t_model(images)

        loss, loss_components = compute_loss(s_outputs, targets, s_model)

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

        logger_summary = [
            ("student/loss/total", loss.detach().item()),
            ("student/loss/box", lbox.avg), 
            ("student/loss/obj", lobj.avg), 
            ("student/loss/cls", lcls.avg), 
            ]

        logger.list_of_scalars_summary(logger_summary,
                                       epoch * len(loader) + train_epoch)

        # progress_bar.set_postfix(loss=losses.avg)

    return losses.avg

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
    for train_epoch, (img_dir, images, depth, targets) in enumerate(loader):
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

def train_student_perc_loss_one_epoch(s_model, t_model, loader, optimizer,
                                      device, epoch, logger, batch_size,
                                      stio_logger, lu_beta):

    # progress_bar = tqdm(loader, leave=True)
    t_activations = Activations(t_model, loader, device, batch_size)
    s_activations = Activations(s_model, loader, device, batch_size)
    layers_idx = t_activations.get_layers_idx()
    layers_dim = t_activations.layers_dim
    num_layers = len(layers_dim)
    # layers = list(range(5))                          # first l layers
    layers = list(range(num_layers - 5, num_layers))    # last l layers
    # act_keys = t_activations.get_act_keys()

    losses = utils.AverageMeter()  # loss
    lda = utils.AverageMeter()  # loss
    lbox = utils.AverageMeter()
    lobj = utils.AverageMeter()
    lcls = utils.AverageMeter()
    # outputs = []
    loss = 0
    lambda_perc = 0.001
    s_model.train()

    for train_epoch, (img_dir, images, depth, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        beta = torch.randint(lu_beta[0], lu_beta[1], (images.shape[0], )) / 100.
        beta = beta.to(device)
        depth = depth.to(device)
        # for synthetic fog
        # hazy_batch = (utils.add_guassi_haze_batch(images, beta)).to(device)

        # for realistic fog
        hazy_batch = utils.add_real_haze_batch(images, depth, beta)

        # Output of student
        s_outputs = s_model(hazy_batch)
        # t_output = t_model(images)

        # Calculating the loss function
        # diff_act = activations.get_batch_diff_activations(s_model, images,
        # hazy_batch)
        diff_act = torch.zeros(len(layers))
        clear_act = t_activations.get_activations(images, layers)
        hazy_act = s_activations.get_activations(hazy_batch, layers)
        # clear_act = t_activations.get_first_activations(images, l)
        # hazy_act = s_activations.get_first_activations(hazy_batch, l)
        # clear_act = t_activations.get_last_activations(images, l)
        # hazy_act = s_activations.get_last_activations(hazy_batch, l)

        is_voc = [True if file_dir.split('/')[4] == "VOC" else False
                  for file_dir in img_dir]

        for idx in range(len(clear_act)):
            # remove rtts from the perceptual loss
            no_rtts = torch.stack([torch.ones(clear_act[idx].shape[1:])
                                   if is_voc[i]
                                   else torch.zeros(clear_act[idx].shape[1:])
                                   for i in range(clear_act[idx].shape[0])]
                                  ).to(device)
            diff_act[idx] = torch.mean(torch.Tensor([torch.sqrt(
                                torch.norm(no_rtts * 
                                          (clear_act[idx] - hazy_act[idx])))
                                for i in range(1, clear_act[idx].shape[0])]))

            # diff_act[idx] = torch.mean(torch.Tensor([torch.sqrt(
            #                     torch.norm((clear_act[idx] - hazy_act[idx])))
            #                     for i in range(1, clear_act[idx].shape[0])]))

        loss, loss_components = compute_loss(s_outputs, targets, s_model)
        # sum_diff_act = torch.tensor(np.sum(diff_act) * beta).to(device)
        sum_diff_act = (torch.sum(diff_act) * lambda_perc).to(device)
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
        lda.update(sum_diff_act.detach().item())
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

        # progress_bar.set_postfix(loss=losses.avg)

    return losses.avg, lda.avg


def evaluate_model(model, valid_dl, args, device, class_names, logger, epoch,
                   on_fog=False, beta=[0, 16]):

    # Evaluate the model on the validation set
    if on_fog:
        metrics_output = evaluate_on_fog(model, valid_dl, args, device,
                                         class_names,
                                         model.hyperparams['height'],
                                         epoch, beta)
    else:
        import ipdb; ipdb.set_trace()

        metrics_output = _evaluate(model, valid_dl, device, class_names,
                                   img_size=model.hyperparams['height'],
                                   iou_thres=args.iou_thres,
                                   conf_thres=args.conf_thres,
                                   nms_thres=args.nms_thres,
                                   verbose=args.verbose, epoch=epoch)

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

def fine_tune(model, optimizer, train_dl, valid_dl, device, args, class_names,
              tb_logger):
    """ Fine tune the model to it's best performance.
    Train the model to the number of epochs specified, save them and return the
    best model and the best mAP.

    return: the directory of the best model
    """
    best_map, cur_map = 0., 0.
    if args.t_pretrained_weights:
        print("---- Fine Tuning Initial Evaluation ----")
        cur_map = evaluate_model(model, valid_dl, args, device,
                                 class_names, tb_logger, epoch=0,
                                 on_fog=False)
        print(f"Initial mAP: {cur_map:.4f}")

    if args.ft_epochs == 0:
        # return f"checkpoints/{len(class_names)}_teacher_{cur_map:.2f}_map.pth"
        return args.t_pretrained_weights, cur_map

    progress_bar = tqdm(range(1, args.ft_epochs + 1), leave=True)

    for epoch in progress_bar:
        progress_bar.write(f"Training Teacher {epoch}")
        loss = train_teacher_one_epoch(model, train_dl, optimizer, device,
                                       epoch, tb_logger)

        # Evaluate the model
        if epoch % args.evaluation_interval == 0:
            progress_bar.write(f"Evaluating model {epoch}")
            cur_map = evaluate_model(model, valid_dl, args, device,
                                     class_names, tb_logger, epoch,
                                     on_fog=False)

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0 and cur_map - best_map > 0.01:
            best_dir = utils.save_best(model, f"{len(class_names)}_teacher",
                                 best_map, cur_map) 
            best_map = cur_map

        progress_bar.set_postfix({"loss":loss, 
                                  "map":cur_map,
                                  })
    print(f"After fine tuning, best mAP: {best_map:.4f}")

    return best_dir, best_map

def train_student(s_model, t_model, s_optimizer, t_optimizer, train_dl,
                  valid_dl, batch_size, args, device, class_names,
                  tb_logger, stio_logger): 

    best_s_map, s_cur_map = 0., 0.
    diff_act = 0.0

    if args.epochs == 0:
        cur_map = evaluate_model(s_model, valid_dl, args, device,
                                 class_names, tb_logger, epoch=0,
                                 on_fog=False)
        print(f"Evaluated student mAP: {cur_map:.4f}")
        # return f"checkpoints/{len(class_names)}_student_{cur_map:.2f}_map.pth"
        return args.s_pretrained_weights, cur_map

    progress_bar = tqdm(range(1, args.epochs + 1), leave=True)

    for epoch in progress_bar:
        # train_dl, valid_dl = train_ds.create_dataloader()
        progress_bar.write(f"Training Student {epoch}")

        # train without perceptual loss
        if args.perc_loss == 0:
            loss = train_student_one_epoch(s_model, t_model, train_dl, s_optimizer,
                                           device, epoch, tb_logger, batch_size,
                                           stio_logger)

        # train with perceptual loss
        elif args.perc_loss == 1:
            loss, diff_act = train_student_perc_loss_one_epoch(s_model, t_model,
                                                            train_dl, s_optimizer,
                                                            device, epoch,
                                                            tb_logger, batch_size,
                                                            stio_logger,
                                                            lu_beta=[0, 16])

        # Evaluate
        if epoch % args.evaluation_interval == 0:
            progress_bar.write(f"Evaluating Student {epoch}")
            s_cur_map = evaluate_model(s_model, valid_dl, args, device,
                                       class_names, tb_logger, epoch,
                                       on_fog=True)
                                       # on_fog=False)

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0 and s_cur_map - best_s_map > 0.01:
            best_model_dir = utils.save_best(s_model, f"{len(class_names)}_student",
                                           best_s_map, s_cur_map)

            best_s_map = s_cur_map

        progress_bar.set_postfix({"loss":loss, 
                                  "d_act":diff_act,
                                  "s_map":s_cur_map,
                                  })

    print(f"best student mAP: {best_s_map:.4f}")

    return best_model_dir, best_s_map


def evaluate_various_fog(model, valid_dl, args, device, class_names, logger):
    maps = [0] * 3
    print("evaluate on [0, 5] fog")
    maps[0] = evaluate_model(model, valid_dl, args, device, class_names, logger,
                             epoch=0, on_fog=True, beta=[0, 5])
    print("evaluate on [5, 10] fog")
    maps[1] = evaluate_model(model, valid_dl, args, device, class_names, logger,
                             epoch=0, on_fog=True, beta=[5, 10])
    print("evaluate on [10, 16] fog")
    maps[2] = evaluate_model(model, valid_dl, args, device, class_names, logger,
                             epoch=0, on_fog=True, beta=[10, 16])
    return maps


def main():
    torch.manual_seed(10)
    args = utils.get_parsed_args()
    tb_logger = Logger(args.logdir)  # Tensorboard logger
    stio_logger = utils.setup_logger_dir(args)
    data_config = utils.parse_data_config(args.data)
    class_names = utils.load_classes(data_config["names"])
    device = utils.get_device(args)

    t_model = load_model(args.model, device, args.t_pretrained_weights)
    s_model = load_model(args.model, device, args.s_pretrained_weights)
    batch_size = t_model.hyperparams['batch']
    num_samples = 5000

    image_size = t_model.hyperparams['height']

    transform = utransforms.simple_transform(image_size)
    # transform = AUGMENTATION_TRANSFORMS
    # inv_transform =  transforms.inv_simple_transform(image_size, norm_mean, norm_std)

    # transform = utransforms.DEFAULT_TRANSFORMS
    train_ds = ds.yolo_dataset(data_config, "train", transform, image_size,
                               batch_size,
                               # num_samples,
                               )

    # train_dl = train_ds.create_dataloader()
    train_dl, valid_dl = train_ds.create_dataloader()

    test_ds = ds.yolo_dataset(data_config, "test", transform, image_size,
                              batch_size,
                              # num_samples=8,
                              )

    test_dl = test_ds.create_dataloader()

    # laod rtts to only validate at the end
    rtts_config = utils.parse_data_config('config/rtts.data')
    rtts_ds = ds.yolo_dataset(rtts_config, "test", transform, image_size,
                              batch_size)
    rtts_dl = rtts_ds.create_dataloader()

    voc_config = utils.parse_data_config('config/voc-5.data')
    voc_ds = ds.yolo_dataset(voc_config, "test", transform, image_size,
                             batch_size)
    voc_dl = voc_ds.create_dataloader()

    t_optimizer = utils.set_optimizer(t_model)
    s_optimizer = utils.set_optimizer(s_model)

    # fine tune the teacher to achieve a good performance on the dataset
    best_t_model, best_t_map = fine_tune(t_model, t_optimizer, train_dl,
                                         valid_dl, device, args, class_names,
                                         tb_logger)

    t_model = load_model(args.model, device, best_t_model)

    # train the student using the clear images
    best_s_model, best_s_map = train_student(s_model, t_model, s_optimizer,
                                             t_optimizer, train_dl, valid_dl,
                                             batch_size, args, device,
                                             class_names, tb_logger,
                                             stio_logger)

    s_model = load_model(args.model, device, best_s_model)

    # voc_t_map = evaluate_model(t_model, test_dl, args, device, class_names,
    #                             tb_logger, epoch=0)

    start_time = time.perf_counter()
    voc_s_map = evaluate_model(s_model, test_dl, args, device, class_names,
                               tb_logger, epoch=0)
    print(time.perf_counter() - start_time)


    # print(f"voc: teacher: {voc_t_map:.4f}, student: {voc_s_map:.4f}")


    # rtts_t_map = evaluate_model(t_model, rtts_dl, args, device, class_names,
    #                             tb_logger, epoch=0, on_fog=False,
    #                             ) 

    # rtts_s_map = evaluate_model(s_model, rtts_dl, args, device, class_names,
    #                             tb_logger, epoch=0, on_fog=False,
    #                             ) 

    # voc_5_t_map = evaluate_model(t_model, voc_dl, args, device, class_names,
    #                             tb_logger, epoch=0, on_fog=False,
    #                             ) 

    # voc_5_s_map = evaluate_model(s_model, voc_dl, args, device, class_names,
    #                             tb_logger, epoch=0, on_fog=False,
    #                             ) 

    # print(f"rtts: teacher: {rtts_t_map:.4f}, student: {rtts_s_map:.4f}")

    # t_map = evaluate_various_fog(t_model, test_dl, args, device, class_names,
    #                              tb_logger)

    # s_map = evaluate_various_fog(s_model, test_dl, args, device, class_names,
    #                              tb_logger)

    # print("teacher:", t_map)
    # print("student:", s_map)
    # data = {"Dataset": ["VOC", "test set", "RTTS", "beta [0, 0.5]", "beta [0.5, 0.10]",
    #                     "beta [0.10, 0.15]"],
    #         "Teacher": np.around([voc_5_t_map, best_t_map, rtts_t_map] + t_map, 4),
    #         "Student": np.around([voc_5_s_map, best_s_map, rtts_s_map] + s_map, 4)
    #         }
    # table = pd.DataFrame.from_dict(data)
    # print(table.to_string(index=False))
                
if __name__ == '__main__':
    main()
