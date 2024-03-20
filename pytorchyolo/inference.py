import os
from tqdm import tqdm, trange
import torch
import cv2
import numpy as np
from PIL import Image
import torch.optim as optim
from torchsummary import summary
from torch.autograd import Variable
from terminaltables import AsciiTable

import util.utils as utils
from util.utils import xywh2xyxy, ap_per_class, get_batch_statistics, non_max_suppression
import util.datasets as ds
import util.transforms as utransforms
from util.logger import Logger
from util.loss import compute_loss
from util.parse_config import parse_data_config, parse_model_config
from util.augmentations import AUGMENTATION_TRANSFORMS
from util.transforms import DEFAULT_TRANSFORMS
from model import load_model
import matplotlib.pyplot as plt
from activation import Activations


# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# # Image
# im = 'https://ultralytics.com/images/zidane.jpg'

# # Inference
# results = model(im)

# print(results.pandas().xyxy[0])

def _evaluate(model, dataloader, args, device, class_names, img_size, logger,
              iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
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

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for _, imgs, targets in tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        imgs = imgs.to(device)

        with torch.no_grad():
            import ipdb; ipdb.set_trace()
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres,
                                          iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets,
                                               iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    utils.print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output


def train(model, train_dataloader, optimizer, args, device, logger, epoch):
       # skip epoch zero, because then the calculations for when to
    # evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the
    # evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    print("\n---- Training Model ----")

    model.train()  # Set model to training mode

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
        batches_done = len(train_dataloader) * epoch + batch_i

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)

        outputs = model(imgs)

        loss, loss_components = compute_loss(outputs, targets, model)

        loss.backward()

        ###############
        # Run optimizer
        ###############

        if batches_done % model.hyperparams['subdivisions'] == 0:
            # Adapt learning rate
            # Get learning rate defined in cfg
            lr = model.hyperparams['learning_rate']
            if batches_done < model.hyperparams['burn_in']:
                # Burn in
                lr *= (batches_done / model.hyperparams['burn_in'])
            else:
                # Set and parse the learning rate to the steps defined in the cfg
                for threshold, value in model.hyperparams['lr_steps']:
                    if batches_done > threshold:
                        lr *= value
            # Log the learning rate
            logger.scalar_summary("train/learning_rate", lr, batches_done)
            # Set learning rate
            for g in optimizer.param_groups:
                g['lr'] = lr

            # Run optimizer
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()

        # ############
        # Log progress
        # ############
        if args.verbose:
            print(AsciiTable(
                [
                    ["Type", "Value"],
                    ["IoU loss", float(loss_components[0])],
                    ["Object loss", float(loss_components[1])],
                    ["Class loss", float(loss_components[2])],
                    ["Loss", float(loss_components[3])],
                    ["Batch loss", utils.to_cpu(loss).item()],
                ]).table)

        # Tensorboard logging
        tensorboard_log = [
            ("train/iou_loss", float(loss_components[0])),
            ("train/obj_loss", float(loss_components[1])),
            ("train/class_loss", float(loss_components[2])),
            ("train/loss", utils.to_cpu(loss).item())]
        logger.list_of_scalars_summary(tensorboard_log, batches_done)

        model.seen += imgs.size(0)

    # #############
    # Save progress
    # #############

    # Save model to checkpoint file
    if epoch % args.checkpoint_interval == 0:
        checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
        print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
        torch.save(model.state_dict(), checkpoint_path)


def main():
    args = utils.get_parsed_args()
    tb_logger = Logger(args.logdir)  # Tensorboard logger
    # stio_logger = utils.setup_logger_dir(args)
    data_config = parse_data_config(args.data)
    class_names = utils.load_classes(data_config["names"])
    device = utils.get_device(args)

    # t_model = load_model(args.model, device, args.pretrained_weights)
    t_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device,
                             classes=20)
    # s_model = load_model(args.model, device)
    batch_size = 16 # t_model.hyperparams['batch']
    num_samples = 100
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # t_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    image_size = 416 # t_model.hyperparams['height']
    transform = utransforms.simple_transform(image_size, norm_mean, norm_std)
    # transform = AUGMENTATION_TRANSFORMS
    # inv_transform =  transforms.inv_simple_transform(image_size, norm_mean, norm_std)

    # transform = utransforms.DEFAULT_TRANSFORMS
    # summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    train_ds = ds.voc_dataset(data_config, "train", transform, image_size,
                              batch_size,
                              # num_samples
                              )
    # train_ds = ds.VOC(data_config["train"], data_config["dest"], "train",
    #                   class_names, transform, image_size, batch_size,
    #                   num_samples=3*num_samples,
    #                   # copy_file=True
    #                   )
    train_dl = train_ds.create_dataloader()
    transform = utransforms.simple_transform(image_size, norm_mean, norm_std)
    # data_batch = next(iter(train_dl))
    # plt.imshow(data_batch[1][0].permute(1, 2, 0))
    # plt.savefig('image.png')

    valid_ds = ds.voc_dataset(data_config, "test", transform, image_size,
                              batch_size)
    valid_dl = valid_ds.create_dataloader()

    # optimizer = utils.set_optimizer(t_model)

    hyperparams = parse_model_config(args.model)[0]
    optimizer = optim.SGD(
        t_model.parameters(),
        lr=float(hyperparams['learning_rate']),
        weight_decay=float(hyperparams['decay']),
        momentum=float(hyperparams['momentum']))

    best_t_map, best_s_map = 0., 0.

    # print("---- Evaluating the teacher model ----")
    # t_cur_map = evaluate_model(t_model, valid_dl, args, device,
    #                             class_names, logger, epoch=0)

    for epoch in trange(args.epochs):
        # print("---- Training Teacher Model ----")
        # loss = train_teacher_one_epoch(t_model, train_dl, optimizer, device,
        #                                epoch, tb_logger)
        # stio_logger.debug("---- Training Student Model ----")
        # loss = train_student_one_epoch(s_model, t_model, train_dl, optimizer,
        #                                device, epoch, tb_logger, batch_size,
        #                                stio_logger)

        # Evaluate
        if epoch % args.evaluation_interval == 0:
            # stio_logger.debug("---- Evaluating the student model ----")
            # s_cur_map = evaluate_model(s_model, valid_dl, args, device,
            #                            class_names, tb_logger, epoch)
            print("---- Evaluating the teacher model ----")
            t_cur_map = _evaluate(t_model, valid_dl, args, device,
                                  class_names, image_size, tb_logger)
        # Save model to checkpoint file
        # if epoch % args.checkpoint_interval == 0:
        #     checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
        #     stio_logger.debug(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
        #     torch.save(s_model.state_dict(), checkpoint_path)
        #     # if best_t_map < t_cur_map:
        #     #     save_best(t_model, "teacher", best_t_map, t_cur_map)
        #     #     best_t_map = t_cur_map

        #     if best_s_map < s_cur_map:
        #         save_best(s_model, "student", best_s_map, s_cur_map)
        #         best_s_map = s_cur_map
        #     # save_best(s_model, best_map, s_cur_map)


if __name__ == '__main__':
    main()
