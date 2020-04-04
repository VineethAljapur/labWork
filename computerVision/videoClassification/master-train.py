#!/usr/bin/env python3
# coding: utf-8

import glob
import os
import sys
import skvideo
from skvideo import io as vp
from time import time
import cProfile
import pstats

import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
import logging

import random
import math
import numbers
import collections

from PIL import Image, ImageOps

try:
    import accimage
except ImportError:
    accimage = None

import torch
from torch import nn
import torchvision

import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

import cv2

from spatial_transformations.Compose import Compose
from spatial_transformations.ToTensor import ToTensor
from spatial_transformations.Scale import Scale
from spatial_transformations.RandomHorizontalFlip import RandomHorizontalFlip
from spatial_transformations.MultiScaleRandomCrop import MultiScaleRandomCrop

from CichlidsDataLoader import Cichlids

from utils.logger import setup_logger

if __name__ == "__main__":
    ############################################################################
    # PARSING ARGUMENTS                                                        #
    # TODO: to parse the arguments regarding model                             #
    # Expects the following arguments:                                         #
    #   model (str, default = r3d)                                             #
    #   pre_trained (bool, default = True)                                     #
    #   fine_tune (bool, default = True)                                       #
    #   num_classes (int, default = 10)                                        #
    #   epochs (int, default = 50)                                             #
    #   learning_rate (float, default = 0.001)                                 #
    #   momentum (float, default = 0.9)                                        #
    #   weight_decay (float, default = 1e-3)                                   #
    #   epochs (int, default = 50)                                             #
    #   save_interval (int, default = 5)                                       #
    #   results (str, default = 'results')                                     #
    ############################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", default="r3d", type=str, help="(r3d | mc3 | r2plus1d)"
    )

    parser.add_argument(
        "--pre_trained",
        default=True,
        type=bool,
        help="To use the pre_trained weights for model initialization",
    )

    parser.add_argument(
        "--fine_tune",
        default=True,
        type=bool,
        help="To fine tune the pretrained weights",
    )

    parser.add_argument(
        "--num_classes", default=2, type=int, help="number of video categories"
    )

    parser.add_argument(
        "--learning_rate",
        default=0.00005,
        type=float,
        help="Learning rate of the model optimizer",
    )

    parser.add_argument(
        "--momentum", default=0.5, type=float, help="momentum of the model optimizer"
    )

    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="weight decay of the model optimizer",
    )

    parser.add_argument(
        "--epochs", default=30, type=int, help="Number of epoch to train the model"
    )

    parser.add_argument(
        "--save_interval", default=1, type=int, help="epoch interval to save the model"
    )

    parser.add_argument(
        "--batch_size",
        default=3,
        type=int,
        help="batch size for training and testing dataloader",
    )

    parser.add_argument(
        "--num_workers", default=6, type=int, help="number of threads to use"
    )

    parser.add_argument(
        "--data", default="data/", type=str, help="path of the data directory"
    )

    parser.add_argument(
        "--initial_scale",
        default=1.0,
        type=float,
        help="Initial scale for multiscale cropping",
    )

    parser.add_argument(
        "--n_scales",
        default=5,
        type=int,
        help="Number of scales for multiscale cropping",
    )

    parser.add_argument(
        "--scale_step",
        default=0.84089641525,
        type=float,
        help="Scale step for multiscale cropping",
    )

    parser.add_argument(
        "--results", default="results", type=str, help="path of the results directory"
    )

    parser.add_argument(
        "--saved_checkpoint", default=None, type=str, help="path of saved model"
    )

    parser.add_argument(
        "--resize",
        nargs="?",
        const="(224, 224)",
        type=str,
        help="to confirm the resize",
    )

    parser.add_argument(
        "--debug", default=False, type=bool, help="To set logging to debug"
    )

    args = parser.parse_args()

    logger = setup_logger("training.log", debug=args.debug)

    logger.info(args)

    writer = SummaryWriter()

    ############################################################################
    #                             END OF PARSING ARGUMENTS                     #
    ############################################################################

    ############################################################################
    # MODEL DEFINATION                                                         #
    # TODO: to deine the model and modify according to our data                #
    # Currently, TorchVision only has 3 architectures for video classification #
    # Change the last fully connected layer according number of classes        #
    # Use stochastic gradient descent optimizer according to hyperparameters   #
    ############################################################################
    try:
        assert args.model in ["r3d", "mc3", "r2plus1d"]
    except AssertionError as err:
        logger.exception("Model not in ['r3d', 'mc3', 'r2plus1d']")
        raise err

    if args.model == "r3d":
        model = torchvision.models.video.r3d_18(
            pretrained=args.pre_trained, progress=False
        )

    if args.model == "mc3":
        model = torchvision.models.video.mc3_18(
            pretrained=args.pre_trained, progress=False
        )

    if args.model == "r2plus1d":
        model = torchvision.models.video.r2plus1d_18(
            pretrained=args.pre_trained, progress=False
        )

    # Modifing the last layer according to our data
    model.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True)

    # To check if we are doing fine-tuning
    if not args.fine_tune:
        for name, param in model.named_parameters():
            param.requires_grad = False

    # To parallalize the model. By default it uses all available gpu.
    # Set visible devices using CUDA_VISIBLE_DEVICE
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    # Optimizer for the model
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.saved_checkpoint is not None:
        checkpoint = torch.load(args.saved_checkpoint)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        logger.debug({"Checkpoint: ": args.saved_checkpoint, "Epoch: ": epoch})

    # Creating results folder if not already exists
    results = args.results
    if not os.path.isdir(results):
        os.mkdir(results)
    ############################################################################
    #                             END OF MODEL DEFINATION                      #
    ############################################################################

    #######################################################################################
    # DATA TRANSFORMATIONS                                                                #
    # TODO: Implement data augmentation and transformations                               #
    # These classes are taken from Kensho Hara, et als implimentaion                      #
    # https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/spatial_transforms.py  #
    #######################################################################################

    scales = [args.initial_scale]
    for i in range(1, args.n_scales):
        scales.append(scales[-1] * args.scale_step)

    logger.debug(("Scales: ", scales))

    # Defining spatial transformations
    spatial_transform = Compose(
        [MultiScaleRandomCrop(scales, 112), RandomHorizontalFlip(), ToTensor()], logger
    )

    # Load the trainset
    trainset = Cichlids(
        root=args.data + "training",
        logger=logger,
        resize=args.resize,
        preload=False,
        spatial_transform=spatial_transform,
    )

    trainset_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # Load the testset
    testset = Cichlids(
        root=args.data + "testing",
        logger=logger,
        resize=args.resize,
        preload=False,
        spatial_transform=spatial_transform,
    )

    testset_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    ############################################################################
    #                             END OF DATA TRANSFORMATIONS                  #
    ############################################################################

    ############################################################################
    # TRAINING AND TESTING DEFINITIONS                                         #
    # TODO: Define train and test function along with saving models and        #
    # creating confusion_matrix                                                #
    ############################################################################

    def train_model(epochs=args.epochs, log_interval=100):
        '''model training defination'''
        model.train()
        for epoch in range(epochs):
            start = time()
            iteration = 0
            avg_loss = 0
            correct = 0
            for batch_idx, (data, target) in enumerate(trainset_loader):

                if torch.cuda.is_available():
                    target = target.cuda(
                        non_blocking=True
                    )  # async=True has deprecated in latest pytorch

                data = Variable(data)
                target = Variable(target)

                output = model(data)  # Add Log here

                lossFunction = nn.CrossEntropyLoss()
                if torch.cuda.is_available():
                    lossFunction = lossFunction.cuda()

                loss = lossFunction(output, target)

                avg_loss += loss

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if iteration % log_interval == 0:
                    logger.info(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]".format(
                            epoch,
                            batch_idx * len(data),
                            len(trainset_loader.dataset),
                            100.0 * batch_idx / len(trainset_loader),
                        )
                    )
                iteration += 1

                pred = output.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability

                correct += pred.eq(target.view_as(pred)).sum().item()

                logger.debug(
                    {
                        "iteration": iteration,
                        "output": output,
                        "target": target,
                        "loss": loss,
                        "pred": pred,
                        "correct": correct,
                    }
                )

            end = time()
            logger.info("\nSummary: Epoch {}".format(epoch))  # f stings
            logger.info("Time taken for this epoch: {:.2f}s".format(end - start))
            logger.info(
                "Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                    avg_loss / len(trainset_loader.dataset),
                    correct,
                    len(trainset_loader.dataset),
                    100.0 * correct / len(trainset_loader.dataset),
                )
            )
            writer.add_scalar('Loss/train', avg_loss / len(trainset_loader.dataset), epoch)
            writer.add_scalar('Accuracy/train', correct / len(trainset_loader.dataset), epoch)
            # cap log
            if epoch % args.save_interval == 0:
                save_file_path = os.path.join(results, "save_{}.pth".format(epoch))
                states = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
            check_accuracy(epoch)  # evaluate at the end of epoch
        torch.cuda.empty_cache()  # Clear cache after training

    def check_accuracy(epoch):
        '''To check accuracy corresponding to epoch'''
        num_correct = 0
        num_samples = 0
        test_loss = 0
        correct = 0

        confusion_matrix = np.zeros((args.num_classes, args.num_classes))
        confidence_for_each_validation = {}

        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for data, target in testset_loader:
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)

                data = Variable(data)
                target = Variable(target)

                output = model(data)

                lossFunction = nn.CrossEntropyLoss()
                if torch.cuda.is_available():
                    lossFunction = lossFunction.cuda()

                rows = [int(x) for x in target]
                columns = [int(x) for x in np.argmax(output.cpu(), 1)]
                assert len(rows) == len(columns)
                for idx in range(len(rows)):
                    confusion_matrix[rows[idx]][columns[idx]] += 1

                test_loss += lossFunction(output, target)  # sum up batch loss
                pred = output.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                logger.debug(
                    {
                        "output": output,
                        "target": target,
                        "loss": test_loss,
                        "pred": pred,
                        "correct": correct,
                    }
                )

        test_loss /= len(testset_loader.dataset)
        logger.info(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(testset_loader.dataset),
                100.0 * correct / len(testset_loader.dataset),
            )
        )

        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', correct / len(testset_loader.dataset), epoch)
        confusion_matrix = pd.DataFrame(confusion_matrix)
        confusion_matrix.to_csv(results + "/ConfusionMatrix_" + str(epoch) + ".csv")
        logger.info(
            "Created Confusion Matrix and saved to: {}".format(
                results + "/ConfusionMatrix_" + str(epoch) + ".csv"
            )
        )

    writer.close()

    profile = cProfile.Profile()
    profile.runcall(train_model)
    ps = pstats.Stats(profile)
    ps.print_stats()

    ############################################################################
    #                          IT'S OVER, IT'S DONE!                           #
    ############################################################################
