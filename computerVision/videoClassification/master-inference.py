#!/usr/bin/env python3
# coding: utf-8

import os
import sys

import argparse
import numpy as np
import pandas as pd

from PIL import Image
from scipy.stats import mode
import cv2

from spatial_transformations.Compose import Compose
from spatial_transformations.ToTensor import ToTensor
from spatial_transformations.Scale import Scale
from spatial_transformations.RandomHorizontalFlip import RandomHorizontalFlip
from spatial_transformations.MultiScaleRandomCrop import MultiScaleRandomCrop

import torch
from torch import nn
import torchvision

import torch.optim as optim
import torch.nn.functional as F

from utils.logger import setup_logger

if __name__ == "__main__":
    ############################################################################
    # PARSING ARGUMENTS                                                        #
    # Done: to parse the arguments regarding model                             #
    # Expects the following arguments:                                         #
    #   model (str, default = r3d)                                             #
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
        "--fine_tune",
        default=True,
        type=bool,
        help="To fine tune the pretrained weights",
    )

    parser.add_argument(
        "--num_classes", default=10, type=int, help="number of video categories"
    )

    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Learning rate of the model optimizer",
    )

    parser.add_argument(
        "--momentum", default=0.9, type=float, help="momentum of the model optimizer"
    )

    parser.add_argument(
        "--weight_decay",
        default=1e-3,
        type=float,
        help="weight decay of the model optimizer",
    )

    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="batch size for training and testing dataloader",
    )

    parser.add_argument(
        "--num_workers", default=0, type=int, help="number of threads to use"
    )

    parser.add_argument(
        "--data", default="inferenceClips/", type=str, help="path of the data directory"
    )

    parser.add_argument(
        "--results",
        default="inferenceResults",
        type=str,
        help="path of the results directory",
    )

    parser.add_argument(
        "--saved_checkpoint", default=None, type=str, help="path of saved model"
    )

    parser.add_argument(
        "--video",
        default="VID_triton02_2019-07-27_07-15-04.mp4",
        type=str,
        help="path of video for inferencing",
    )

    parser.add_argument(
        "--time_interval", default=4, type=int, help="no of secs in between inference"
    )

    parser.add_argument("--fps", default=16, type=int, help="no of frames per second")

    parser.add_argument(
        "--get_video", default=False, type=bool, help="Use this to get inference clip"
    )

    parser.add_argument(
        "--display_inference",
        default=False,
        type=bool,
        help="Use this to get inference clip",
    )

    parser.add_argument(
        "--get_csv",
        default=True,
        type=bool,
        help="Use this to generate CSV file for inference",
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
        "--debug", default=False, type=bool, help="To set logging to debug"
    )

    args = parser.parse_args()

    logger = setup_logger("inference.log", debug=args.debug)

    logger.info(args)
    ############################################################################
    #                             END OF PARSING ARGUMENTS                     #
    ############################################################################

    ############################################################################
    # MODEL DEFINATION                                                         #
    # Done: to deine the model and modify according to our data                #
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
        model = torchvision.models.video.r3d_18(pretrained=False, progress=False)

    if args.model == "mc3":
        model = torchvision.models.video.mc3_18(pretrained=False, progress=True)

    if args.model == "r2plus1d":
        model = torchvision.models.video.r2plus1d_18(pretrained=False, progress=True)

    # Modifing the last layer according to our data
    model.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True)

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

    try:
        assert args.saved_checkpoint is not None
    except AssertionError as err:
        logger.exception("Saved model not provided")
        raise err

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
    # Done: Implement data augmentation and transformations                               #
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

    def infer(epoch):
        '''perform inference'''
        resultList = [
            [
                "videoName",
                "clipName",
                "action_score",
                "status",
            ]
        ]

        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            frameShift = 0
            actionStatus = []
            stopStatus = []

            if args.get_video:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(
                    "{}.mp4".format(args.video), fourcc, 10.0, (1280, 720)
                )  # get the dimensions of video

            cap = cv2.VideoCapture(args.video)
            frameNo = 1
            clip = []

            while True:
                ret, frame = cap.read()
                frameNo += 1

                if frame is None:
                    frameShift = frameNo
                    break

                if args.get_video:
                    returnFrame = frame.copy()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clip.append(frame)
                if frameNo % (args.fps * args.time_interval) == 0:
                    spatial_transform.randomize_parameters()
                    clip = [spatial_transform(Image.fromarray(img)) for img in clip]
                    video = torch.stack(clip, 0).permute(1, 0, 2, 3)

                    video.unsqueeze_(0)

                    clip = []

                    output = model(video)

                    # To get the confidence of the prediction
                    confidence = F.softmax(output, dim=1)

                    videoName = args.video.split("/")[-1]

                    rtime = "{}:{:02}".format(
                        ((frameShift + frameNo) // args.fps) // 60,
                        ((frameShift + frameNo) // args.fps) % 60,
                    )

                    status = np.argmax(output.cpu(), 1)[0].item()

                    actionStatus.append(status)

                    if args.get_csv:
                        resultList.append(
                            [
                                videoName,
                                rtime,
                                max(confidence[0]).item(),
                                status,
                            ]
                        )

                    result = {
                        "Time": rtime,
                        "actionScore": max(confidence[0]).item(),
                        "status": status,
                    }

                    logger.info(result)  # Return this result for integration

                    if args.display_inference:
                        cv2.putText(
                            frame,
                            "Status: {} ".format(status),
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 0, 0),
                            2,
                        )
                        cv2.imshow(
                            "Behavior: {}".format(videoName),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        )

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            logger.debug("User Ended the inference")
                            break

                    if args.get_video:
                        cv2.putText(
                            returnFrame,
                            "Status: {} ".format(status),
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            2,
                        )
                        out.write(returnFrame)

        # To release the video being read
        cap.release()

        if args.get_video:
            out.release()
            logger.debug("Released the video")

        if args.display_inference:
            cv2.destroyAllWindows()
            logger.debug("Distroyed the windows")

        if args.get_csv:
            resultList = pd.DataFrame(resultList)
            resultList.to_csv(
                results + "/results_" + args.video.split("/")[-1] + ".csv",
                index=False,
                header=None,
            )
            logger.info(
                "saved the results to {}".format(
                    results + "/results_" + args.video.split("/")[-1] + ".csv",
                    index=False,
                    header=None,
                )
            )

    ############################################################################
    #                             INFER THE MODEL                              #
    ############################################################################

    infer(epoch)
