#!/usr/bin/env python3
# coding: utf-8

import glob
import os

from ast import literal_eval

import cv2

import skvideo
from skvideo import io as vp

import torch
from torch.utils.data import Dataset

from PIL import Image

############################################################################
# DATA LOADER                                                              #
# Done: Implement the custom data loader for Cichlids dataset              #
# Expects the file structure of the following format:                      #
# data/                                                                    #
#    training/                                                             #
#      class1/                                                             #
#        .../ (directories of class names)                                 #
#          .../ (mp4 files)                                                #
#    testing/                                                              #
#      class1/                                                             #
#        .../ (directories of class names)                                 #
#          .../ (mp4 files)                                                #
############################################################################


class Cichlids(Dataset):
    """
    A customized data loader for Cichlids.
    """

    def __init__(
        self, root, logger, resize=None, spatial_transform=None, preload=False
    ):
        """ Intialize the Cichlids dataset
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        Returns:
            - video: Tensor correponding to RGB clip after spatial transformations on each
                     image in the clip
            - Label: Value corresponding to the label of the action class of the clip
        """
        self.videos = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.spatial_transform = spatial_transform
        self.resize = resize
        self.logger = logger

        # read filenames
        for i, class_dir in enumerate(os.listdir(root)):
            filenames = glob.glob(os.path.join(root, class_dir, "*.mp4"))
            for fn in filenames:
                self.filenames.append((fn, i))
                self.logger.debug(
                    "Filename: {}, Label: {}".format(fn, i)
                )  # (filename, label) pair # Log in DEBUG

        # if preload dataset into memory
        if preload:
            self._preload()

        self.len = len(self.filenames)

    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.videos = []
        for video_fn, label in self.filenames:
            # load videos
            video = vp.vread(video_fn)
            self.videos.append(video.copy())
            self.labels.append(label)

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.videos is not None:
            # If dataset is preloaded
            video = self.videos[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            video_fn, label = self.filenames[index]
            video = vp.vread(video_fn)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            if self.resize is not None:
                video = [cv2.resize(img, literal_eval(self.resize)) for img in video]
            clip = [self.spatial_transform(Image.fromarray(img)) for img in video]
            video = torch.stack(clip, 0).permute(1, 0, 2, 3)

        self.logger.debug((video.shape, label))
        # return video and label
        return video, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


if __name__ == "__main__":
    pass
