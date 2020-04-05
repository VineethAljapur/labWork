#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import unittest
import warnings
from PIL import Image
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

import skvideo.io

from utils.logger import setup_logger

from spatial_transformations.Compose import Compose
from spatial_transformations.ToTensor import ToTensor
from spatial_transformations.Scale import Scale
from spatial_transformations.RandomHorizontalFlip import RandomHorizontalFlip
from spatial_transformations.MultiScaleRandomCrop import MultiScaleRandomCrop

from CichlidsDataLoader import Cichlids


class TestCalc(unittest.TestCase):
    """Test all modules of feed analysis"""

    @classmethod
    def setUpClass(cls):
        """Defining common terms"""
        cls.logger = setup_logger("unitTesting.log", debug=False)
        cls.size = 56
        scales = [1.0]
        for i in range(1, 5):
            scales.append(scales[-1] * 0.84089641525)
        cls.scales = scales
        cls.model = torchvision.models.video.r3d_18(pretrained=False, progress=False)
        cls.num_classes = 2

        cls.dataDirectory = "testData"
        os.mkdir(cls.dataDirectory)
        os.mkdir("testData/testing")

        warnings.simplefilter("ignore", DeprecationWarning)
        video_array = (np.random.random(size=(16, 112, 200, 3)) * 255).astype(np.uint8)

        skvideo.io.vwrite(os.path.join("testData/testing", "video.mp4"), video_array)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dataDirectory)

    def setUp(self):
        """Creating test image and video from random values"""
        img_array = np.random.randint(0, 256, (200, 112, 3))
        video_array = np.random.randint(0, 256, (16, 200, 112, 3))
        self.img = Image.fromarray(img_array.astype("uint8")).convert("RGB")
        self.video = [
            Image.fromarray(img.astype("uint8")).convert("RGB") for img in video_array
        ]

    def test_Compose(self):
        """Testing compose module
		Test Cases:
			1. Getting a composer instance
			2. Applying spatial transformations through composer
			3. Checking the output image after transformations
		"""
        composer = Compose(
            [
                MultiScaleRandomCrop(self.scales, self.size),
                RandomHorizontalFlip(),
                ToTensor(),
            ],
            self.logger,
        )
        self.assertIsNotNone(composer)
        composer.randomize_parameters()
        clip = [composer(img) for img in self.video]
        self.assertIsNotNone(clip)
        self.assertEqual(len(clip), len(self.video))

    def test_ToTensor(self):
        """Testing ToTensor module
		Test Cases:
			1. Getting a convTensor instance
			2. Converting image to tensor through convTensor
			3. Checking the output as Tensor
			4. Checking the output values between 0 and 1
			5. Checking output channels
		"""
        convTensor = ToTensor()
        value = convTensor(self.img, self.logger)
        self.assertIsNotNone(value)
        self.assertEqual(type(value), torch.Tensor)
        self.assertEqual(torch.prod(value >= 0).item(), 1)
        self.assertEqual(torch.prod(value <= 1).item(), 1)
        self.assertEqual(value.size()[0], 3)

    def test_Scale(self):
        """Testing Scale module
		Test Cases:
			1. Getting a scaler instance
			2. Rescaling image
			3. Checking the output image dimension
		"""
        scaler = Scale(self.size)
        value = scaler(self.img, self.logger)
        self.assertIsNotNone(value)
        self.assertTrue(self.size == value.size[0] or self.size == value.size[1])

    def test_RandomHorizontalFlip(self):
        """Testing RandomHorizontalFlip module
		Test Cases:
			1. Getting a randomFlip instance
			2. Randomly Fliping the image horizontally
			3. Checking the output image dimension
		"""
        randomFlip = RandomHorizontalFlip()
        randomFlip.randomize_parameters()
        value = randomFlip(self.img, self.logger)
        self.assertIsNotNone(value)
        self.assertTrue(self.img.size == value.size)

    def test_MultiScaleRandomCrop(self):
        """Testing MultiScaleRandomCrop module
		Test Cases:
			1. Getting a randomCrop instance
			2. Cropping the image in a random location
			3. Checking the output image dimension with expected size
		"""
        randomCrop = MultiScaleRandomCrop(self.scales, self.size)
        randomCrop.randomize_parameters()
        value = randomCrop(self.img, self.logger)
        self.assertIsNotNone(value)
        self.assertEqual(value.size, (self.size, self.size))

    def test_training(self):
        """Testing master-train module
		Test Cases:
			1. Getting a model from PyTorch
			2. modifying last fully connected layer
			3. Geting output of classification for the video
			4. Checking the number of output classes
		"""
        self.assertIsNotNone(self.model)
        self.model.fc = torch.nn.Linear(
            in_features=512, out_features=self.num_classes, bias=True
        )
        convTensor = ToTensor()
        clip = [convTensor(img, self.logger) for img in self.video]
        video = torch.stack(clip, 0).permute(1, 0, 2, 3)
        video.unsqueeze_(0)
        output = self.model(video)
        self.assertIsNotNone(output)
        self.assertEqual(len(output[0]), self.num_classes)

    def test_inference(self):
        """Testing master-train module
		Test Cases:
			1. Getting a model from PyTorch
			2. modifying last fully connected layer
			3. Geting output of classification for the video
			4. Checking the number of output classes
			5. Checking for confidence of prediction
		"""
        self.assertIsNotNone(self.model)
        self.model.fc = torch.nn.Linear(
            in_features=512, out_features=self.num_classes, bias=True
        )
        convTensor = ToTensor()
        clip = [convTensor(img, self.logger) for img in self.video]
        video = torch.stack(clip, 0).permute(1, 0, 2, 3)
        video.unsqueeze_(0)
        output = self.model(video)
        self.assertIsNotNone(output)
        self.assertEqual(len(output[0]), self.num_classes)
        confidence = F.softmax(output, dim=1)
        self.assertIsNotNone(confidence)
        self.assertEqual(len(confidence[0]), self.num_classes)

    def test_CichlidsDataLoader(self):
        """Testing Cichlids Data Loader module
		Test Cases:
			1. Getting a testset instance
			2. Getting a dataLoader instance
			3. Checking for data and target from dataLoader
			4. Checking for length of target
			5. Checking for target value
			6. Checking for type of data as Tensor
		"""
        spatial_transform = Compose(
            [
                MultiScaleRandomCrop(self.scales, self.size),
                RandomHorizontalFlip(),
                ToTensor(),
            ],
            self.logger,
        )

        # Load the trainset
        testset = Cichlids(
            root=self.dataDirectory,
            logger=self.logger,
            preload=False,
            spatial_transform=spatial_transform,
        )

        self.assertIsNotNone(testset)

        dataLoader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

        self.assertIsNotNone(dataLoader)

        for batch_idx, (data, target) in enumerate(dataLoader):
            self.assertIsNotNone(data)
            self.assertIsNotNone(target)
            self.assertEqual(len(target), 1)
            self.assertEqual(target.item(), 0)
            self.assertEqual(type(data), torch.Tensor)


if __name__ == "__main__":
    unittest.main()
