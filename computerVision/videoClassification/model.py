#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torchvision

import torch.optim as optim
import torch.nn.functional as F  

import torchvision.transforms as transforms
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import glob
import os.path as osp

import os
import skvideo
from skvideo import io as vp
from time import time

import argparse
import numpy as np
import pandas as pd

import random
import math
import numbers
import collections
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None

############################################################################
# PARSING ARGUMENTS                                                        #
# TODO: to parse the arguments regarding model                             #
# Expects the following arguments:                                         #
#   model (str, default = r3d)                                             #
#   pre_trained (bool, default = True)                                     #
#   fine_tune (bool, default = True)                                       #
#   learning_rate (float, default = 0.001)                                 #
#   momentum (float, default = 0.9)                                        #
#   weight_decay (float, default = 1e-3)                                   #
#   epochs (int, default = 50)                                             #
#   save_interval (int, default = 5)                                       #
############################################################################
parser = argparse.ArgumentParser()

parser.add_argument(
    '--model',
    default='r3d',
    type=str,
    help='(r3d | mc3 | r2plus1d)')

parser.add_argument(
    '--pre_trained',
    default=True,
    type=bool,
    help='To use the pre_trained weights for model initialization')

parser.add_argument(
    '--fine_tune',
    default=True,
    type=bool,
    help='To fine tune the pretrained weights')

parser.add_argument(
    '--num_classes',
    default=10,
    type=int,
    help='number of video categories')

parser.add_argument(
    '--learning_rate',
    default=0.001,
    type=float,
    help='Learning rate of the model optimizer')

parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    help='momentum of the model optimizer')

parser.add_argument(
    '--weight_decay',
    default=1e-3,
    type=float,
    help='weight decay of the model optimizer')

parser.add_argument(
    '--epochs',
    default=50,
    type=int,
    help='Number of epoch to train the model')

parser.add_argument(
    '--save_interval',
    default=5,
    type=int,
    help='epoch interval to save the model')

parser.add_argument(
    '--results',
    default='',
    type=str,
    help='path of the results directory')

args = parser.parse_args()

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
assert args.model in ['r3d', 'mc3', 'r2plus1d']

if args.model == 'r3d':
    model = torchvision.models.video.r3d_18(pretrained=args.pre_trained, progress=True)

if args.model == 'mc3':
    model = torchvision.models.video.mc3_18(pretrained=args.pre_trained, progress=True)    

if args.model == 'r2plus1d':
    model = torchvision.models.video.r2plus1d_18(pretrained=args.pre_trained, progress=True)    

# Modifing the last layer according to our data
model.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True)

if not args.fine_tune:
    for name,param in model.named_parameters():
        param.requires_grad = False

# To parallalize the model. By default it uses all available gpu. 
# Set visible devices using CUDA_VISIBLE_DEVICE
model = model.cuda()
model = nn.DataParallel(model, device_ids=None)

# Optimizer for the model
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
 momentum=args.momentum, weight_decay=args.weight_decay)

############################################################################
#                             END OF MODEL DEFINATION                      #
############################################################################


############################################################################
# DATA LOADER                                                              #
# TODO: Implement the custom data loader for Cichlids dataset              #
# Expects the file structure of the following format:                      #
# MLclips/                                                                 #
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
    def __init__(self,
                 root,
                 spatial_transform=None,
                 preload=False):
        """ Intialize the Cichlids dataset
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        self.videos = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.spatial_transform = spatial_transform

        # read filenames
        for i, class_dir in enumerate(os.listdir(root)):
            filenames = glob.glob(osp.join(root, class_dir, '*.mp4'))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        # if preload dataset into memory
        if preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
        for image_fn, label in self.filenames:            
            # load videos
            video = vp.vread(image_fn)
            video = np.reshape(video, (video.shape[3], video.shape[0], video.shape[1], video.shape[2]))
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
            video = np.reshape(video, (video.shape[3], video.shape[0], video.shape[1], video.shape[2]))
                
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            video = video.reshape((video.shape[1], video.shape[2], video.shape[3], video.shape[0]))
            clip = [self.spatial_transform(Image.fromarray(img)) for img in video]
            video = torch.stack(clip, 0).permute(1, 0, 2, 3)
        
        # return video and label
        return video, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

############################################################################
#                             END OF DATA LOADER                           #
############################################################################


#######################################################################################
# DATA TRANSFORMATIONS                                                                #
# TODO: Implement data augmentation and transformations                               #
# These classes are taken from Kensho Hara, et als implimentaion                      #
# https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/spatial_transforms.py  #
#######################################################################################

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()

class MultiScaleRandomCrop(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

scales = [1.0]
for i in range(1, 5):
    scales.append(scales[-1] * 0.84089641525)

# Defining spatial transformations
spatial_transform = Compose([
            MultiScaleRandomCrop(scales, 112),
            RandomHorizontalFlip(),
            ToTensor(), Normalize([0, 0, 0], [1, 1, 1])
        ])

# Load the trainset
trainset = Cichlids(
    root='MLclips/training',
    preload=False, spatial_transform=spatial_transform, transform=None
)

trainset_loader = DataLoader(trainset, batch_size=3, shuffle=True, num_workers=6)

# Load the testset
testset = Cichlids(
    root='MLclips/testing',
    preload=False, spatial_transform= spatial_transform, transform=None
)

testset_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

############################################################################
#                             END OF DATA TRANSFORMATIONS                  #
############################################################################

results = 'r3d_18FTweights2'
if not os.path.isdir(results):
    os.mkdir(results)

def train_model(epochs=50, log_interval=1000):    
    model.train()
    for t in range(epochs):
        start = time()
        iteration = 0
        avg_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(trainset_loader):
            
            target = target.cuda(async=True)
            
            data = Variable(data)
            target = Variable(target)
            
            output = model(data)
            
            lossFunction = nn.CrossEntropyLoss()
            lossFunction = lossFunction.cuda()
            
            loss = lossFunction(output, target)
            avg_loss += loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    t, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        end = time()
        print ('\nSummary: Epoch {}'.format(t))
        print('Time taken for this epoch: {:.2f}s'.format(end-start))
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        avg_loss/len(trainset_loader.dataset), correct, len(trainset_loader.dataset),
        100. * correct / len(trainset_loader.dataset)))
        
        save_file_path = os.path.join(results,
                                      'save_{}.pth'.format(t))
        states = {
            'epoch': t + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
        check_accuracy(t) # evaluate at the end of epoch
    torch.cuda.empty_cache()

def check_accuracy(epoch):
    num_correct = 0
    num_samples = 0
    test_loss = 0
    correct = 0

    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    confidence_for_each_validation = {}

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for data, target in testset_loader:
            target = target.cuda(async=True)
    
            data = Variable(data)
            target = Variable(target)

            output = model(data)
            
            lossFunction = nn.CrossEntropyLoss()
            lossFunction = lossFunction.cuda()
            
            rows = [int(x) for x in target]
            columns = [int(x) for x in np.argmax(output.cpu(),1)]
            assert len(rows) == len(columns)
            for idx in range(len(rows)):
                confusion_matrix[rows[idx]][columns[idx]] +=1

            test_loss += lossFunction(output, target) # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(testset_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))

    confusion_matrix = pd.DataFrame(confusion_matrix)
    confusion_matrix.to_csv(results + '/ConfusionMatrix_' + str(epoch) + '.csv')
    
train_model()