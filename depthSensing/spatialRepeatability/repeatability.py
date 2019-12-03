#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm as colormap
from scipy import stats
from scipy import ndimage

from DepthProcessor import DepthProcessor as DP

np.warnings.filterwarnings('ignore')

class spacialRepeatability:
    def __init__(self, project1='MC20_1', project2='MC20_2', frames=1):
        self.height_threshold = 1 #Adjust height threshold
        self.min_bower_size = 1000 #Adjust minimum bower size
        self.frames = frames #Adjust number of frames for start and end positions

        self.project1 = project1
        self.project2 = project2

        self.localMasterDirectory1 = 'numpy_data/' + project1 + '/'
        self.localMasterDirectory2 = 'numpy_data/' + project2 + '/'

        self.logFile1 = self.localMasterDirectory1 + 'LogFile.txt'
        self.logFile2 = self.localMasterDirectory2 + 'LogFile.txt'

        self.cloudMasterDirectory = '/' #Update cloudMasterDirectory

        self.localRepeatability = '___repeatability/'
        self.overlapFile = 'overlap.csv'
        self.repeatabilityFile = 'repeatability.csv'
        self.overlapImage = 'overlapImages/'

        if not os.path.exists(self.localRepeatability):
            os.makedirs(self.localRepeatability)
            os.makedirs(self.localRepeatability+self.overlapImage)

        self.depthObj1 = DP(self.localMasterDirectory1, self.cloudMasterDirectory, self.logFile1)
        self.depthObj2 = DP(self.localMasterDirectory2, self.cloudMasterDirectory, self.logFile2)

        self.depthObj1.loadSmoothedArray()
        self.depthObj2.loadSmoothedArray()

    def overlap_analysis(self):
        xl = [['Trial1', 'Trial2', '%_Trial1', '%_Trial2']]
        trial1 = self.depthObj1.smoothDepthData
        trial2 = self.depthObj2.smoothDepthData
        
        print ('loading '+ self.project1 + ' ' + self.project2)
        if self.frames > 1: ax = 1
        else: ax = 0
        start = trial1[:self.frames].mean(axis=ax)
        end = trial1[trial1.shape[0]-self.frames:trial1.shape[0]].mean(axis=ax)

        diff1 = start-end
        img1 = ndimage.gaussian_filter(diff1.astype(np.double), (1, 1))
        blobs1 = abs(img1) > self.height_threshold
        labels1, nlabels1 = ndimage.label(blobs1)

        if nlabels1:
            ma = 0
            label1 = 0
            for i in range(1, nlabels1+1):
                ind = np.where(labels1==i)
                if ma < abs(diff1[ind].mean()) and sum(~np.isnan(diff1[np.where(labels1==i)])) > self.min_bower_size:
                    ma = abs(diff1[ind].mean())
                    label1 = i
            bower1 = np.zeros(shape=blobs1.shape, dtype=int)
            bower1[np.where(labels1==label1)] = 1

        start = trial2[:self.frames].mean(axis=ax)
        end = trial2[trial2.shape[0]-self.frames:trial2.shape[0]].mean(axis=ax)
        diff2 = start-end

        img2 = ndimage.gaussian_filter(diff2.astype(np.double), (1, 1))
        blobs2 = abs(img2) > self.height_threshold
        labels2, nlabels2 = ndimage.label(blobs2)

        if nlabels2:
            ma = 0
            label2 = 0
            for i in range(1, nlabels2+1):
                ind = np.where(labels2==i)
                if ma < abs(diff2[ind].mean()) and sum(~np.isnan(diff2[np.where(labels2==i)])) > self.min_bower_size:
                    ma = abs(diff2[ind].mean())
                    label2 = i
            bower2 = np.zeros(shape=blobs2.shape, dtype=int)
            bower2[np.where(labels2==label2)] = blobs2[np.where(labels2==label2)]
            bower2[np.where(labels2==label2)] = 2
            

        if label1 and label2 and bower1.shape == bower2.shape:
            xl.append([self.project1, self.project2, 
            round(len(np.where(bower1 + bower2 == 3)[0])*100/len(np.where(bower1==1)[0]), 2), 
            round(len(np.where(bower1 + bower2 == 3)[0])*100/len(np.where(bower2==2)[0]), 2)])
            
            fig = plt.figure()
            axes = fig.add_subplot(111)

            overlap = bower1+bower2

            mask = np.zeros(shape=overlap.shape)
            mask[np.where(overlap==0)] = 1
            masked_array = np.ma.array(overlap, mask=mask)
            cmap = colormap.jet
            cmap.set_bad('white',1.)
            
            plt.imshow(masked_array, interpolation='nearest', cmap=cmap)

            x1 = int(np.median(np.where(bower1!=0)[1]))
            y1 = int(np.median(np.where(bower1!=0)[0]))
            x2 = int(np.median(np.where(bower2!=0)[1]))
            y2 = int(np.median(np.where(bower2!=0)[0]))
            circle1 = plt.Circle((x1, y1), 5, color='orange')
            circle2 = plt.Circle((x2, y2), 5, color='black')
            plt.gcf().gca().add_artist(circle1)
            plt.gcf().gca().add_artist(circle2)
            plt.title(self.project1 + '_with_' + self.project2)
            plt.savefig(self.localRepeatability + self.overlapImage + self.project1 + '_with_' + self.project2 +'.png')
        else: print(label1, label2 , bower1.shape , bower2.shape)

        df = pd.DataFrame(xl)
        df.to_csv(self.localRepeatability + self.overlapFile, index=False, header=None)

    def repeatability(self):
        rvalues = [['Trial1', 'Trial2', 'R-Squared']]
        trial1 = self.depthObj1.smoothDepthData
        trial2 = self.depthObj2.smoothDepthData

        print ('loading '+ self.project1 + ' ' + self.project2)
        if self.frames > 1: ax = 1
        else: ax = 0
        start = trial1[:self.frames].mean(axis=ax)
        end = trial1[trial1.shape[0]-self.frames:trial1.shape[0]].mean(axis=ax)
        diff1 = start-end

        img1 = ndimage.gaussian_filter(diff1.astype(np.double), (1, 1))
        blobs1 = abs(img1) > self.height_threshold
        labels1, nlabels1 = ndimage.label(blobs1)

        start = trial2[:self.frames].mean(axis=ax)
        end = trial2[trial2.shape[0]-self.frames:trial2.shape[0]].mean(axis=ax)
        diff2 = start-end

        img2 = ndimage.gaussian_filter(diff2.astype(np.double), (1, 1))
        blobs2 = abs(img2) > self.height_threshold
        labels2, nlabels2 = ndimage.label(blobs2)

        if nlabels1 and nlabels2 and blobs1.shape == blobs2.shape:
            plt.title(self.project1 + '_vs_' + self.project2)
            x = np.ma.masked_array(diff1, ~blobs1).flatten()
            y = np.ma.masked_array(diff2, ~blobs2).flatten()
            plt.scatter(x, y)
            mask = ~np.isnan(x) & ~np.isnan(y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask],y[mask])
            rvalues.append([self.project1, self.project2, r_value*r_value])
            print (r_value*r_value)
        else:
            print (nlabels1, nlabels2, blobs1.shape, blobs2.shape)
        plt.savefig(self.localRepeatability + self.overlapImage + self.project1 + '_Vs_' + self.project2 +'.png')
        # plt.show()

        df2 = pd.DataFrame(rvalues)
        df2.to_csv(self.localRepeatability + self.overlapFile, index=False, header=None)
