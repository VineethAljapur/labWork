#!/usr/bin/env python
# coding: utf-8

import datetime
import numpy as np

from scipy import ndimage
from matplotlib import pyplot as plt

from DepthProcessor import DepthProcessor as DP

np.warnings.filterwarnings("ignore")  # to ignore warnings

class spacialUniformity:
    def __init__(
            self, project="MC20_1", day_height_threshold=0.5, min_bower_size=1000, frames=1
    ):
        self.day_height_threshold = day_height_threshold  # Adjust height threshold
        self.trial_height_threshold = 1  # Adjust height threshold
        self.min_bower_size = min_bower_size  # Adjust minimum bower size
        self.frames = frames  # Adjust number of frames for start and end positions

        self.file = project

        self.localMasterDirectory = "numpy_data/" + project + "/"
        self.cloudMasterDirectory = "/"
        self.logFile = self.localMasterDirectory + "LogFile.txt"

        self.depthObj = DP(
            self.localMasterDirectory, self.cloudMasterDirectory, self.logFile
        )
        self.depthObj.loadSmoothedArray()

    def getUniformity(self):
        if self.frames > 1:
            ax = 1
        else:
            ax = 0

        returnArray = []
        data = self.depthObj.smoothDepthData
        lp = self.depthObj.lp
        frames = self.frames

        start = data[: self.frames].mean(axis=ax)
        end = data[data.shape[0] - self.frames : data.shape[0]].mean(axis=ax)

        diff1 = start - end

        img = ndimage.gaussian_filter(diff1.astype(np.double), (1, 1))
        blobs1 = abs(img) > self.trial_height_threshold
        labels, nlabels = ndimage.label(blobs1)

        label = 0
        true_blobs = []  # blobs which pass both height and area threshold
        ma = 0
        for i in range(1, nlabels + 1):
            indexs = np.where(labels == i)
            if sum(~np.isnan(diff1[indexs])) > self.min_bower_size:
                true_blobs.append(i)
                if ma < abs(diff1[indexs].mean()):
                    ma = abs(diff1[indexs].mean())
                    label = i

        # if trial bower exists
        blob_diff = np.zeros(shape=diff1.shape)
        for blob in true_blobs:
            blob_diff[np.where(labels == blob)] = 1

        trial_diff = diff1
        trial_diff = np.nan_to_num(trial_diff)

        tFirst = lp.frames[0].time.replace(hour=0, minute=0, second=0, microsecond=0)
        tLast = lp.frames[-1].time.replace(
            hour=23, minute=59, second=59, microsecond=999999
        )

        trial_mask = np.zeros(shape=diff1.shape)
        trial_volume_change = 0
        for day in range(lp.numDays):

            start = tFirst + datetime.timedelta(hours=24 * day)
            end = tFirst + datetime.timedelta(hours=24 * (day + 1))

            diff = self.depthObj._returnHeightChange(start, end)

            img = ndimage.gaussian_filter(diff.astype(np.double), (1, 1))
            blobs = abs(img) > self.day_height_threshold
            labels, nlabels = ndimage.label(blobs)

            label = 0
            true_blobs = []
            ma = 0
            for i in range(1, nlabels + 1):
                indexs = np.where(labels == i)
                if sum(~np.isnan(diff[indexs])) > self.min_bower_size:
                    true_blobs.append(i)
                    if ma < abs(diff[indexs].mean()):
                        ma = abs(diff[indexs].mean())
                        label = i

            diff = np.nan_to_num(diff)
            blob_diff1 = np.zeros(shape=diff.shape)
            if label:

                for blob in true_blobs:
                    blob_diff1[np.where(labels == blob)] = 1

                diff_above_thres1 = np.zeros(shape=diff.shape)
                diff_above_thres1[np.where(blob_diff1 == 1)] = diff[
                    np.where(blob_diff1 == 1)
                ]

                day_volume_change = sum(sum(abs(diff_above_thres1)))
            else:
                day_volume_change = 0

            trial_volume_change += day_volume_change
            for blob in true_blobs:
                trial_mask[np.where(labels == blob)] = 1

        diff_above_thres = np.zeros(shape=diff1.shape)
        diff_above_thres[np.where(trial_mask == 1)] = diff1[np.where(trial_mask == 1)]

        # summing all the volume change for all blobs passing threshold
        trial_above_thres = diff_above_thres

        trial_mask = trial_mask == 1

        plots = plt.figure()

        fig, ax = plt.subplots(lp.numDays, 5, sharey=True, figsize=(20, lp.numDays * 5))

        for day in range(lp.numDays):

            start = tFirst + datetime.timedelta(hours=24 * day)
            end = tFirst + datetime.timedelta(hours=24 * (day + 1))

            diff = self.depthObj._returnHeightChange(start, end)

            img = ndimage.gaussian_filter(diff.astype(np.double), (1, 1))
            blobs = abs(img) > self.day_height_threshold
            labels, nlabels = ndimage.label(blobs)

            ax[day][0].set_title(
                "Combined Bower for whole trial ({})".format(self.file)
            )
            im = ax[day][0].imshow(np.ma.masked_array(diff1, ~trial_mask))
            plt.colorbar(im, ax=ax[day][0], fraction=0.035, pad=0.03)
            im.set_clim(-4, 4)

            label = 0
            true_blobs = []
            ma = 0
            for i in range(1, nlabels + 1):
                indexs = np.where(labels == i)
                if sum(~np.isnan(diff[indexs])) > self.min_bower_size:
                    true_blobs.append(i)
                    if ma < abs(diff[indexs].mean()):
                        ma = abs(diff[indexs].mean())
                        label = i

            diff = np.nan_to_num(diff)
            blob_diff1 = np.zeros(shape=diff.shape)
            if label:

                for blob in true_blobs:
                    blob_diff1[np.where(labels == blob)] = 1

                diff_above_thres1 = np.zeros(shape=diff.shape)
                diff_above_thres1[np.where(blob_diff1 == 1)] = diff[
                    np.where(blob_diff1 == 1)
                ]

                day_volume_change = sum(sum(abs(diff_above_thres1)))
            else:
                day_volume_change = 0

            ax[day][1].hold(True)
            blob_diff1 = blob_diff1 == 1
            im2 = ax[day][1].imshow(np.ma.masked_array(diff, ~blob_diff1))
            im2.set_clim(-2, 2)
            if day_volume_change != 0:
                ax[day][1].set_title("Day" + str(day + 1) + " bower")
                plt.colorbar(im2, ax=ax[day][1], fraction=0.035, pad=0.03)
            else:
                ax[day][1].set_title("No bower on Day" + str(day + 1))

            buildType = "N/A"

            # To check if there is building on that day
            if day_volume_change != 0:
                if sum(sum(diff_above_thres1)) > 0:
                    buildType = "castle"
                else:
                    buildType = "pit"
                day_diff = diff

                # volume ratio
                height_change_ratio = day_volume_change / trial_volume_change
                diff_index = np.zeros(shape=diff.shape)
                diff_index = day_diff - trial_diff * height_change_ratio

                unexpectedVolume = round(
                    np.nansum(abs(np.ma.masked_array(diff_index, ~trial_mask)))
                    * 0.1030168618
                    * 0.1030168618,
                    2,
                )
                actualChange = round(
                    np.nansum(abs(np.ma.masked_array(day_diff, ~trial_mask)))
                    * 0.1030168618
                    * 0.1030168618,
                    2,
                )

                uniformityIndex = round(1 - (unexpectedVolume / actualChange), 4)

                ax[day][2].hold(True)
                ax[day][2].set_title("Day" + str(day + 1) + " Change")
                im3 = ax[day][2].imshow(np.ma.masked_array(diff, ~trial_mask))
                plt.colorbar(im3, ax=ax[day][2], fraction=0.035, pad=0.03)
                im3.set_clim(-2, 2)

                ax[day][3].hold(True)
                ax[day][3].set_title("Day " + str(day + 1) + " Uniformity")
                display = np.zeros(shape=diff_index.shape)
                display[np.where(trial_mask == True)] = diff_index[
                    np.where(trial_mask == True)
                ]

                im4 = ax[day][3].imshow(
                    np.ma.masked_array(diff_index, ~trial_mask), cmap=plt.cm.coolwarm
                )
                plt.colorbar(im4, ax=ax[day][3], fraction=0.035, pad=0.03)
                im4.set_clim(-2, 2)

                ax[day][4].hold(True)
                ax[day][4].set_title("Day " + str(day + 1) + " TotalChange")

                im5 = ax[day][4].imshow(
                    np.ma.masked_array(abs(diff_index), ~trial_mask),
                    cmap=plt.cm.gray.reversed(),
                )
                plt.colorbar(im5, ax=ax[day][4], fraction=0.035, pad=0.03)
                im5.set_clim(0, 2)
                plt.hold(True)

                returnArray.append(
                    [
                        self.file,
                        day + 1,
                        buildType,
                        uniformityIndex,
                        int(unexpectedVolume),
                        int(day_volume_change * 0.1030168618 * 0.1030168618),
                    ]
                )
            else:
                ax[day][2].hold(True)
                ax[day][2].set_title("Day" + str(day + 1) + " Change")
                im3 = ax[day][2].imshow(np.ma.masked_array(diff, ~trial_mask))
                plt.colorbar(im3, ax=ax[day][2], fraction=0.035, pad=0.03)
                im3.set_clim(-2, 2)

                ax[day][3].hold(True)
                ax[day][3].set_title("No bower on day" + str(day + 1))
                im4 = ax[day][3].imshow(np.full(diff.shape, np.nan))
                im4.set_clim(
                    np.ma.masked_array(diff1, ~trial_mask).min(),
                    np.ma.masked_array(diff1, ~trial_mask).max(),
                )

                ax[day][4].hold(True)
                ax[day][4].set_title("No bower on day" + str(day + 1))
                im5 = ax[day][4].imshow(np.full(diff.shape, np.nan))
                im5.set_clim(
                    np.ma.masked_array(diff1, ~trial_mask).min(),
                    np.ma.masked_array(diff1, ~trial_mask).max(),
                )

                returnArray.append(
                    [self.file, day + 1, buildType, 0, 0, day_volume_change]
                )

        fig.tight_layout()
        plt.savefig("./uniformityResults/" + self.file + ".pdf")

        return returnArray
