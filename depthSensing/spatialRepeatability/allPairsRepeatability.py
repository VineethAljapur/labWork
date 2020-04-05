#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

home = "numpy_data/"
np.warnings.filterwarnings("ignore")
xl = [
    [
        "Trial1",
        "Trial2",
        "%_downward",
        "%_upwards",
        "%_overlap",
        "Trial 1 # above threshold",
        "Trial 1 # pit",
        "Trial 1 # castle",
        "Trial 2 # above threshold",
        "Trial 2 # pit",
        "Trial 2 # castle",
        "# Union",
        "Shared up and down",
        "Shared down",
        "Shared up",
    ]
]

df = pd.read_excel("numpy_data/overlap_pairs.xlsx")

for index, row in df.iterrows():
    print("loading " + row["Trial1"] + " " + row["Trial2"])
    try:
        data_trial1 = np.load(
            os.path.join(home, row["Trial1"], "smoothedDepthData.npy"), mmap_mode="r"
        )
    except:
        print(row["Trial1"] + " does not exists")
        continue
    f1 = data_trial1[:1]
    fn = data_trial1[data_trial1.shape[0] - 2 : data_trial1.shape[0] - 1]
    diff1 = (f1 - fn)[0]
    img1 = ndimage.gaussian_filter(diff1.astype(np.double), (1, 1))
    blobs1 = abs(img1) > 1
    labels, nlabels = ndimage.label(blobs1)
    if nlabels:
        ma = 0
        label1 = 0
        passed_labels = []
        for i in range(1, nlabels + 1):
            ind = np.where(labels == i)
            if sum(~np.isnan(diff1[ind])) > 1000:
                passed_labels.append(i)
                label1 = 1
        temp_bower1 = np.zeros(shape=blobs1.shape, dtype=int)
        for label in passed_labels:
            temp_bower1[np.where(labels == label)] = 1

        bower1 = np.zeros(shape=blobs1.shape, dtype=int)

        bower1[np.where((diff1 > 0) & (temp_bower1 == 1))] = 1
        bower1[np.where((diff1 < 0) & (temp_bower1 == 1))] = 3

    try:
        data_trial1 = np.load(
            os.path.join(home, row["Trial2"], "smoothedDepthData.npy"), mmap_mode="r"
        )
    except:
        print(row["Trial2"] + " does not exists")
        continue
    f1 = data_trial1[:1]
    fn = data_trial1[data_trial1.shape[0] - 2 : data_trial1.shape[0] - 1]
    diff1 = (f1 - fn)[0]

    img1 = ndimage.gaussian_filter(diff1.astype(np.double), (1, 1))
    blobs1 = abs(img1) > 1
    labels, nlabels2 = ndimage.label(blobs1)

    if nlabels2:
        ma = 0
        label2 = 0
        passed_labels = []
        for i in range(1, nlabels2 + 1):
            ind = np.where(labels == i)
            if sum(~np.isnan(diff1[np.where(labels == i)])) > 1000:
                passed_labels.append(i)
                label2 = i
        temp_bower2 = np.zeros(shape=blobs1.shape, dtype=int)
        for label in passed_labels:
            temp_bower2[np.where(labels == label)] = 1

        bower2 = np.zeros(shape=blobs1.shape, dtype=int)

        bower2[np.where((diff1 > 0) & (temp_bower2 == 1))] = 1
        bower2[np.where((diff1 < 0) & (temp_bower2 == 1))] = 3

    if label1 and label2 and bower1.shape == bower2.shape:
        overlap = bower1 + bower2
        try:
            downwards = round(
                len(np.where(overlap == 6)[0])
                * 100
                / len(np.where((bower1 == 3) | (bower2 == 3))[0]),
                2,
            )
        except:
            downwards = 0

        try:
            upwards = round(
                len(np.where(overlap == 2)[0])
                * 100
                / len(np.where((bower1 == 1) | (bower2 == 1))[0]),
                2,
            )
        except:
            upwards = 0

        try:
            both = round(
                len(np.where((overlap == 2) | (overlap == 6))[0])
                * 100
                / len(
                    np.where(
                        (bower1 == 1) | (bower1 == 3) | (bower2 == 1) | (bower2 == 3)
                    )[0]
                ),
                2,
            )
        except:
            both = 0

        trial1_up_down = len(np.where((bower1 == 1) | (bower1 == 3))[0])
        trial1_down = len(np.where(bower1 == 3)[0])
        trial1_up = len(np.where(bower1 == 1)[0])

        trial2_up_down = len(np.where((bower2 == 1) | (bower2 == 3))[0])
        trial2_down = len(np.where(bower2 == 3)[0])
        trial2_up = len(np.where(bower2 == 1)[0])

        union_pixels = len(
            np.where((overlap == 2) | (overlap == 6) | (overlap == 4))[0]
        )
        shared_up_down = len(np.where((overlap == 2) | (overlap == 6))[0])
        shared_down = len(np.where(overlap == 6)[0])
        shared_up = len(np.where(overlap == 2)[0])

        xl.append(
            [
                row["Trial1"],
                row["Trial2"],
                downwards,
                upwards,
                both,
                trial1_up_down,
                trial1_down,
                trial1_up,
                trial2_up_down,
                trial2_down,
                trial2_up,
                union_pixels,
                shared_up_down,
                shared_down,
                shared_up,
            ]
        )
        print(
            "overlap of {} with {} is {}".format(
                row["Trial1"],
                row["Trial2"],
                round(
                    len(np.where((overlap == 2) | (overlap == 6))[0])
                    * 100
                    / len(
                        np.where(
                            (bower1 == 1)
                            | (bower1 == 3)
                            | (bower2 == 1)
                            | (bower2 == 3)
                        )[0]
                    ),
                    2,
                ),
            )
        )

        fig = plt.figure()
        axes = fig.add_subplot(111)

        colors = [
            (31 / 255, 144 / 255, 140 / 255),
            (42 / 255, 188 / 255, 88 / 255),
            (254 / 255, 232 / 255, 35 / 255),
            (49 / 255, 72 / 255, 112 / 255),
            (129 / 255, 123 / 255, 80 / 255),
            (68 / 255, 0, 84 / 255),
        ]
        cmap = matplotlib.colors.ListedColormap(colors, name="colors", N=None)

        plt.imshow(overlap, interpolation="nearest", cmap=cmap, vmin=0, vmax=6)
        plt.title(row["Trial1"] + "_with_" + row["Trial2"])
        plt.savefig(
            home + "overlaps/" + row["Trial1"] + "_with_" + row["Trial2"] + ".png",
            dpi=300,
        )
        plt.show()
    else:
        print(label1, label2, bower1.shape, bower2.shape)
    print()
df = pd.DataFrame(xl)
df.to_csv(home + "repeatability.csv", index=False, header=None)
