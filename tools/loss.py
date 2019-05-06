"""
Author: Omkar Damle
Date: May 2018

All the functions required for offline and online training of Deeplab Resnet for MaskTrack method
"""

import numpy as np
import scipy.stats as scipystats
import torch.nn as nn
import torch

import os
import matplotlib.pyplot as plt
import scipy.misc as sm
import cv2
import random

def cross_entropy_loss(output, labels):
    """According to Pytorch documentation, nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss
    For loss,
        first argument should be class scores with shape: N,C,h,w
        second argument should be class labels with shape: N,h,w
    Assumes labels are binary
    """


    ce_loss = nn.CrossEntropyLoss()
    images, channels, height, width = output.data.shape
    loss = ce_loss(output, labels.long().view(images, height, width))
    return loss

def cross_entropy_loss_weighted(output, labels):

    temp = labels.data.cpu().numpy()
    freqCount = scipystats.itemfreq(temp)
    total = freqCount[0][1]+freqCount[1][1]
    perc_1 = freqCount[1][1]/total
    perc_0 = freqCount[0][1]/total

    weight_array = [perc_1, perc_0]

    if torch.cuda.is_available():
        weight_tensor = torch.FloatTensor(weight_array).cuda()
    else:
        weight_tensor = torch.FloatTensor(weight_array)

    ce_loss = nn.CrossEntropyLoss(weight=weight_tensor)
    images, channels, height, width = output.data.shape
    loss = ce_loss(output, labels.long().view(images, height, width))
    return loss
