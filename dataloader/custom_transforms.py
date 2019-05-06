
import torch
import torch.utils.data as data
import torch.nn as nn

import os, math, random
from os.path import join
import numpy as np

import cv2
import imgaug as ia
from imgaug import augmenters as iaa


def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    j = (j+1)/2
    j = int(np.ceil((j+1)/2.0))
    j = (j+1)/2
    return int(j)

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.Upsample(size=(size, size), mode='bilinear')
    labelVar = torch.from_numpy(label.transpose(3, 2, 0, 1))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)
    label_resized[label_resized>0.3]  = 1
    label_resized[label_resized != 0]  = 1

    return label_resized

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def aug_batch(img, gt):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes2 = lambda aug: iaa.Sometimes(0.9, aug)

    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.Add((-10, 10), per_channel=0.5),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ], random_order=True
        )
    
    seq2 = iaa.Sequential(
        [
            sometimes2(iaa.Affine(
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                shear=(-10, 10), # shear by -16 to +16 degrees
                order=0, # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            #sometimes2(iaa.CoarseDropout(0.2, size_percent=(0.1, 0.5)
            #))
        ], random_order=True
        )
    scale = random.uniform(0.5, 1.3) #random.uniform(0.5,1.5) does not fit in a Titan X with the present version of pytorch, so we random scaling in the range (0.5,1.3), different than caffe implementation in that caffe used only 4 fixed scales. Refer to read me
    scale=1
    dim = int(scale*321)

    flip_p = random.uniform(0, 1)

    img_temp = flip(img,flip_p)
    gt_temp = flip(gt,flip_p)

    seq_det = seq.to_deterministic()
    img_temp = seq_det.augment_image(img_temp).astype(float)
    img_temp[:,:,0] = img_temp[:,:,0] - 104.00699
    img_temp[:,:,1] = img_temp[:,:,1] - 116.66877
    img_temp[:,:,2] = img_temp[:,:,2] - 122.67892
    img_temp = cv2.resize(img_temp,(dim,dim))

    gt_temp = ia.SegmentationMapOnImage(gt_temp, shape=gt_temp.shape, nb_classes=2)
    gt_temp_map = seq_det.augment_segmentation_maps([gt_temp])[0]
    gt_temp = gt_temp_map.get_arr_int()
    gt_temp = cv2.resize(gt_temp,(321,321) , interpolation = cv2.INTER_NEAREST).astype(float)

    kernel = np.ones((int(scale*5), int(scale*5)), np.uint8)

    
    mask = seq2.augment_segmentation_maps([gt_temp_map])[0].get_arr_int()
    mask= cv2.resize(mask,(dim,dim) , interpolation = cv2.INTER_NEAREST).astype(float)
    
    bb = cv2.boundingRect(gt_temp.astype('uint8'))
 
    if bb[2] != 0 and bb[3] != 0:
        fc = np.ones([dim, dim, 1]) * -100
        #fc[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], 0] = 100
        if flip_p <= 1.0:
            aug_p = random.uniform(0, 1)
            it = random.randint(1, 5)

            aug = np.expand_dims(cv2.dilate(mask, kernel, iterations=it), 2)
            fc[np.where(aug==1)] = 100 
    else:
        fc = np.ones([dim, dim, 1]) * -100
    
    image = np.dstack([img_temp, fc])
    gt_temp = np.expand_dims(gt_temp, 2)
    gt = np.expand_dims(gt_temp, 3)
    label = resize_label_batch(gt, outS(dim))
    label = label.squeeze(3)

    return image, label
