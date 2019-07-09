import torch
import torch.utils.data as data

import os, math, random
from os.path import join
import numpy as np

import cv2
from .custom_transforms import aug_batch
from PIL import Image
import matplotlib.pyplot as plt

class DAVIS(data.Dataset):
    def __init__(self, train=False, root = '', replicates = 1, aug=False):
        self.replicates = replicates
        self.aug = aug

        if train:
            seqs_file = 'train.txt'
        else:
            seqs_file = 'val.txt'

        with open(join(root, 'ImageSets/480p', seqs_file)) as f:
            files = f.readlines()

        self.image_list = []
        self.gt_list = []
        for f in files:
            im, gt = f.split()
            img = join(root, im[1:])
            gt = join(root, gt[1:])
            self.image_list += [img]
            self.gt_list += [gt]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape
       
        assert (len(self.image_list) == len(self.gt_list))

    def __getitem__(self, index):

        index = index % self.size
        img = cv2.imread(self.image_list[index])

        #gt = np.expand_dims(np.array(Image.open(self.gt_list[index])), axis=2)
        gt = np.expand_dims(cv2.imread(self.gt_list[index], 0), axis=2)
        gt[gt==255] = 1


        image_size = img.shape[:2]

        if self.aug:
            img, gt = aug_batch(img, gt)
            
        
        img = img.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))



        return img, gt

    def __len__(self):
        return self.size * self.replicates

class DAVIS2016(DAVIS):
    def __init__(self, train=False, root = '', replicates = 1, aug=False):
        super(DAVIS2016, self).__init__(train=train, root = root, replicates = replicates, aug=aug)

class YTB_VOS(data.Dataset):
    def __init__(self, train=False, root = '', replicates = 1, aug=False):
        self.replicates = replicates
        self.aug = aug

        if train:
            seq = 'train'
        else:
            seq = 'valid'

        image_root = join(root, seq, 'JPEGImages')
        gt_root = join(root, seq, 'Annotations')


        self.image_list = []
        self.gt_list = []
        for seq in os.listdir(image_root):
            files = sorted(os.listdir(join(image_root, seq)))
            for i in range(len(files)):
                img = join(image_root, seq, files[i])
                gt = join(gt_root, seq, files[i][:-4]+'.png')
                self.image_list += [img]
                self.gt_list += [gt]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape

        assert (len(self.image_list) == len(self.gt_list))

    def __getitem__(self, index):

        index = index % self.size
        img = cv2.imread(self.image_list[index])

        gt = np.expand_dims(np.array(Image.open(self.gt_list[index])), axis=3)
        labels = np.unique(gt).tolist()
        if len(labels) != 1:
            labels = [l for l in labels if l != 0]
            idx = random.choice(labels)
            gt[gt!=idx] = 0
            gt[gt==idx] = 1

        if self.aug:
            img, gt = aug_batch(img, gt)

        img = img.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img, gt

    def __len__(self):
        return self.size * self.replicates

class ECSSD(data.Dataset):
    def __init__(self, root = '', replicates = 1, aug=False):
        self.replicates = replicates
        self.aug = aug

        image_root = join(root, 'images')
        gt_root = join(root, 'ground_truth_mask')
        
        self.image_list = []
        self.gt_list = []
        files = sorted(os.listdir(image_root))
        for i in range(len(files)):
            img = join(image_root, files[i])
            gt = join(gt_root, files[i][:-4]+'.png')
            self.image_list += [img]
            self.gt_list += [gt]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape
       
        assert (len(self.image_list) == len(self.gt_list))

    def __getitem__(self, index):

        index = index % self.size
        img = cv2.imread(self.image_list[index])
        

        gt = np.expand_dims(np.array(Image.open(self.gt_list[index])), axis=2)
        gt[gt==255] = 1


        image_size = img.shape[:2]

        if self.aug:
            img, gt = aug_batch(img, gt)
            
        
        img = img.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))



        return img, gt

    def __len__(self):
        return self.size * self.replicates

class MSRA10K(data.Dataset):
    def __init__(self, root = '', replicates = 1, aug=False):
        self.replicates = replicates
        self.aug = aug

        image_root = join(root, 'images')
        gt_root = join(root, 'annotations')
        
        
        self.image_list = []
        self.gt_list = []
        files = sorted(os.listdir(image_root))
        for i in range(len(files)):
            img = join(image_root, files[i])
            gt = join(gt_root, files[i][:-4]+'.png')
            self.image_list += [img]
            self.gt_list += [gt]
                
        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape

        assert (len(self.image_list) == len(self.gt_list))

    def __getitem__(self, index):

        index = index % self.size
        img = cv2.imread(self.image_list[index])
        

        gt = np.expand_dims(np.array(Image.open(self.gt_list[index])), axis=2)
        gt[gt!=255] = 0
        gt[gt==255] = 1


        image_size = img.shape[:2]

        if self.aug:
            img, gt = aug_batch(img, gt)
            
        
        img = img.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img, gt

    def __len__(self):
        return self.size * self.replicates

