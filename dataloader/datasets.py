import torch
import torch.utils.data as data

import os, math, random
from os.path import join
import numpy as np

import cv2
from .custom_transforms import aug_batch
from PIL import Image
import matplotlib.pyplot as plt


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class DAVIS(data.Dataset):
    def __init__(self, inference_size=[-1, -1], train=False, is_cropped = False, root = '', replicates = 1, aug=False):
        self.is_cropped = is_cropped
        self.render_size = inference_size
        self.replicates = replicates
        self.aug = aug

        image_root = join(root, 'JPEGImages/480p')
        gt_root = join(root, 'Annotations/480p')
        if train:
            seqs_file = 'train_seqs.txt'
        else:
            seqs_file = 'val_seqs.txt'
        seq_list = sorted(np.loadtxt(join(root, seqs_file), dtype=str).tolist())
        #seq_list = sorted(os.listdir(image_root))

        self.image_list = []
        self.gt_list = []
        for seq in seq_list:
            files = sorted(os.listdir(join(image_root, seq)))
            for i in range(len(files)):
                img = join(image_root, seq, files[i])
                gt = join(gt_root, seq, files[i][:-4]+'.png')
                self.image_list += [img]
                self.gt_list += [gt]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

       
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
            
        #if self.is_cropped:
        #    cropper = StaticRandomCrop(image_size, self.crop_size)
        #else:
        #    cropper = Statimreadize)
        #images = list(map(imread

        #img = cropper(img)imread
        #gt = cropper(gt)

        #img = np.array(img).transpose(2,0,1)
        #gt = gt.transpose(2,0,1)
        #gt = np.squeeze(gt)
        img = img.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))



        return img, gt

    def __len__(self):
        return self.size * self.replicates

class DAVIS2016(DAVIS):
    def __init__(self, inference_size=[-1, -1], train=False, is_cropped = False, root = '', replicates = 1, aug=False):
        super(DAVIS2016, self).__init__(inference_size=inference_size, train=train, is_cropped = is_cropped, root = root, replicates = replicates, aug=aug)

class YTB_VOS(data.Dataset):
    def __init__(self, inference_size=[-1, -1], train=False, is_cropped = False, root = '', replicates = 1, aug=False):
        self.is_cropped = is_cropped
        self.render_size = inference_size
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

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

       
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

        #if self.is_cropped:
        #    cropper = StaticRandomCrop(image_size, self.crop_size)
        #else:
        #    cropper = StaticCenterCrop(image_size, self.render_size)
        #images = list(map(cropper, images))

        #img = cropper(img)
        #gt = cropper(gt)

        #img = np.array(img).transpose(2,0,1)
        #gt = gt.transpose(2,0,1)
        #gt = np.squeeze(gt)
        img = img.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))



        return img, gt

    def __len__(self):
        return self.size * self.replicates

class ECSSD(data.Dataset):
    def __init__(self, inference_size=[-1, -1], is_cropped = False, root = '', replicates = 1, aug=False):
        self.is_cropped = is_cropped
        self.render_size = inference_size
        self.replicates = replicates
        self.aug = aug

        image_root = join(root, 'images')
        gt_root = join(root, 'ground_truth_mask')
        #if train:
        #    seqs_file = 'train_seqs.txt'
        #else:
        #    seqs_file = 'val_seqs.txt'
        #seq_list = sorted(np.loadtxt(join(root, seqs_file), dtype=str).tolist())

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

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

       
        assert (len(self.image_list) == len(self.gt_list))

    def __getitem__(self, index):

        index = index % self.size
        img = cv2.imread(self.image_list[index])
        

        gt = np.expand_dims(np.array(Image.open(self.gt_list[index])), axis=2)
        gt[gt==255] = 1


        image_size = img.shape[:2]

        if self.aug:
            img, gt = aug_batch(img, gt)
            
        #if self.is_cropped:
        #    cropper = StaticRandomCrop(image_size, self.crop_size)
        #else:
        #    cropper = StaticCenterCrop(image_size, self.render_size)
        #images = list(map(cropper, images))

        #img = cropper(img)
        #gt = cropper(gt)

        #img = np.array(img).transpose(2,0,1)
        #gt = gt.transpose(2,0,1)
        #gt = np.squeeze(gt)
        img = img.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))



        return img, gt

    def __len__(self):
        return self.size * self.replicates

class MSRA10K(data.Dataset):
    def __init__(self, inference_size=[-1, -1], is_cropped = False, root = '', replicates = 1, aug=False):
        self.is_cropped = is_cropped
        self.render_size = inference_size
        self.replicates = replicates
        self.aug = aug

        image_root = join(root, 'images')
        gt_root = join(root, 'annotations')
        #if train:
        #    seqs_file = 'train_seqs.txt'
        #else:
        #    seqs_file = 'val_seqs.txt'
        #seq_list = sorted(np.loadtxt(join(root, seqs_file), dtype=str).tolist())
        #seq_list = sorted(os.listdir(image_root))
        
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

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

       
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
            
        #if self.is_cropped:
        #    cropper = StaticRandomCrop(image_size, self.crop_size)
        #else:
        #    cropper = StaticCenterCrop(image_size, self.render_size)
        #images = list(map(cropper, images))

        #img = cropper(img)
        #gt = cropper(gt)

        #img = np.array(img).transpose(2,0,1)
        #gt = gt.transpose(2,0,1)
        #gt = np.squeeze(gt)
        img = img.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))



        return img, gt

    def __len__(self):
        return self.size * self.replicates

