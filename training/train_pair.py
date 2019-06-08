import cv2
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import random
import timeit
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim

import sys
import os
sys.path.append('..')

from models import deeplab_resnet_pair
from dataloader.datasets_pair import DAVIS2016, YTB_VOS, ECSSD_dreaming, MSRA10K_dreaming
from tools.loss import cross_entropy_loss_weighted, cross_entropy_loss
from evaluation.test_pair import test_model
from tools.utils import vis_2

DATASET_PATH = '/home/hakjine/datasets'
DAVIS_PATH = os.path.join(DATASET_PATH, 'DAVIS/DAVIS-2016/DAVIS')
VOS_PATH = os.path.join(DATASET_PATH, 'Youtube-VOS')
#ECSSD_path = '../data/ECSSD'
#MSRA10K_path = '../data/MSRA10K'
ECSSD_PATH = os.path.join(DATASET_PATH, 'ECSSD')
MSRA10K_PATH = os.path.join(DATASET_PATH, 'MSRA10K')
SAVED_DICT_PATH = '../data/MS_DeepLab_resnet_trained_VOC.pth'

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    max_iter = args.maxIter
    batch_size = args.batchSize
    weight_decay = args.wtDecay
    base_lr = args.lr

    start = timeit.timeit()

    model = deeplab_resnet_pair.Res_Deeplab_4chan(2)
    saved_state_dict = torch.load(SAVED_DICT_PATH)

    for i in saved_state_dict:
        i_parts = i.split('.')
        #if i_parts[1]=='layer5':
        #    saved_state_dict[i] = model.state_dict()[i]
        #if i_parts[1] == 'conv1':
        #    saved_state_dict[i] = torch.cat((saved_state_dict[i], torch.FloatTensor(64, 1, 7, 7).normal_(0,0.0001)), 1)
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)

    model.load_state_dict(model_dict)

    """
    saved_state_dict = torch.load('../data/snapshots/box-12000.pth')
    model.load_state_dict(saved_state_dict)
    """
    model.cuda()



    db_davis_train = DAVIS2016(train=True,root=DAVIS_PATH, aug=True)
    db_ytb_train = YTB_VOS(train=True, root=VOS_PATH, aug=True)
    db_ECSSD = ECSSD_dreaming(root=ECSSD_PATH, aug=True)
    db_MSRA10K = MSRA10K_dreaming(root=MSRA10K_PATH, aug=True)
    db_train = ConcatDataset([db_davis_train, db_ytb_train, db_ECSSD, db_MSRA10K])

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True)

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9,weight_decay = weight_decay)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = base_lr, momentum = 0.9,weight_decay = weight_decay)
    optimizer.zero_grad()


    losses = []
    acc = []
    numerics = {'loss':losses, 'acc': acc}
    import json
    iter = 0
    for epoch in range(0, 20):
        for ii, sample in enumerate(train_loader):
            iter += 1
            images, mask, target, box, label = sample[0], sample[1], sample[2], sample[3], sample[4]
            images = images.cuda()
            mask = mask.cuda()
            target = target.cuda()
            box = box.cuda()

            out = model(images, mask, target, box)
            loss = cross_entropy_loss_weighted(out, label.cuda())
            numerics['loss'].append(float(loss.data.cpu().numpy()))
            loss.backward()

            if iter %1 == 0:
                print('iter = ',iter, 'of',max_iter,'completed, loss = ', (loss.data.cpu().numpy()))

        
            #if iter % 5 == 0:
            #    vis_2(images[0], mask[0], target[0], box[0], label[0], out[0])

            optimizer.step()
            lr_ = lr_poly(base_lr,iter,max_iter,0.9)
            print('(poly lr policy) learning rate',lr_)
            optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = weight_decay)
            #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr_, momentum = 0.9,weight_decay = weight_decay)
            optimizer.zero_grad()

            if iter == 10000:
                lr_ *= 10
                base_lr *= 10

            if iter == 20000:
                lr_ *= 50
                base_lr *= 50

            if iter % 1000 == 0 and iter!=0:
                print('taking snapshot ...')
                iou = test_model(model, save=True)
                numerics['acc'].append(iou)
                torch.save(model.state_dict(),'../data/snapshots/temp_mod-'+str(iter)+'.pth')
                with open('../data/losses/temp_mod-'+str(iter)+'.json', 'w') as f:
                    json.dump(numerics, f)

            if iter == max_iter:
                break

    end = timeit.timeit()
    print('time taken ', end-start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet-DeepLab on segmentation datasets in pytorch using VOC12\
        pretrained initialization')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--batchSize', '-b', type=int, default=12, help='Number of samples per batch')
    parser.add_argument('--wtDecay', type=float, default=0.0005, help='Weight decay during training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--maxIter', type=int, default=20000, help='Maximum number of iterations')

    args = parser.parse_args()

    if not os.path.exists('../data/snapshots'):
        os.makedirs('../data/snapshots')
    if not os.path.exists('../data/losses'):
        os.makedirs('../data/losses')

    main(args)

