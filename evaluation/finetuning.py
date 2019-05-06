import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('..')

from collections import OrderedDict
from tools.loss import cross_entropy_loss_weighted
import json

def outS(i):
    j = int(i)
    j = (j+1)/2
    j = int(np.ceil((j+1)/2.))
    j = (j+1)/2
    return j

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1, 1))
    interp = nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = torch.FloatTensor(label[np.newaxis, np.newaxis, :, :])
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)
    label_resized[label_resized>0.3]  = 1
    label_resized[label_resized != 0]  = 1

    return label_resized


base_lr = 0.0001
weight_decay = 0.00005
def finetune(model, image, gt):
    model.train()
    #optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr},
    #                       {'params': get_10x_lr_params(model), 'lr': 10 * base_lr}], lr=base_lr, momentum=0.9,
    #                      weight_decay=weight_decay)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001, momentum = 0.9, weight_decay = 0.0005)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.00001, momentum = 0.9,
    #    weight_decay = 0.000005)
    optimizer.zero_grad()

    for i in range(200):
        flag = True
        for j in range(10):
            if flag:
                img_temp, gt_temp = aug_batch(image, gt)
                img_temp = np.expand_dims(img_temp, 0)
                gt_temp = np.expand_dims(gt_temp, 0)
                flag = False
            else:
                img_ttemp, gt_ttemp = aug_batch(image, gt)
                img_ttemp = np.expand_dims(img_ttemp, 0)
                gt_ttemp = np.expand_dims(gt_ttemp, 0)
                #print(img_temp.shape, img_ttemp.shape)
                img_temp = np.concatenate([img_temp, img_ttemp], 0)
                gt_temp = np.concatenate([gt_temp, gt_ttemp], 0)

        inp = torch.FloatTensor(img_temp.transpose(0,3,1,2)).cuda()
        out = model(inp)


        loss = cross_entropy_loss_weighted(out[3], torch.FloatTensor(gt_temp).cuda())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    model.eval()

    return model


def finetune_naive(model, image, gt):
    model.train()
    #optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr},
    #                       {'params': get_10x_lr_params(model), 'lr': 10 * base_lr}], lr=base_lr, momentum=0.9,
    #                      weight_decay=weight_decay)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001, momentum = 0.9, weight_decay = 0.0005)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.00001, momentum = 0.9,
    #    weight_decay = 0.000005)
    optimizer.zero_grad()

    for i in range(200):

        inp = torch.FloatTensor(np.expand_dims(image, 0).transpose(0,3,1,2)).cuda()
        out = model(inp)


        gta = resize_label_batch(gt, outS(321))
        loss = cross_entropy_loss_weighted(out[3], torch.FloatTensor(gta).cuda())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    model.eval()

    return model
