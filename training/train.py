import cv2
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import random
from docopt import docopt
import timeit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim

import sys
import os
sys.path.append('..')

from models import deeplab_resnet
from dataloader.datasets import DAVIS2016, YTB_VOS, ECSSD, MSRA10K
from tools.loss import cross_entropy_loss_weighted, cross_entropy_loss
from evaluation.evaluate import test_model

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
davis_path = '/home/hakjinlee/datasets/DAVIS/DAVIS-2016/DAVIS'
davis_im_path = os.path.join(davis_path, 'JPEGImages/480p')
davis_gt_path = os.path.join(davis_path, 'Annotations/480p')
vos_path = '/home/hakjinlee/datasets/Youtube-VOS/train/'
vos_im_path  = os.path.join(vos_path, 'JPEGImages')
vos_gt_path  = os.path.join(vos_path, 'Annotations')
ECSSD_path = '../data/ECSSD'
MSRA10K_path = '../data/MSRA10K'

start = timeit.timeit()
docstr = """Train ResNet-DeepLab on segmentation datasets in pytorch using VOC12 pretrained initialization 

Usage: 
    train.py [options]

Options:
    --NoLabels=<int>            The number of from custom_transforms import aug_batchfferent labels in training data, VOC has 21 labels, including background [default: 2]
    --lr=<float>                Learning Rate from custom_transforms import aug_batch [default: 0.001]
    -b, --batchSize=<int>       Num sample per batch [default: 12]
    --wtDecay=<float>           Weight decay during training [default: 0.0005]
    --gpu=<int>                GPU number [default: 4]
    --maxIter=<int>             Maximum number of iterations [default: 50000]
"""

args = docopt(docstr, version='v0.9')
print(args)

gpu_num = int(args['--gpu'])
num_labels = int(args['--NoLabels'])
torch.cuda.set_device(gpu_num)


def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def vis(img, gt, out):

    plt.ion()
    plt.subplot(2, 2, 1)
    im = img.data.cpu().numpy()[:3, :, :].transpose(1, 2, 0)
    im[:,:,0] += 104.000699
    im[:,:,1] += 116.66877
    im[:,:,2] += 122.67892
    #gt = gt[:,:,:,0]
    #out = out.data.cpu().numpy().transpose(1, 2, 0)
    out = out.data.cpu().numpy()
    out = np.argmax(out, 0)
    #out = 1/ (1+np.exp(-out))
    #print(im.shape)
    plt.imshow(cv2.cvtColor(im.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.subplot(2, 2, 2)
    fg = img.data.cpu().numpy()[3:, :, :].transpose(1, 2, 0)
    plt.imshow(fg.squeeze())
    plt.subplot(2, 2, 3)
    plt.imshow(gt.squeeze())
    #plt.imshow(gt)
    plt.subplot(2, 2, 4)
    plt.imshow(out)
    plt.show()
    plt.pause(0.05)
    plt.clf()

if not os.path.exists('../data/snapshots'):
    os.makedirs('../data/snapshots')


model = deeplab_resnet.Res_Deeplab_4chan(num_labels)

#saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth')
saved_state_dict = torch.load('../data/MS_DeepLab_resnet_trained_VOC.pth')
for i in saved_state_dict:
    i_parts = i.split('.')
    if i_parts[1]=='layer5':
        saved_state_dict[i] = model.state_dict()[i]
    if i_parts[1] == 'conv1':
        saved_state_dict[i] = torch.cat((saved_state_dict[i], torch.FloatTensor(64, 1, 7, 7).normal_(0,0.0001)), 1)

model.load_state_dict(saved_state_dict)


max_iter = int(args['--maxIter']) 
batch_size = int(args['--batchSize'])
weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])

model.float()
model.cuda()
#model.eval() # use_global_stats = True


db_davis_train = DAVIS2016(train=True,root=davis_path, aug=True)
#db_ytb_train = YTB_VOS(train=True, root='/home/hakjine/datasets/Youtube-VOS', aug=True)
db_ECSSD = ECSSD(root=ECSSD_path, aug=True)
db_MSRA10K = MSRA10K(root=MSRA10K_path, aug=True)
db_train = ConcatDataset([db_davis_train, db_ECSSD, db_MSRA10K])

train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True)

optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9,weight_decay = weight_decay)
optimizer.zero_grad()


losses = []
acc = []
numerics = {'loss':losses, 'acc': acc}
import json
iter = 0
for epoch in range(0, 20):
    for ii, sample in enumerate(train_loader):
        iter += 1
        images, label = sample[0], sample[1]
        images = images.cuda()


        out = model(images)
        #loss = loss_calc(out[0], label[0],)
        loss = cross_entropy_loss_weighted(out[3], label.cuda())
        numerics['loss'].append(float(loss.data.cpu().numpy()))
        loss.backward()

        if iter %1 == 0:
            print('iter = ',iter, 'of',max_iter,'completed, loss = ', (loss.data.cpu().numpy()))

    
        #if iter % 10 == 0:
        #    vis(images[0], label[0], out[-1][0])

        optimizer.step()
        lr_ = lr_poly(base_lr,iter,max_iter,0.9)
        print('(poly lr policy) learning rate',lr_)
        optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = weight_decay)
        optimizer.zero_grad()

        if iter % 1000 == 0 and iter!=0:
            print('taking snapshot ...')
            iou = test_ms4chan.test(model, save=True)
            iou =0
            numerics['acc'].append(iou)
            torch.save(model.state_dict(),'../data/snapshots/box-'+str(iter)+'.pth')
            with open('../data/losses/box-'+str(iter)+'.json', 'w') as f:
                json.dump(numerics, f)

        if iter == max_iter:
            break

end = timeit.timeit()
print('time taken ', end-start)
