import scipy.misc
import cv2
import numpy as np
import sys
sys.path.append('..')

import torch
import torch.nn.functional as F
import torch.nn as nn

from models import deeplab_resnet_pair
from collections import OrderedDict
import os
import matplotlib.pyplot as plt

from docopt import docopt
#from dataloader.datasets import DAVIS2016
from tools.utils import *

import json
from collections import OrderedDict

davis_path = '/data/hakjin-workspace/DAVIS/DAVIS-2016/DAVIS'
im_path = os.path.join(davis_path, 'JPEGImages/480p')
gt_path = os.path.join(davis_path, 'Annotations/480p')


docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage: 
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapPrefix=<str>          Snapshot [default: VOC12_scenes_]
    --testGTpath=<str>          Ground truth path prefix [default: data/gt/]
    --testIMpath=<str>          Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --gpu0=<int>                GPU number [default: 0]
"""

args = docopt(docstr, version='v0.1')
#print args
gpu0 = 0

max_label = int(args['--NoLabels'])-1 # labels from 0,1, ... 20(for VOC) 


def test_model(model, vis=False, save=True):
    dim = 328
    model.eval()
    val_seqs = np.loadtxt(os.path.join(davis_path, 'val_seqs.txt'), dtype=str).tolist()
    dumps = OrderedDict()
    #val_seqs = np.loadtxt(os.path.join(davis_path, 'train_seqs.txt'), dtype=str).tolist()


    #hist = np.zeros((max_label+1, max_label+1))
    pytorch_list = []
    tiou = 0
    for seq in val_seqs:
        seq_path = os.path.join(im_path, seq)
        img_list = [os.path.join(seq, i)[:-4] for i in sorted(os.listdir(seq_path))]

        seq_iou = 0
        for idx, i in enumerate(img_list):

            img_original = cv2.imread(os.path.join(im_path,i+'.jpg'))
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            img_temp = img_original.copy().astype(float)/255.
            oh, ow, _ = img_temp.shape
            
            gt_original = cv2.imread(os.path.join(gt_path,i+'.png'),0)
            gt_original[gt_original==255] = 1   

            if idx == 0:
                bb = cv2.boundingRect(gt_original)
                template = crop_and_padding(img_temp, gt_original, (dim, dim))
                fg = crop_and_padding(gt_original, gt_original, (dim, dim))
                bb = cv2.boundingRect(fg)
                box = np.zeros([dim, dim, 1])
                if bb is not None:
                    box[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1, 0] = 1
                    #fg[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1, 0] = 100 / 255.
                    #fg[fg==0] = -100/255.
                    template = np.expand_dims(template, 0).transpose(0,3,1,2)
                    template = torch.FloatTensor(template).cuda(gpu0)
                    box = np.expand_dims(box, 0).transpose(0,3,1,2)
                    box = torch.FloatTensor(box).cuda(gpu0)
                previous = gt_original.copy()
                bb = cv2.boundingRect(previous)
                previous = np.zeros(gt_original.shape).astype('uint8')
                previous[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1]= 1

            #search_region = crop_and_padding(img_temp, previous, (dim, dim))
            search_region = img_temp.copy()
            #mask = crop_and_padding(previous, previous, (dim, dim))
            mask = previous.copy()
            #inp = np.dstack([search_region, mask])
            image = torch.FloatTensor(np.expand_dims(search_region,0).transpose(0,3,1,2)).cuda()
            mask = torch.FloatTensor(mask[np.newaxis, :, :, np.newaxis].transpose(0,3,1,2)).cuda()


            #output = model(template, torch.FloatTensor(np.expand_dims(inp, 0).transpose(0,3,1,2)).cuda(gpu0))
            output = model(image, mask, template, box)
            #pred_c = output.data.cpu().numpy()
            #pred_c = output.data.cpu().numpy()
            #pred_c[pred_c>0.5] = 1
            #pred_c[pred_c!=1] = 0
            #pred_c = scipy.misc.imresize(pred_c.astype('uint8'), (321, 321))
            #pred_c = scipy.misc.imresize(pred_c.astype('uint8').squeeze(), (321, 321))
            #print(output.shape)
            #pred_c = F.upsample(output, scale_factor=2).data.cpu().numpy()
            pred_c = F.interpolate(output, size=(oh,ow), mode='bilinear').data.cpu().numpy()
            pred_c = pred_c.squeeze(0).transpose(1,2,0)
            pred = np.argmax(pred_c,axis = 2)

            #bb = list(cv2.boundingRect(previous.astype('uint8')))
            #bb = list(cv2.boundingRect(gt_original.astype('uint8')))
            #pred = crop_and_padding(gt_original, previous, (dim, dim))
            #pred = crop_and_padding(gt_original, gt_original, (dim, dim))
            #pred = restore_mask(pred_c, bb, img_original.shape)
            #pred = np.argmax(pred, axis=2).astype('uint8')
            #pred[pred>0.5]=1
            #pred[pred!=1]=0
                        
            plt.ion()
            if vis:
                plt.subplot(2, 2, 1)
                plt.imshow(img_original)
                plt.subplot(2, 2, 2)
                plt.title('previous mask')
                plt.imshow(previous)
                plt.subplot(2, 2, 3)
                plt.title('gt - pred')
                bg = np.zeros(img_original.shape)
                bg[:, :, 0] = gt_original*255
                bg[:,:, 1 ] = pred*255

                plt.imshow(bg.astype('uint8'))
                plt.subplot(2, 2, 4)
                output = output.data.cpu().numpy().squeeze()
                output = np.argmax(output, 0)
                plt.imshow(output.astype('uint8'))
                #plt.subplot(2, 2, 4)
                #plt.title('prediction')
                #plt.imshow(pred)
                plt.show()
                plt.pause(.001)
                plt.clf()
            
            previous = pred

            
            iou = get_iou(previous, gt_original.squeeze(), 0)

            if save:
                save_path = 'Result_tempmod'
                folder = os.path.join(save_path, i.split('/')[0])
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                #Image.fromarray(output.astype(int)).save(os.path.join('Results', i+'.png'))
                scipy.misc.imsave(os.path.join(save_path, i+'.png'),previous*255)
            seq_iou += iou

        print(seq, seq_iou/len(img_list))
        dumps[seq] = seq_iou/len(img_list)
        tiou += seq_iou/len(img_list)
    #miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    #print 'pytorch',iter,"Mean iou = ",np.sum(miou)/len(miou)
    print('total:', tiou/len(val_seqs))
    model.train()
    dumps['t mIoU'] = tiou/len(val_seqs)
    with open('dump_tempmod.json', 'w') as f:
        json.dump(dumps, f, indent=2)

    return tiou/len(val_seqs)

if __name__ == '__main__':
    #model = deeplab_resnet.Res_Deeplab_4chan(2)
    model = deeplab_resnet_pair.Res_Deeplab_4chan(2)
    #model = deeplab_resnet.Deep_EncoderDecoder(2)
    #state_dict = torch.load('data/snapshots/DAVIS16-20000.pth')
    #state_dict = torch.load('../data/snapshots/box-24000.pth')
    #model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    res = test_model(model, vis=True)
    print(res)
