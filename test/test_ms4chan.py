
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')


from models import deeplab_resnet 
from collections import OrderedDict
from docopt import docopt
from tools.utils import get_iou
from datasets import DAVIS2016
import json
from finetune import *

davis_path = '/home/hakjine/datasets/DAVIS/DAVIS-2016/DAVIS'
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


def test(model, vis=False, save=True):
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
            img_temp = img_original.copy().astype(float)
            img_temp[:,:,2] -= 104.00699
            img_temp[:,:,1] -= 116.66877
            img_temp[:,:,0] -= 122.67892
            oh, ow, oc = img_original.shape
            img_temp= cv2.resize(img_temp, (321, 321))
            h, w, c = img_temp.shape
            img_temp = img_temp.astype(float)

            gt_original = cv2.imread(os.path.join(gt_path,i+'.png'),0)
            gt_original[gt_original==255] = 1   
            gt = cv2.resize(gt_original, (w, h), interpolation=cv2.INTER_NEAREST)
            if idx == 0:
                #model = finetune(model, img_original, gt_original)

                fc = np.ones([h, w, 1], dtype=float) *-100
                bb = cv2.boundingRect(gt.astype('uint8'))
                fc[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], 0] = 100
                #fc[np.where(gt==1)] = 100
                #fc += np.expand_dims(gt, 2)
                #fc[fc==1] = 100
            img = np.dstack([img_temp, fc])



            output = model(torch.FloatTensor(np.expand_dims(img, 0).transpose(0,3,1,2)).cuda(gpu0))
            interp = nn.UpsamplingBilinear2d(size=(h, w))
            #output = output[3][0].data.cpu().numpy()
            output = interp(output).data.cpu().numpy().squeeze()
            #output = output[:,:img_temp.shape[0],:img_temp.shape[1]]
            
            output = output.transpose(1,2,0)
            output = np.argmax(output,axis = 2)

            #upsampled_output = cv2.resize(np.expand_dims(output,2), (ow, oh))
            upsampled_output = cv2.resize(output.astype(np.float32), (ow, oh))
            upsampled_output[upsampled_output > 0.5] = 1
            upsampled_output[upsampled_output != 1] = 0

            iou = get_iou(upsampled_output, gt_original, 0)

            plt.ion()
            if vis:
                plt.subplot(2, 2, 1)
                plt.imshow(img_original)
                plt.subplot(2, 2, 2)
                plt.imshow(fc.squeeze())
                plt.subplot(2, 2, 3)
                #plt.imshow(gt)
                plt.imshow(gt_original)
                plt.subplot(2, 2, 4)
                #plt.imshow(output)
                plt.imshow(upsampled_output)
                plt.show()
                plt.pause(0.01)
                plt.clf()

            fc = np.ones([h, w, 1], dtype=float) * -100
            bb = cv2.boundingRect(output.astype('uint8'))
            if bb[2] != 0 and bb[3] != 0:
                #fc[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], 0] = 100
                #erode = cv2.erode(output, kernel, iterations=3)
                fc[np.where(output==1)] = 100

            if save:
                save_path = 'Result_masktrack_ed_saliency'
                folder = os.path.join(save_path, i.split('/')[0])
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                #Image.fromarray(output.astype(int)).save(os.path.join('Results', i+'.png'))
                cv2.imwrite(os.path.join(save_path, i+'.png'),output)
            seq_iou += iou

        print(seq, seq_iou/len(img_list))
        dumps[seq] = seq_iou/len(img_list)
        tiou += seq_iou/len(img_list)
    #miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    #print 'pytorch',iter,"Mean iou = ",np.sum(miou)/len(miou)
    model.train()
    dumps['t mIoU'] = tiou/len(val_seqs)
    with open('dump_ed_saliency.json', 'w') as f:
        json.dump(dumps, f, indent=2)

    return tiou/len(val_seqs)

if __name__ == '__main__':
    model = deeplab_resnet.Res_Deeplab_4chan(2)
    #model = deeplab_resnet.Deep_EncoderDecoder(2)
    #state_dict = torch.load('data/snapshots/DAVIS16-20000.pth')
    state_dict = torch.load('data/snapshots/masktrack-20000.pth')
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    res = test(model, vis=False)
    print(res)
