import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import OrderedDict
import json

import torch
import torch.nn.functional as F
import torch.nn as nn

sys.path.append('..')

from models import deeplab_resnet_pair
from tools.utils import *

DATASET_PATH = '/data/hakjin-workspace'
DAVIS_PATH= os.path.join(DATASET_PATH, 'DAVIS/DAVIS-2016/DAVIS')
im_path = os.path.join(DAVIS_PATH, 'JPEGImages/480p')
gt_path = os.path.join(DAVIS_PATH, 'Annotations/480p')

SAVE_PATH = 'template_module'
PRETRAINED_PATH = '../data/snapshots/trained_masktrack_box.pth'

def test_model(model, vis=False, save=True):
    dim = 328
    model.eval()
    val_seqs = np.loadtxt(os.path.join(DAVIS_PATH, 'val_seqs.txt'), dtype=str).tolist()
    dumps = OrderedDict()

    tiou = 0
    for seq in val_seqs:
        seq_path = os.path.join(im_path, seq)
        img_list = [os.path.join(seq, i)[:-4] for i in sorted(os.listdir(seq_path))]

        seq_iou = 0
        for idx, i in enumerate(img_list):

            img_original = cv2.imread(os.path.join(im_path,i+'.jpg'))
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            oh, ow, _ = img_original.shape
            img_temp = img_original.copy().astype(float)/255.
            
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
                    template = np.expand_dims(template, 0).transpose(0,3,1,2)
                    template = torch.FloatTensor(template).cuda()
                    box = np.expand_dims(box, 0).transpose(0,3,1,2)
                    box = torch.FloatTensor(box).cuda()
                previous = gt_original.copy()
                bb = cv2.boundingRect(previous)
                previous = np.zeros(gt_original.shape).astype('uint8')
                previous[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1]= 1

            #search_region = crop_and_padding(img_temp, previous, (dim, dim))
            #mask = crop_and_padding(previous, previous, (dim, dim))
            search_region = img_temp.copy()
            mask = previous.copy()
            search_region = cv2.resize(search_region, (dim, dim))
            mask = cv2.resize(mask, (dim, dim), cv2.INTER_NEAREST)
            image = torch.FloatTensor(np.expand_dims(search_region,0).transpose(0,3,1,2)).cuda()
            mask = torch.FloatTensor(mask[np.newaxis, :, :, np.newaxis].transpose(0,3,1,2)).cuda()


            output = model(image, mask, template, box)
            pred_c = F.interpolate(output, size=(oh,ow), mode='bilinear').data.cpu().numpy()
            pred_c = pred_c.squeeze(0).transpose(1,2,0)

            pred = np.argmax(pred_c, axis=2).astype('uint8')
                        
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
                save_path = os.path.join('../data/save', 'Result_'+SAVE_PATH)
                folder = os.path.join(save_path, i.split('/')[0])
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                cv2.imwrite(os.path.join(save_path, i+'.png'),output*255)
            seq_iou += iou

        print(seq, seq_iou/len(img_list))
        dumps[seq] = seq_iou/len(img_list)
        tiou += seq_iou/len(img_list)

    print('total:', tiou/len(val_seqs))
    model.train()
    dumps['t mIoU'] = tiou/len(val_seqs)
    with open('dump_tempmod.json', 'w') as f:
        json.dump(dumps, f, indent=2)

    return tiou/len(val_seqs)

if __name__ == '__main__':
    num_gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(num_gpu)

    model = deeplab_resnet_pair.Res_Deeplab_4chan(2)
    #state_dict = torch.load(PRETRAINED_PATH)
    #model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    res = test_model(model, vis=True)
