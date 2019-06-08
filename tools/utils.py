import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import cv2

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

def get_10x_lr_params(model, pair=False):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    if pair:
        b.append(model.aspp.parameters())
        b.append(model.branch.parameters())
        b.append(model.fuse.parameters())
        b.append(model.template_refine.parameters())
        b.append(model.template_fuse.parameters())
        b.append(model.refine.parameters())
        b.append(model.predict.parameters())
    else:
        b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def overlay(img, mask, color=[255, 0, 0], transparency=0.6):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_over = np.zeros(img.shape)
    for c in range(3):
        im_over[:, :, c] = (1 - mask) * gray_img + mask * (color[c]*transparency + (1-transparency)*gray_img)

    im_over[im_over>255] = 255
    im_over[im_over<0] =0 

    return im_over

def vis_masktrack(img, gt, out):
    plt.ion()
    plt.subplot(2, 2, 1)
    im = img.data.cpu().numpy()[:3, :, :].transpose(1, 2, 0)*255
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


def vis(img, mask, gt, out, analysis=True):
    plt.ion()
    im = img.data.cpu().numpy().transpose(1, 2, 0)
    im *= 255
    fg = mask.data.cpu().numpy().transpose(1, 2, 0)
    fg[fg>0] = 1
    fg[fg!=1] = 0
    im = overlay(im, fg.squeeze())

    out = out.data.cpu().numpy()
    out = np.argmax(out, 0)
    
    shape = out.shape

    plt.subplot(2, 3, 1).set_title('Search region')
    plt.imshow(im.astype('uint8'))

    gt_pred = np.zeros([shape[0], shape[1], 3])
    gt_pred[:,:,0] = gt.squeeze() * 255
    gt_pred[:,:,2] = out * 255
    #plt.subplot(2, 3, 6).set_title('GT-pred')
    plt.subplot(2, 3, 6)
    plt.text(3, 3, 'GT and Pred', bbox={'facecolor':'white', 'pad':1})
    precision, recall = calculate_precision(out, gt.squeeze()), calculate_recall(out, gt.squeeze())
    plt.xlabel('prec:'+str(precision)+' reca:'+str(recall))
    plt.imshow(gt_pred.astype('uint8'))

    plt.subplot(2, 3, 2).set_title('Ground Truth')
    plt.imshow(gt.squeeze())

    plt.subplot(2, 3, 4).set_title('Prediction')
    plt.imshow(out)

    mask_pred = np.zeros([shape[0], shape[1], 3])
    fg = cv2.resize(fg, (shape[0], shape[1]), cv2.INTER_NEAREST)
    mask_pred[:,:,0] = fg * 255
    mask_pred[:,:,2] = out * 255
    plt.subplot(2, 3, 5).set_title('Mask-pred')
    precision, recall = calculate_precision(out, fg), calculate_recall(out, fg)
    plt.xlabel('prec:'+str(precision)+' reca:'+str(recall))
    plt.imshow(mask_pred.astype('uint8'))


    plt.show()
    plt.pause(0.05)
    plt.clf()

def vis_2(img, mask,target, gt, out, analysis=True):
    plt.ion()
    im = img.data.cpu().numpy().transpose(1, 2, 0)
    im *= 255
    fg = mask.data.cpu().numpy().transpose(1, 2, 0)
    fg[fg>0] = 1
    fg[fg!=1] = 0
    im = overlay(im, fg.squeeze())

    target_im= target.data.cpu().numpy().transpose(1, 2, 0)
    target_im *= 255

    out = out.data.cpu().numpy()
    out = np.argmax(out, 0)
    
    shape = out.shape

    plt.subplot(2, 3, 1).set_title('Template')
    plt.imshow(target_im.astype('uint8'))

    plt.subplot(2, 3, 2).set_title('Search region')
    plt.imshow(im.astype('uint8'))

    gt_pred = np.zeros([shape[0], shape[1], 3])
    gt_pred[:,:,0] = gt.squeeze() * 255
    gt_pred[:,:,2] = out * 255
    #plt.subplot(2, 3, 6).set_title('GT-pred')
    plt.subplot(2, 3, 6)
    plt.text(3, 3, 'GT and Pred', bbox={'facecolor':'white', 'pad':1})
    precision, recall = calculate_precision(out, gt.squeeze()), calculate_recall(out, gt.squeeze())
    plt.xlabel('prec:'+str(precision)+' reca:'+str(recall))
    plt.imshow(gt_pred.astype('uint8'))

    plt.subplot(2, 3, 4).set_title('Ground Truth')
    plt.imshow(gt.squeeze())

    plt.subplot(2, 3, 5).set_title('Prediction')
    plt.imshow(out)

    mask_pred = np.zeros([shape[0], shape[1], 3])
    fg = cv2.resize(fg, (128, 128), cv2.INTER_NEAREST)
    mask_pred[:,:,0] = fg * 255
    mask_pred[:,:,2] = out * 255
    plt.subplot(2, 3, 3).set_title('Mask-pred')
    precision, recall = calculate_precision(out, fg), calculate_recall(out, fg)
    plt.xlabel('prec:'+str(precision)+' reca:'+str(recall))
    plt.imshow(mask_pred.astype('uint8'))


    plt.show()
    plt.pause(0.05)
    plt.clf()


def get_iou(pred, gt, ignore_cls=None):
    if pred.shape != gt.shape:
        print('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)
    #gt = gt.astype(np.float32)
    #pred = pred.astype(np.float32)

    labels = np.unique(gt).tolist()
    if isinstance(ignore_cls, int):
        labels = [label for label in labels if label != ignore_cls]
    if isinstance(ignore_cls, list):
        labels = [label for label in labels if label not in ignore_cls]

    if len(labels) == 0:
        
        if (len(np.unique(pred)) == 1 and np.unique(pred)[0] == ignore_cls):
            return 1
        else:
            return 0

    count = dict()

    for j in labels:
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))

        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / len(u_jj)

    result_class = list(count.values())

    Aiou = np.sum(result_class[:]) / len(labels)

    return Aiou


def get_general_iou(pred, gt):
    if pred.shape != gt.shape:
        print('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    labels = np.unique(gt).tolist()

    count = dict()

    for j in labels:
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))

        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count.values()
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))

    return Aiou

def calculate_precision(pred, gt, save_imgs=0):
    x = np.where(pred == 1)
    pred_idx = set(zip(x[0].tolist(), x[1].tolist()))
    x = np.where(gt == 1)
    gt_idx = set(zip(x[0].tolist(), x[1].tolist()))
    true_positives = set.intersection(pred_idx, gt_idx)

    if len(pred_idx) == 0:
        return 0

    return round(float(len(true_positives)) / len(pred_idx), 3)

def calculate_recall(pred, gt, save_imgs=0):
    x = np.where(pred == 1)
    pred_idx = set(zip(x[0].tolist(), x[1].tolist()))
    x = np.where(gt == 1)
    gt_idx = set(zip(x[0].tolist(), x[1].tolist()))
    true_positives = set.intersection(pred_idx, gt_idx)

    x = np.where(pred == 0)
    pred_idx_neg = set(zip(x[0].tolist(), x[1].tolist()))
    false_negatives = set.intersection(pred_idx_neg, gt_idx)
    if(len(true_positives)+len(false_negatives)) == 0:
        return 0

    return round(float(len(true_positives)) / (len(true_positives)+len(false_negatives)), 3)


