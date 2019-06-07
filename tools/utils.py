import os
import numpy as np
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
    im = img.data.cpu().numpy()[:3, :, :].transpose(1, 2, 0)*255
    out = out.data.cpu().numpy()
    out = np.argmax(out, 0)
    plt.imshow(cv2.cvtColor(im.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.subplot(2, 2, 2)
    fg = img.data.cpu().numpy()[3:, :, :].transpose(1, 2, 0)
    plt.imshow(fg.squeeze())
    plt.subplot(2, 2, 3)
    plt.imshow(gt.squeeze())
    plt.subplot(2, 2, 4)
    plt.imshow(out)
    plt.show()
    plt.pause(0.05)
    plt.clf()

def get_iou(pred, gt, ignore_cls=None):
    if pred.shape != gt.shape:
        print('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)

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

