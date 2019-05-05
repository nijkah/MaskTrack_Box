import os
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2

overlay_color = [255, 0, 0]
transparency = 0.6
result_path = './DAVIS-template-'


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


def validate(name, train=False):
    seg_path = result_path+name
    davis_path = '/home/hakjine/datasets/DAVIS/DAVIS-2016/DAVIS/Annotations/480p'
    vot_gt_path = '/home/hakjine/datasets/VOT/vot2016/Annotations'
    path = davis_path
    zf = 0
    if path is davis_path:
        zf = 9
    else:
        zf = 12

    seqs = sorted(np.loadtxt(os.path.join(davis_path, 'val_seqs.txt')).tolist())
    if train:
        seqs = sorted(np.loadtxt(os.path.join(davis_path, 'train_seqs.txt')).tolist())

    import json

    miou = 0
    result = OrderedDict()
    for seq in seqs:

        imgs = sorted(os.listdir(os.path.join(seg_path, seq)))
        mmiou = 0
        for frame in imgs:
            img = cv2.imread(os.path.join(seg_path, seq, frame))
            img = np.asarray(Image.open(os.path.join(seg_path, seq, frame)))
            gt = np.asarray(Image.open(os.path.join(path, seq, frame)))
            #img = cv2.resize(img, (gt.shape[1], gt.shape[0]))
            img = 1.*img/np.max(img)
            img[img > 0.6] = 255
            img[img != 255] = 0
            iou = get_iou(img, gt, 0)
            mmiou += iou
        mmiou = mmiou / len(imgs)
        result[seq] = mmiou
        miou += mmiou
    #    print(seq, " miou: ", mmiou)
    print("Total mIoU:", miou/len(seqs))
    with open('results/davis_result'+name+'.txt', 'w') as file:
        result['t mIoU'] = miou/len(seqs)
        json.dump(result, file, indent=2)
    return miou/len(seqs)

if __name__ == '__main__':
    validate(name='adam')

