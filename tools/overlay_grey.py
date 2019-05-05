import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import cv2

# Set User parameters
davis_path = "/home/hakjine/datasets/DAVIS/DAVIS-2016/DAVIS"
segtrack_path = '/home/hakjine/datasets/SegTrackv2/'
seq_name = "scooter-black-chicane"
save_path = "./save/test3"

# Show results
def overlay_seq(result_path, seq_name=seq_name, data_path=davis_path):
    if os.path.isdir(os.path.join(save_path, seq_name)):
        print('exist')
    else:
        os.makedirs(os.path.join(save_path, seq_name))
    overlay_color = [255, 0, 0]
    #gt_color = [255, 0, 30]
    transparency = 0.6
    test_frames = sorted(os.listdir(os.path.join(data_path, 'JPEGImages', '480p', seq_name)))
    plt.ion()
    i = 0
    masks = sorted(os.listdir(os.path.join(result_path)))
    gts = sorted(os.listdir(os.path.join(davis_path, 'Annotations', '480p', seq_name)))
    for i, img_p in enumerate(test_frames):
        frame_num = img_p.split('.')[0]
        img = np.array(Image.open(os.path.join(data_path, 'JPEGImages', '480p', seq_name, img_p)).convert('L'))
        img_new = np.zeros([img.shape[0], img.shape[1], 3])
        for j in range(3):
            img_new[:, :, j] = img
        img = img_new
        #img = cv2.imread(os.path.join(data_path, 'JPEGImages', '480p', seq_name, img_p))
        mask = np.array(Image.open(os.path.join(result_path, masks[i])))
        mask = scipy.misc.imresize(mask, (img.shape[0], img.shape[1]))
        gt = np.array(Image.open(os.path.join(davis_path, 'Annotations/480p', seq_name, gts[i])))

        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
        mask = mask/np.max(mask)
        gt = gt/np.max(gt)
        im_over = np.ndarray(img.shape)
        im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
        im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask* (overlay_color[1]*transparency+ (1-transparency)*img[:, :, 1])

        im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
        #im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*0.9+ (1-0.9)*img[:, :, 2])

        #im_over[:, :, 0] = (1 - gt) * img[:, :, 0] + gt * (gt_color[0]*transparency + (1-transparency)*img[:, :, 0])
        #im_over[:, :, 1] = (1 - gt) * img[:, :, 1] + gt * (gt_color[1]*transparency + (1-transparency)*img[:, :, 1])
        #im_over[:, :, 2] = (1 - gt) * img[:, :, 2] + gt * (gt_color[2]*transparency + (1-transparency)*img[:, :, 2])
        scipy.misc.imsave(os.path.join(save_path,seq_name,frame_num+'.png'), im_over.astype(np.float32))
        """
        plt.imshow(im_over.astype(np.uint8))
        plt.axis('off')
        plt.show()
        plt.pause(0.01)
        plt.clf()
        """
        i = i + 1

if __name__ == "__main__":
    #seqs = os.listdir(os.path.join(segtrack_path, 'JPEGImages/480p'))
    seqs = np.loadtxt(os.path.join(davis_path, 'val_seqs.txt'),dtype='str')
    #seq = 'camel'
    #result_path = os.path.join('./DAVIS/Results/Segmentations/480p/OSVOSbb/test2/'+seq)
    #result_path = os.path.join('./Result_masktrack_training', seq)
    #overlay_seq(result_path, seq, davis_path)
    for seq in seqs:
        result_path = os.path.join('./Result_masktrack_training/'+seq)
        overlay_seq(result_path, seq, davis_path)
