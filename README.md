# MaskTrack_Box

This is simplified MaskTrack_Box implementation in pytorch.

MaskTrack_Box requires only a bounding box of the target for video object segmentation.
Althohugh original MaskTrack_Box consists of two model(1. model extracting mask from box 2. Mask propagation model),
I simplified the models into one.

- The deformation method is simplified by imgaug library.
- Test code does not include fine-tuning as necessary.

(Perazzi F, Khoreva A, Benenson R, Schiele B, Sorkine-Hornung A. Learning video object segmentation from static images. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2017 (pp. 2663-2672).)



The original base code is borrowed from https://github.com/isht7/pytorch-deeplab-resnet.
