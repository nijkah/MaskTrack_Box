# MaskTrack_Box

This is simplified MaskTrack_Box implementation in pytorch.

MaskTrack_Box requires only a bounding box of the target for video object segmentation.
Althohugh original MaskTrack_Box consists of two model(1. model extracting mask from box 2. Mask propagation model),
I simplified the models into one.

- The deformation method is simplified by imgaug library.
- Test code does not include fine-tuning as necessary.

## Environment setup
All the code has been tested on Ubuntu 16.04, python3.6, Pytorch0.4.1, CUDA 9.0, GTX TITAN x GPU

- Clone the repository
```
git clone https://github.com/nijkah/masktrack_box.git && cd masktrack_box
```

- Setup python environment
```
conda create -n masktrack_box python=3.5
source activate siammask
conda install pytorch-0.4.1 cu90 -c pytorch
pip install -r requirments.txt
```

- Download data
```
cd data
sh download_datasets.sh
cd ..
```



















## Citations
The original paper is
```
@inproceedings{Perazzi2017,
  author={F. Perazzi and A. Khoreva and R. Benenson and B. Schiele and A.Sorkine-Hornung},
  title={Learning Video Object Segmentation from Static Images},
  booktitle = {Computer Vision and Pattern Recognition},
  year = {2017}
}
```

The original base code is borrowed from https://github.com/isht7/pytorch-deeplab-resnet.
