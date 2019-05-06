# MaskTrack_Box

This is simplified [MaskTrack_Box](https://arxiv.org/abs/1612.02646) implementation in pytorch.

Compared to conventional semi-supervised video object segmentation methods,
MaskTrack_Box requires only a bounding box of the target for video object segmentation.

Althohugh original MaskTrack_Box consists of two models(1. Model extracting a mask from the box 2. Mask propagation model),
I simplified these models into one.

### Main differences between this project and original paper
- Model is just one, whereas Masktrack_box in the paper consists of two models.
- The deformation method is simplified by imgaug library, running on-the-fly.

* I added fine-tuning code, but this is not necessary to evaluate.

## Environment setup
All the code has been tested on Ubuntu 16.04, python3.6, Pytorch0.4.1, CUDA 9.0, GTX TITAN x GPU

- Clone the repository
```
git clone https://github.com/nijkah/masktrack_box.git && cd masktrack_box
```

- Setup python environment
```
conda create -n masktrack_box python=3.6
source activate masktrack_box 
conda install pytorch=0.4.1 cu90 -c pytorch
pip install -r requirments.txt
```

- Download data
```
cd data
sh download_datasets.sh
cd ..
```
and you can download the pre-trained deeplab model from
[here](https://drive.google.com/file/d/0BxhUwxvLPO7TeXFNQ3YzcGI4Rjg/view).
Put this in the 'data' folder.

- train the model
```
cd train
python train.py
```

- evaluate the model
```
cd test
python test.py
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

## Acknowledgement
The original base code is borrowed from
[https://github.com/isht7/pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet).

This project is inspired by [https://github.com/omkar13/MaskTrack](https://github.com/omkar13/MaskTrack).
