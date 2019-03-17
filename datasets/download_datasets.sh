#!/bin/bash

wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
unzip DAVIS-data.zip

mkdir ECSSD
cd ECSSD
wget http://www.cse.cuhk.edu.hk/~leojia/projects/hsaliency/data/ECSSD/images.zip
wget http://www.cse.cuhk.edu.hk/~leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip
unzip images.zip
unzip ground_truth_mask.zip

cd ..


