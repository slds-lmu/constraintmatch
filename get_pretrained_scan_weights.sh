#!/bin/bash
# bash script to download model weights from the SCAN author's gdrive acc. to his github repo
# https://github.com/wvangansbeke/Unsupervised-Classification for Cifar10, Cifar100-20 and STL10 
# model weights for the ImageNet10/Dogs datasets are downloaded from our anonymous google drive
# script relies on the pip-installable gdown package

targetDir='experiments/data/model_weights'

# get cifar10 resnet18 weights
gdown https://drive.google.com/uc?id=18gITFzAbQsGS5vt8hyi5HjbeRDsVLihw
# get cifar20 resnet18 weights
gdown https://drive.google.com/uc?id=11mEmpDMyq63pM4kmDy6ItHouI6Q__uB7
# get stl10 resnet18 weights
gdown https://drive.google.com/uc?id=1uNYN9XOMIPb40hmxOzALg4PWhU_xwkEF
# get imagenet10 resnet34 weights
gdown https://drive.google.com/uc?id=1qZI3bqLuReMLvjP-0qPwxDUV984SmcBz
# get imagenetdog resnet34 weights
gdown https://drive.google.com/uc?id=1A3xHbtnQ9s1f_CE3N7c-LakfBbCsAsVn

# create model weights folder and move weights there
mkdir $targetDir

mv selflabel_cifar-10.pth.tar $targetDir
mv selflabel_cifar-20.pth.tar $targetDir
mv selflabel_stl-10.pth.tar $targetDir
mv selflabel_imagenet10.pth.tar $targetDir
mv selflabel_imagenetdog.pth.tar $targetDir