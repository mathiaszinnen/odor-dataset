# The Object Detection for Olfactory References (ODOR) Dataset

This repository contains the ODOR dataset ([zenodo](https://zenodo.org/record/7732038)) and the code to reproduce the benchmark methods  reported in [1].

## Introduction
Real-world applications of computer vision in the humanities require algorithms to be robust against artistic abstraction, peripheral objects, and subtle differences between fine-grained target classes. Existing datasets provide instance-level
annotations on artworks but are generally biased towards the image centre and limited with regard to detailed object classes. The proposed ODOR dataset fills this gap, offering 38,116 object-level annotations across 4,712 images, spanning an extensive set of 139 fine-grained categories. Conducting a statistical analysis, we showcase challenging dataset properties, such as a detailed set of categories, dense and overlapping objects, and spatial distribution over the whole image canvas. Furthermore, we provide an extensive baseline analysis for object detection models and highlight the challenging properties of the dataset through a set of secondary studies. Inspiring further research on artwork object detection and broader visual cultural heritage studies, the dataset challenges researchers to explore the intersection of object recognition and smell perception.

## Preparation
To download the dataset images, run the `download_imgs.py` script in the [data](data) subfolder. The images will be downloaded to `data/imgs`.

## How to Use The Dataset
The annotations are provided in COCO JSON format. To represent the two-level hierarchy of the object classes, we make use of the supercategory field in the categories array as defined by COCO. In addition to the object-level annotations, we provide an additional CSV file with image-level metadata, which includes content-related fields, such as Iconclass codes or image descriptions, as well as formal annotations, such as artist, license, or creation year. For the sake of license compliance, we do not publish the images directly (although most of the images are public domain). Instead, we provide links to their source  collections in the metadata file (meta.csv) and a python script to download the artwork images (download_images.py).

## How to Train Benchmark Methods
Multiple frameworks were used for the training of the different methods, for instructions on how to reproduce training please refer to the respective subfolder:

| Method | Subfolder/Framework |
| --- | --- |
| DINO(FocalNet) | [detrex](detrex) |
| DINO(SWIN-L) | [mmdetection](mmdetection) |
| Faster R-CNN | [mmdetection](mmdetection) |
| YOLO-v8 | [ultralytics](ultralytics) |

## Installation
For installation instructions for the benchmark methods, please refer to the original framework repository (see subfolders).

## References 
[1] Mathias Zinnen, Prathmesh Madhu, Inger Leemans, Peter Bell, Azhar Hussian, Hang Tran, Ali Hürriyetoğlu, Andreas Maier, Vincent Christlein, Smelly, dense, and spreaded: The Object Detection for Olfactory References (ODOR) dataset, Expert Systems with Applications, Volume 255, Part B, 2024, 124576, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2024.124576. 
[arXiv](https://arxiv.org/abs/2507.08384)
