# The Object Detection for Olfactory References (ODOR) Dataset

This repository contains the ODOR dataset ([zenodo](https://zenodo.org/doi/10.5281/zenodo.6362951)) and the code to reproduce the benchmark methods  reported in [1].

## Introduction
Real-world applications of computer vision in the humanities require algorithms to be robust against artistic abstraction, peripheral objects, and subtle differences between fine-grained target classes. Existing datasets provide instance-level
annotations on artworks but are generally biased towards the image centre and limited with regard to detailed object classes. The proposed ODOR dataset fills this gap, offering 38,116 object-level annotations across 4,712 images, spanning an extensive set of 139 fine-grained categories. Conducting a statistical analysis, we showcase challenging dataset properties, such as a detailed set of categories, dense and overlapping objects, and spatial distribution over the whole image canvas. Furthermore, we provide an extensive baseline analysis for object detection models and highlight the challenging properties of the dataset through a set of secondary studies. Inspiring further research on artwork object detection and broader visual cultural heritage studies, the dataset challenges researchers to explore the intersection of object recognition and smell perception.

## Preparation
To download and prepare the dataset annotations and images, run the `prepare_dataset.py` script in the project root. The dataset will be downloaded to `data/`.

## How to Use The Dataset
The annotations are provided in COCO JSON format. To represent the two-level hierarchy of the object classes, we make use of the supercategory field in the categories array as defined by COCO. In addition to the object-level annotations, we provide an additional CSV file with image-level metadata, which includes content-related fields, such as Iconclass codes or image descriptions, as well as formal annotations, such as artist, license, or creation year. In addition to the images downloadable via zenodo, we provide links to their source collections in the metadata file (meta.csv) and a python script to download the artwork images (download_images.py).

## How to Train Benchmark Methods
For instructions on how to use the baseline models reported in [1], please refer to the respective subdirectories. 
The Faster R-CNN models were trained using [mmdetection](mmdetection), the DINO models via [detrex](detrex), and the YOLO using [ultralytics](ultralytics). 

| Method | Subfolder/Framework |
| --- | --- |
| DINO | [detrex](detrex) |
| Faster R-CNN | [mmdetection](mmdetection) |
| YOLO-v8 | [ultralytics](ultralytics) |
| Mutual-Assistance Learning | [MADet](MADet) |

## Installation
For installation instructions for the benchmark methods, please refer to the original framework repository (see subfolders).

## References 
[1] 