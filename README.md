# Lung Cancer Segmentation

## Overview

This repository contains source codes from my school project for Biomedical Image Segmentation. It will contain only the latest update since I've stored all previous iterations in my private gitlab repository.

## Task

The goal of this project is to successfully segment data from lung CT scans using deep learning networks.

For this I've used U-Net network proposed in [1]

## Data

For the dataset, I chose covid-19 CT scan lesion dataset (2) from Kaggle. This dataset consists of over 2700 lesion images and corresponding masks. Since I wanted to simulate a real situation, I've picked 110 images and their masks to use in this project.

These 110 images come from 5 different patients (22 from each) and resemble different stages of leison development.

## Usage

In order to use the scripts, it is possible to download both input data and model checkpoints from https://drive.google.com/drive/u/1/folders/1Zdb7oIrF6f0K824FFhic5unKZcMuzM0S.

* For data augmentation run `python3 codes/data_aug.py.`
* For model training run `python3 codes/train.py`.
* To generate masks run `python3 codes/test.py`.

## Results

In the results folder there are few segmented images to demonstrate the capability of the model.

-------------

## Resources

[1] Ronneberger, O., Fisher, P., Brox, T.: U-Net: Convolutional Networks for Biomedical Image Segmentation. In: CoRR (2015). DOI: https://doi.org/10.48550/arXiv.1505.04597

[2] COVID-19 CT scan lesion segmentation dataset. https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset

