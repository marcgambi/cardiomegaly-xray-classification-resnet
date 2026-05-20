# cardiomegaly-xray-classification-resnet
Deep learning pipeline for binary classification of cardiomegaly from chest X-ray images using image preprocessing, data augmentation and ResNet-based CNN models in PyTorch.

# Binary Classification of Cardiomegaly using Deep Learning

This project focuses on the binary classification of cardiomegaly from chest X-ray images using a deep learning pipeline developed in Python.

The workflow includes dataset analysis, image preprocessing, data augmentation, ResNet-based convolutional neural network training, and model evaluation using standard classification metrics.

## Project Overview

The dataset included more than 50,000 chest X-ray images. The objective was to classify images as either showing cardiomegaly or no cardiomegaly.

The project involved:
- dataset label analysis
- class distribution evaluation
- image size and histogram analysis
- image preprocessing and enhancement
- data augmentation
- ResNet-based binary classification
- training and validation with early stopping
- test evaluation using accuracy, precision, recall, F1-score, confusion matrix, ROC curve and precision-recall curve

## Technologies

- Python
- PyTorch
- Torchvision
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Main Features

### Dataset Analysis
Exploratory analysis was performed to study the distribution of cardiomegaly labels and patient-related metadata.

### Image Preprocessing
Several preprocessing techniques were applied, including resizing, grayscale conversion, normalization, histogram equalization, CLAHE, gamma correction and image filtering.

### Deep Learning Model
A ResNet-based convolutional neural network was implemented for binary classification.

### Training Pipeline
The model was trained using PyTorch with binary cross-entropy loss, Adam optimizer, learning rate scheduling and early stopping.

### Evaluation
Model performance was evaluated using classification metrics and diagnostic plots, including confusion matrix, ROC curve and precision-recall curve.

## Skills Developed

- Biomedical image analysis
- Deep learning for medical imaging
- Chest X-ray classification
- Computer vision preprocessing
- PyTorch model training
- Evaluation of binary classifiers
- Data analysis and visualization

## Notes

This project was developed for educational and research purposes in the biomedical engineering and medical imaging field.

## Dataset

The original chest X-ray images and related clinical metadata are not included in this repository due to privacy and institutional restrictions.
