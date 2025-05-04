check the new files for updated ones

# Deepfake Detection Repository

This repository contains various scripts and tools for deepfake detection, specifically using facial features, audio features, and multi-modal data. It utilizes models like EfficientNet-B7, MediaPipe Face Mesh, and UNet to process and classify deepfake videos and images.

## Table of Contents

- [Introduction](#introduction)
- [Scripts](#scripts)
  - [Dataset Extraction](#dataset-extraction)
  - [Feature Extraction](#feature-extraction)
  - [Model Definition](#model-definition)
  - [Deepfake Generation](#deepfake-generation)
  - [Data Processing](#data-processing)
  - [Performance Evaluation](#performance-evaluation)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)

## Introduction

This repository provides a collection of scripts for detecting deepfakes by leveraging facial region analysis, audio processing, and state-of-the-art neural network models. It includes pre-processing, augmentation, and feature extraction, all integrated into a structured pipeline for deepfake detection.

## Scripts

### Dataset Extraction

1. **`celeb_extract.py`**: Extracts frames from videos in the Celeb-DF dataset, detects faces using MTCNN, and augments them using Albumentations. It organizes the processed data into structured train and validation sets for deepfake detection.
2. **`celeb_sys.py`**: Extracts frames from `.mp4` videos in the Celeb-Synthesis dataset and saves them into a structured output folder for deepfake detection tasks.
3. **`celebe_1.py`**: Resumes processing extracted video frames by detecting faces, applying data augmentation in batches, and organizing them into train and validation folders for the Celeb-real dataset, skipping already processed images.

### Feature Extraction

4. **`datainject_ff.py`**: Defines a custom PyTorch Dataset class (`HybridDataset`) and a `get_dataloaders` function to load multi-modal data for deepfake detection. It works with original face images, heatmaps, ear regions, optical flow, and segmented facial features.
5. **`ear_lips_ff.py`**: Creates grayscale "ear" images and blurred "optical flow" approximations from the dataset.
6. **`extract_again.py`**: Extracts facial region features using MediaPipe Face Mesh.
7. **`extract_all_1.py`**: Extracts specific facial features (eyes, lips, nose) from images using MediaPipe Face Mesh and creates binary mask images.
8. **`extract_features_2.py`**: Detects a face in each image, extracts specific facial regions (eyes, nose, lips), crops those regions, and saves them as separate image files.
9. **`extract_frames_valid.py`**: Prepares an augmented face dataset for deep learning models from a folder of raw face images.
10. **`ff.py`**: Converts deepfake videos into individual frames and organizes them into a train/validation dataset.

### Deepfake Generation

11. **`gan_deepfake_generate.py`**: Adds newly generated deepfake images into an existing dataset, keeping the data in a consistent JSON format.
12. **`integrate_gan_deepfake.py`**: Merges newly generated deepfakes into the main dataset for training or evaluation.

### Data Processing

13. **`mediapipe_ff.py`**: Generates landmark heatmaps for each face in the dataset, highlighting the 68 facial landmark points for deepfake detection.
14. **`mediapipe_segment.py`**: Processes images to extract facial features using MediaPipe's Face Mesh model.
15. **`model_effii_ff.py`**: Defines a hybrid neural network model combining CNNs and Vision Transformers (ViT) for image classification tasks.
16. **`model.py`**: Defines a UNet architecture using a pre-trained ResNet-18 as the encoder for feature extraction.
17. **`modeli_details.py`**: Evaluates various aspects of a PyTorch model, such as model size, the number of parameters, FLOPs, inference time, and memory usage.
18. **`new_extract_1.py`**: Extracts specific facial features (eyes, lips, nose) from images using MediaPipe and saves them as separate images.
19. **`processed_data.py`**: Handles video frame extraction, face normalization, audio feature extraction, augmentation, and deepfake detection, saving the results into JSON files.

### Performance Evaluation

20. **`time.py`**: Calculates the total processing time for a set of files based on metadata from `metadata.json`.

## Setup

To get started, clone this repository:

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
