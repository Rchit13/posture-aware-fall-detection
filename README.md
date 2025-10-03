# Posture-Aware Modular Fall Detection System

This repository presents a modular deep learning pipeline for real-time human fall detection using pose estimation and posture-specific classification. The system integrates pose keypoint extraction via YOLOv8 with separate BiLSTM classifiers tailored to three human postures: standing, sitting (chair), and lying (bed). It is designed for safety monitoring in clinical and elderly care environments.

## Overview

The fall detection pipeline operates in three stages:

1. **Pose Estimation**: Skeletal keypoints are extracted from input video using the YOLOv8-pose model.
2. **Posture Classification**: A posture classifier trained on normalized keypoints identifies whether the subject is standing, seated, or in bed.
3. **Fall Detection**: Based on the predicted posture, a corresponding BiLSTM model evaluates whether the observed motion sequence constitutes a fall.

Each model (posture classifier and fall detectors) is trained on temporally structured pose sequences normalized relative to the subject’s mid-hip position.

## Features

- Real-time frame-by-frame inference with minimal latency
- Robust keypoint normalization centered on anatomical landmarks
- Separate fall detection models for distinct posture contexts
- High-recall configuration optimized for safety-critical use cases

## Directory Structure

posture-aware-fall-detection/
├── src/ # Inference and training scripts
├── models/ # Pretrained .h5 models and .pkl scalers
├── tests/ # Sample test videos
├── data/ # Dataset links and structure
├── paper.md # JOSS manuscript
├── requirements.txt
├── README.md
└── LICENSE

## Installation

Clone the repository and install the required dependencies:


```bash
git clone https://github.com/Rchit13/posture-aware-fall-detection.git
cd posture-aware-fall-detection
pip install -r requirements.txt

This project requires:
- Python 3.10+
- TensorFlow 2.x
- PyTorch (for YOLOv8)
- Ultralytics library
- OpenCV
- Scikit-learn
```

## Inference

To run fall detection on a video:
python src/inference.py --video path/to/input.mp4


This performs:
- Keypoint extraction via YOLOv8-pose
- Posture classification using the posture model
- Fall prediction using the appropriate BiLSTM model

All models and scalers must be placed in the models/ directory.

## Model Training

To train a new fall detection model:
python src/train_lstm.py --data data/standing/

To train the posture classifier:
python src/train_posture.py --data data/postures/

### Pretrained Models

The repository includes the following pretrained components:
- posture_classifier.h5
- standing_fall_model.h5, chair_fall_model.h5, bed_fall_model.h5
- *_scaler.pkl for input standardization

All models are trained on sequences of 30 frames, using normalized (x, y) joint coordinates derived from YOLOv8 pose keypoints.

### Dataset

Training data was curated from publicly available datasets including, namely Fall Vision: A Benchmark Video Dataset for Advancing Fall Detection Technology

Each sequence is extracted at 30 FPS and processed to generate normalized pose sequences for training.

## License

This software is released under the MIT License. See LICENSE for details.

## Citation

A paper associated with this software is currently under submission to the Journal of Open Source Software (JOSS)
. Citation information will be added upon acceptance.
