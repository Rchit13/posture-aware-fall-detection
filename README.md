# Posture-Aware Modular Fall Detection System

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)


A modular deep learning framework for real-time fall detection using pose estimation and posture-specific classifiers. This system integrates YOLOv8-based keypoint extraction with BiLSTM models customized for three distinct postures: standing, sitting (chair), and lying (bed). Designed for safety-critical applications in elder care, clinical monitoring, and activity-aware surveillance.

## System Architecture

The fall detection pipeline operates in three modular stages:

1. **Pose Estimation**: 2D Skeletal keypoints are extracted from input video using the [YOLOv8-Pose](https://docs.ultralytics.com/models/yolov8-pose/) model.
2. **Posture Classification**: A posture classifier trained on normalized keypoints identifies whether the subject is standing, seated, or in bed.
3. **Fall Detection**: A posture-specific BiLSTM model determines whether a fall has occurred based on the classified posture and motion sequence.

Each model (posture classifier and fall detectors) is trained on temporally structured pose sequences normalized relative to the subject’s mid-hip position.

<div align="center"> <img src="docs/architecture.png" width="600"> </div>

## Features

- Real-time frame-by-frame inference with minimal latency
- Temporal Modeling with posture-aware BiLSTM architectures
- Mid-hip Normalization of keypoints for spatial consistency
- Separate fall detection models for distinct posture contexts
- Modular Training & Inference Pipelines

## Directory Structure

```bash
posture-aware-fall-detection/
├── src/                  # Training & inference scripts
├── scripts/              # Preprocessing tools
├── models/               # Pretrained weights and scalers
├── tests/                # Sample video and test script
├── reports/              # Classification reports
├── figures/              # Training curves and visualizations
├── docs/                 # Architecture diagram, documentation assets
├── paper.md              # JOSS manuscript
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

Clone the repository and install the required dependencies using:
```bash
pip install -r requirements.txt
```

Core Dependencies:
- Python ≥ 3.10
- TensorFlow 2.x
- PyTorch + Ultralytics
- OpenCV
- Scikit-learn

## Inference Usage

Run fall detection on a video clip:
```bash
python src/inference.py \
  --video tests/sample_video.mp4 \
  --model_dir models/ \
  --weights yolov8n-pose.pt
```

This will:
- Extract keypoints using YOLOv8-Pose
- Classify posture using posture_classifier.h5
- Dynamically load the appropriate posture-specific BiLSTM
- Output the predicted posture (standing, chair, bed) and fall status (fall, not fall)

All .h5 and .pkl models must reside in the models/ directory.

## Dataset

This project uses the [Fall Vision Dataset](https://doi.org/10.7910/DVN/75QPKK) for training and validation. The dataset includes staged fall and non-fall videos across standing, sitting, and lying contexts.

Keypoint sequences were extracted from video frames using the [YOLOv8-Pose](https://docs.ultralytics.com/models/yolov8-pose/) model. The extracted (x, y, confidence) values were saved as CSVs.

- Fall detection models use 12 keypoints (24 features per frame)
- Posture classifier uses 10 keypoints (20 features per frame)
- Sequences are sampled as 60-frame (2 sec) and 30-frame (1 sec) windows respectively
- Only keypoints with confidence > 0.2 were retained

## Model Training

Before training any models, you must first generate the training datasets (pose keypoint sequences) from the raw Harvard Fall dataset. These keypoints are extracted using the yolov8-pose model and saved as .csv files, which are then used to train the posture classifier and fall detection models.

### Step 1: Generate Keypoint Data (Preprocessing)

Use the scripts/extract_keypoints.py script to download, extract, and process the Harvard dataset:
```bash
python scripts/extract_keypoints.py
```

This script performs the following:
- Downloads .rar files from the Harvard Dataverse
- Extracts all videos
- Runs pose estimation (YOLOv8-Pose) to save per-frame keypoint .csv files for each video

This creates:
```bash
/data/
├── harvard_rars/         # Downloaded .rar files
├── harvard_raw/          # Extracted video folders
├── all_keypoints_csv/    # Flattened .csv keypoints (all videos)
```

Note: You may need to install unrar locally:
```
On Ubuntu/Debian
sudo apt install unrar

On Mac (with Homebrew)
brew install unrar
```

Or manually extract .rar files if preferred.


### Step 2: Structure by Posture

Once raw keypoints are generated, use the scripts/split_by_posture.py script to classify each CSV file by posture and place it into the appropriate subfolder:
```bash
python scripts/split_keypoints_by_type.py
```

This creates:
```bash
/data/
├── train_keypoints_csv/         # Standing sequences
├── train_keypoints_csv_chair/  # Sitting (chair) sequences
├── train_keypoints_csv_bed/    # Bed sequences
```

These posture-specific directories are used to train each corresponding fall model as well as the posture classifier.

### Step 3: Train the Posture Classifier

Train the posture classifier using the generated CSVs for all three postures:
```bash
python src/train_posture.py --data data/
```

This will produce:

```bash
models/
├── posture_classifier.h5
├── posture_scaler.pkl
```

### Step 4: Train Fall Detection Models

Train posture-specific fall detection models:
```bash
python src/train_lstm.py --data data/train_keypoints_csv/          # Standing
python src/train_lstm.py --data data/train_keypoints_csv_chair/    # Chair
python src/train_lstm.py --data data/train_keypoints_csv_bed/      # Bed
```

This will generate:

```bash
models/
├── standing_fall_model.h5
├── standing_scaler.pkl
├── chair_fall_model.h5
├── chair_scaler.pkl
├── bed_fall_model.h5
├── bed_scaler.pkl
```

These are binary classifiers (fall vs non-fall) trained on 30-frame sequences of mid-hip normalized keypoints. All models use custom focal loss to handle class imbalance.

## Model Summary

| Model | Input | Output Classes | Output File |
|-------|-------|----------------|-------------|
| Posture Classifier | 60-frame window of 10 keypoints | Standing / Chair / Bed | `posture_classifier.h5` |
| Standing Fall Detector | 30-frame window of 12 keypoints | Fall / Not Fall | `standing_fall_model.h5` |
| Chair Fall Detector | 30-frame window of 12 keypoints | Fall / Not Fall | `chair_fall_model.h5` |
| Bed Fall Detector | 30-frame window of 12 keypoints | Fall / Not Fall | `bed_fall_model.h5` |

## Pretrained Models

The repository includes the following pretrained components:
- posture_classifier.h5
- standing_fall_model.h5, chair_fall_model.h5, bed_fall_model.h5
- *_scaler.pkl for input standardization

All models are trained on sequences of 30 or 60 frames, using normalized (x, y) joint coordinates derived from YOLOv8 pose keypoints.

## Testing

A minimal test script is provided:
```bash
python tests/test_inference.py
```

## License

This software is released under the MIT License. See LICENSE for details.

## Citation

A paper associated with this software is currently under submission to the Journal of Open Source Software (JOSS)  . Citation information will be added upon acceptance.

Developed and maintained by Archit Gupta
