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

```
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
```
git clone https://github.com/Rchit13/posture-aware-fall-detection.git
cd posture-aware-fall-detection
pip install -r requirements.txt
```

**Core Dependencies:**
- Python ≥ 3.10  
- TensorFlow 2.x  
- PyTorch + Ultralytics  
- OpenCV  
- Scikit-learn  
- NumPy  
- Pandas  

**Running in Google Colab (GPU)**

Google Colab’s CUDA/cuDNN versions occasionally mismatch TensorFlow 2.17 and PyTorch 2.2 wheels.
If you encounter `libcudnn.so` or GPU initialization errors, run:

```
!apt-get install -y libcudnn8=8.9.2.* libcudnn8-dev=8.9.2.*
```

## QuickStart

```
# Run demo inference
git clone https://github.com/Rchit13/posture-aware-fall-detection.git
cd posture-aware-fall-detection
pip install -r requirements.txt
python tests/test_inference.py
```

## Inference Usage

Run fall detection on a video clip:
```
python src/inference.py \
  --video tests/sample_video.mp4 \
  --model_dir models/ \
  --weights yolov8n-pose.pt
```

This command will:
- Extract keypoints using YOLOv8-Pose  
- Classify posture using `posture_classifier.h5`  
- Dynamically load the corresponding posture-specific BiLSTM  
- Output predicted posture and fall status

Example output:
```
[INFO] Posture: Chair | Fall Status: Not Fall
```

All `.h5` and `.pkl` models must reside in the `models/` directory.

## Dataset

This project uses the [Fall Vision Dataset](https://doi.org/10.7910/DVN/75QPKK) for training and validation. The dataset includes staged fall and non-fall videos across standing, sitting, and lying contexts.

Keypoint sequences were extracted from video frames using the [YOLOv8-Pose](https://docs.ultralytics.com/models/yolov8-pose/) model. The extracted (x, y, confidence) values were saved as CSVs.

- Each fall detection model uses 12 keypoints (24 features per frame)  
- The posture classifier uses 10 keypoints (20 features per frame)  
- Sequences are sampled as 60-frame (2 s) and 30-frame (1 s) windows respectively  
- Only keypoints with confidence > 0.2 were retained  
- All videos are publicly available and contain staged falls with participant consent  

## Model Training

Before training any models, you must first generate the training datasets (pose keypoint sequences) from the raw Harvard Fall dataset. These keypoints are extracted using the yolov8-pose model and saved as .csv files, which are then used to train the posture classifier and fall detection models.

### Step 1: Generate Keypoint Data (Preprocessing)

```
python scripts/extract_keypoints.py
```

This script:
- Downloads `.rar` files from the Harvard Dataverse  
- Extracts all videos  
- Runs pose estimation (YOLOv8-Pose) to save per-frame keypoint `.csv` files  


This creates:
```
/data/
├── harvard_rars/         # Downloaded .rar files
├── harvard_raw/          # Extracted video folders
├── all_keypoints_csv/    # Flattened .csv keypoints (all videos)
```

Install `unrar` if needed:

```
# Ubuntu/Debian
sudo apt install unrar

# macOS (Homebrew)
brew install unrar
```

### Step 2: Structure by Posture

Once raw keypoints are generated, use the scripts/split_by_posture.py script to classify each CSV file by posture and place it into the appropriate subfolder:
```bash
python scripts/split_keypoints_by_type.py
```

This creates:
```
/data/
├── train_keypoints_csv/         # Standing sequences
├── train_keypoints_csv_chair/  # Sitting (chair) sequences
├── train_keypoints_csv_bed/    # Bed sequences
```

These posture-specific directories are used to train each corresponding fall model as well as the posture classifier.

### Step 3: Train the Posture Classifier

Train the posture classifier using the generated CSVs for all three postures:
```
python src/train_posture.py --data data/
```

This will produce:

```
models/
├── posture_classifier.h5
├── posture_scaler.pkl
```

### Step 4: Train Fall Detection Models

```
python src/train_lstm.py --data data/train_keypoints_csv/          # Standing
python src/train_lstm.py --data data/train_keypoints_csv_chair/    # Chair
python src/train_lstm.py --data data/train_keypoints_csv_bed/      # Bed
```

This will generate:

```
models/
├── standing_fall_model.h5
├── standing_scaler.pkl
├── chair_fall_model.h5
├── chair_scaler.pkl
├── bed_fall_model.h5
├── bed_scaler.pkl
```

Each BiLSTM model comprises two LSTM layers (128 and 64 units) with dropout (0.3) and early stopping for regularization.  

Custom focal loss is applied to address class imbalance.

## Model Summary

| Model | Input | Output Classes | Output File |
|-------|-------|----------------|-------------|
| Posture Classifier | 30-frame window of 20 keypoints | Standing / Chair / Bed | `posture_classifier.h5` |
| Standing Fall Detector | 60-frame window of 24 keypoints | Fall / Not Fall | `standing_fall_model.h5` |
| Chair Fall Detector | 60-frame window of 24 keypoints | Fall / Not Fall | `chair_fall_model.h5` |
| Bed Fall Detector | 60-frame window of 24 keypoints | Fall / Not Fall | `bed_fall_model.h5` |

## Pretrained Models

The repository includes the following pretrained components:
- posture_classifier.h5
- standing_fall_model.h5, chair_fall_model.h5, bed_fall_model.h5
- *_scaler.pkl for input standardization

These models are provided **for inference only**; retraining can be performed using the scripts above.

## Testing

A minimal test script is provided:
```bash
python tests/test_inference.py
```

## License

This software is released under the **MIT License**.  
See the `LICENSE` file for details.

## Citation

A paper associated with this software is currently under submission to the Journal of Open Source Software (JOSS). Citation information will be added upon acceptance.

Developed and maintained by Archit Gupta
