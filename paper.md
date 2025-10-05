title: "Modular Fall Detection via Pose Estimation and Posture-Specific LSTM Models for Real-Time Activity-Aware Safety Systems"

tags:
  - fall detection
  - pose estimation
  - LSTM
  - real-time AI
  - elderly care
  - activity recognition

authors:
  - name: Archit Gupta
    orcid: 
    affiliation: 1
affiliations:
  - name: Independent Researcher, Delhi, India
    index: 1
date: 2025-10-03
bibliography: paper.bib
---

# Summary

Falls are one of the leading causes of injury, hospitalization, and mortality among elderly and disabled populations, accounting for millions of emergency visits each year. Rapid and reliable detection of falls can significantly reduce medical complications and improve response times. However, existing solutions either depend on wearable sensors—which suffer from low compliance—or monolithic vision-based models that treat all fall scenarios uniformly. This often results in high false-positive rates and reduced generalization to different fall types.

This paper presents a modular, posture-aware fall detection framework that integrates pose estimation with activity-specific routing to improve detection performance. Using YOLOv8-Pose for 2D skeletal keypoint extraction, a lightweight posture classifier categorizes input sequences as standing, chair-based, or bed-based. Each sequence is then routed to a dedicated Bidirectional LSTM classifier trained exclusively on fall and non-fall data for that posture context. 

By combining posture-aware routing with specialized models, the system achieves high recall and improved robustness to context-specific false alarms while maintaining real-time performance on commodity hardware.

# Statement of Need

Falls account for more than **37 million severe injuries and 646,000 deaths annually worldwide** (WHO, 2024). In the United States alone, the CDC estimates **one in four adults aged 65+ falls each year**, leading to **over 3 million emergency visits**. The ability to detect falls quickly and reliably—especially without requiring wearable devices—is therefore critical for hospitals, eldercare facilities, and home monitoring systems. Real-time detection can shorten response times, prevent complications such as “long-lie” syndrome, and improve patient outcomes. 

Existing monolithic models typically fail to adapt to the different kinematics of standing, seated, or bed-based falls. Our system addresses this by explicitly modeling posture context and routing data to posture-specific classifiers, thereby reducing false positives and increasing recall in safety-critical settings.

This work provides an open-source, reproducible, and posture-aware fall detection pipeline that can be readily adapted for research or deployment in clinical and eldercare environments.

# System Overview

The core modules are:
1. **Pose Extraction** (YOLOv8): 17-keypoint skeletons extracted per frame.
2. **Posture Classification**: Trained on 10 keypoints to classify `standing`, `sitting`, or `bed`.
3. **Posture-Specific Fall Models**:
   - Each uses 12 keypoints as input sequences to a BiLSTM model.
   - Classifies fall (1) or non-fall (0) within a 2s window.
   - Independent scaler per model.

> **Source Code**: [GitHub Repository](https://github.com/Rchit13/multi-model-fall-detection-system) 
> **Pose Estimation Backbone**: YOLOv8-Pose by [Ultralytics](https://github.com/ultralytics/ultralytics)

## Model Summary

| Component             | Input Shape    | Model Type     | Output       | Frames | Keypoints |
|----------------------|----------------|----------------|--------------|--------|-----------|
| Posture Classifier   | (30, 20)       | LSTM           | 3 classes    | 30     | 10        |
| Fall Detector (All)  | (60, 24)       | BiLSTM         | Binary       | 60     | 12        |

# Implementation

## Dataset and Preprocessing

## Dataset and Preprocessing

The system is trained on the publicly available **Fall Vision: A Benchmark Video Dataset for Advancing Fall Detection Technology** (Rahman et al., 2024) [@fallvision]. This dataset consists of videos of staged falls and non-falls across multiple scenarios. Keypoints for each frame were extracted and stored as CSV files.  

For **posture classification**, sequences of **30 frames** (≈1s) are used.  
For **fall detection**, sequences of **60 frames** (≈2s) are used.  

Only keypoints with detection confidence >0.2 were retained. Sequences missing keypoints were zero-filled, but only sequences with >90% valid frames were included.  

- **Fall Detection Keypoints (12 total)**:  
  - Shoulders, Elbows, Wrists, Hips, Knees, Ankles (Left/Right)  

- **Posture Classifier Keypoints (10 total)**:  
  - Shoulders, Elbows, Hips, Knees, Ankles (Left/Right)

Each keypoint vector was flattened per frame and standardized using a `StandardScaler`. Each model has its own scaler saved as a `.pkl` file for consistent inference-time normalization.

### Posture Classifier  
- Input: `(30, 20)` sequences  
- LSTM (64 units)  
- Dropout (0.3)  
- Dense (32 units, ReLU)  
- Dense (3 units, Softmax for standing/chair/bed)

### Posture-Specific Fall Detectors  
Each of the three posture contexts uses an identical architecture:  
- Input: `(60, 24)` sequences  
- Bidirectional LSTM (128 units, return sequences)  
- Dropout (0.4)  
- Bidirectional LSTM (64 units)  
- Dense (64 units, ReLU)  
- Dropout (0.3)  
- Dense (1 unit, Sigmoid for fall/non-fall)

### Training Strategy  
Class imbalance between fall and non-fall samples was addressed with **class weighting** and a custom **focal loss**:

```python
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return loss
```

All models were trained using:
- Adam optimizer
- Early stopping (patience=4)
- 20% validation split
- Threshold tuning via grid search for best F1-score

# Evaluation

Metrics reported include Precision, Recall, F1-score, confusion matrices, and probability traces. The system achieved:

- Standing falls: Recall 0.92, F1-score 0.89
- Chair falls: Recall 0.88, F1-score 0.89
- Bed falls: Recall 0.85, F1-score 0.73

Training/validation loss and accuracy curves showed stable convergence with minimal overfitting.

All models were trained on pose sequences extracted from the Fall Vision Dataset [@fallvision2024]. Performance metrics were computed on held-out validation sets and optimized for recall. Thresholds were selected to maximize F1-score.

All models were trained on sequences extracted from the Fall Vision Dataset [@fallvision]. Results are based on stratified held-out validation sets.

> Visualizations and reports are saved in `/figures/`:
> - Accuracy/loss curves (`*_accuracy_curve.png`, `*_loss_curve.png`)
> - Confusion matrices (`*_confusion_matrix.png`)
> - Probability traces (`*_probability_trace.png`)
> - Classification reports (`*_classification_report.txt`)

# Real-Time Inference

At inference, YOLOv8-Pose extracts keypoints from incoming frames. The first second (30 frames) is fed into the posture classifier. The predicted posture then determines which fall detection model is used for the next stage. The complete pipeline runs at >20 FPS with <200 ms latency on an NVIDIA GTX 1660 Ti GPU, making it viable for real-time deployment.

# Reproducibility

This project is fully reproducible:
- All training/inference scripts are in the GitHub repo
- Models and scalers are checkpointed and versioned
- `requirements.txt` is provided
- Seeds are fixed across training runs
- Dataset is publicly available via Harvard Dataverse

# Acknowledgements

The author gratefully acknowledges the creators of the Fall Vision dataset for making their data publicly available, which formed the foundation for this work.

# References

### `paper.bib`
# References

```bibtex
@dataset{fallvision,
  author       = {Rahman, Nakiba Nuren and Mahi, Abu Bakar Siddique and Mistry, Durjoy and Masud, Shah Murtaza Rashid Al and Saha, Aloke Kumar and Rahman, Rashik and Islam, Md. Rajibul},
  year         = {2024},
  title        = {Fall Vision: A Benchmark Video Dataset for Advancing Fall Detection Technology},
  version      = {2.0},
  publisher    = {Harvard Dataverse},
  doi          = {10.7910/DVN/75QPKK},
  url          = {https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK&version=2.0}
}

@misc{ultralytics2023yolov8,
  title={YOLOv8: Ultralytics},
  url={https://github.com/ultralytics/ultralytics},
  author={Ultralytics},
  year={2023},
  note={Accessed: 2025-10-03}
}
```
