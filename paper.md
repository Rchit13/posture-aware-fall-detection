title: "Multi-Model Fall Detection Using Pose Estimation and Posture Classification for Activity-Aware Safety Systems"

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
   - Classifies fall (1) or non-fall (0) within a 2.5s window.
   - Independent scaler per model.

Each model is saved as `.h5` with corresponding `.pkl` scalers.

# Implementation

## Dataset and Preprocessing

The system is trained on the publicly available **Fall Vision: A Benchmark Video Dataset for Advancing Fall Detection Technology** (Rahman et al., 2024) [@fallvision]. This dataset consists of videos of staged falls and non-falls across multiple scenarios. Keypoints for each frame were extracted and stored as CSV files.  

For each posture category, sequences of 60 consecutive frames (≈2 seconds at 30 FPS) were extracted. Only keypoints with detection confidence >0.2 were retained. For fall detection, 12 limb joints were selected:

- Left/Right Shoulders  
- Left/Right Elbows  
- Left/Right Wrists  
- Left/Right Hips  
- Left/Right Knees  
- Left/Right Ankles  

These were flattened into 24-dimensional vectors per frame. For posture classification, a reduced set of 10 keypoints was used.

All sequences were standardized using a `StandardScaler` fitted on the training set, and the scaler was saved for use at inference time.

## Model Architecture

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
All models were trained with the Adam optimizer, early stopping (patience=4), and 20% validation split. The classification threshold was tuned via grid search to maximize F1-score.

# Evaluation

Metrics reported include Precision, Recall, F1-score, confusion matrices, and probability traces. The system achieved:

Standing falls: Recall 0.92, F1-score 0.89

Chair falls: Recall 0.88, F1-score 0.89

Bed falls: Recall 0.85, F1-score 0.73

Training/validation loss and accuracy curves showed stable convergence with minimal overfitting.

All models were trained on pose sequences extracted from the Fall Vision Dataset [@fallvision2024]. Performance metrics were computed on held-out validation sets and optimized for recall. Thresholds were selected to maximize F1-score.

All relevant curves are saved in the repository under `/figures/`, including:

- Accuracy and loss curves (`*_accuracy_curve.png`, `*_loss_curve.png`)
- Probability traces (`*_probability_trace.png`)
- Confusion matrices (`*_confusion_matrix.png`)
- Classification reports (`*_classification_report.txt`)

# Real-Time Inference

At inference, YOLOv8-Pose extracts keypoints from incoming frames. The first second (30 frames) is fed into the posture classifier. The predicted posture then determines which fall detection model is used for the next stage. The complete pipeline runs at >20 FPS with <200 ms latency on an NVIDIA GTX 1660 Ti GPU, making it viable for real-time deployment.

# Acknowledgements

I thank the authors of the Fall Vision dataset for making their data publicly available.

# References

### `paper.bib`
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
```
