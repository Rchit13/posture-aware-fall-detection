title: "Multi-Model Fall Detection Using Pose Estimation and Posture Classification for Activity-Aware Safety Systems"

tags:
  - fall detection
  - pose estimation
  - deep learning
  - eldercare
  - computer vision

authors:
  - name: Archit Gupta
    orcid: 0000-0000-0000-0000  # replace with your actual ORCID
    affiliation: 1
affiliations:
  - name: Independent Researcher, Delhi, India
    index: 1
date: 2025-10-03
bibliography: paper.bib
---

# Summary

Falls are among the leading causes of injury and mortality in elderly and disabled populations. Existing fall detection systems often rely on wearable sensors or monolithic vision-based models, which face limitations such as high false-positive rates, low user compliance, and poor generalization across diverse activity contexts.  

This paper presents a **modular, posture-aware fall detection framework** that integrates 2D skeletal keypoint extraction, posture classification, and posture-specific fall detection models. Using YOLOv8-Pose for real-time keypoint estimation, the system first classifies each sequence as standing, chair-based, or bed-based activity. Each sequence is then routed to a dedicated Bidirectional LSTM classifier trained exclusively on falls and non-falls within that posture context.  

This modular design improves detection precision without compromising recall and enables real-time performance on consumer-grade hardware. The approach is suitable for deployment in eldercare, hospitals, and assisted living facilities, where high recall of true falls is critical to safety.

# Statement of Need

Falls represent a major public health challenge: according to the WHO, approximately 684,000 fatal falls occur each year globally, making falls the second leading cause of unintentional injury deaths worldwide. Quick and reliable detection is crucial, as delays in response time dramatically increase the likelihood of severe injury and long-term health consequences.  

Traditional solutions such as accelerometer-based wearables often fail due to user noncompliance, while monolithic vision-based systems struggle to differentiate between posture contexts (e.g., a fall from standing versus voluntary lying down). The proposed system addresses this gap by combining pose-based representations with posture-specific detection, resulting in more robust and interpretable performance.

# Software Description

The framework operates in three stages:

1. **Pose Estimation**: YOLOv8-Pose extracts 2D skeletal keypoints (17 COCO joints) from each frame.
2. **Posture Classification**: A lightweight LSTM-based model categorizes each 30-frame sequence as standing, chair, or bed activity.
3. **Posture-Specific Fall Detection**: Three specialized BiLSTM classifiers—one per posture—predict fall versus non-fall events for their respective contexts.

The models were trained on normalized keypoint sequences of 30 frames (≈1 second at 30 FPS). Mid-hip normalization was applied to reduce camera-position effects. Class imbalance was addressed with focal loss and class weighting.  

During inference, the system runs end-to-end in real time, automatically selecting the correct model for each detected posture, ensuring both high recall and low latency.

# Example Usage

```python
from fall_detection import PostureFallSystem

system = PostureFallSystem(
    posture_model_path='posture_classifier.h5',
    fall_model_paths={
        'standing': 'standing_fall_model.h5',
        'chair': 'chair_fall_model.h5',
        'bed': 'bed_fall_model.h5'
    }
)

system.run(video_path='test_video.mp4')
