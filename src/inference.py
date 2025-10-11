#!/usr/bin/env python3
"""
Inference script for Posture-Aware Modular Fall Detection System.

Usage:
    python src/inference.py --video tests/sample_video.mp4 \
        --model_dir models/ \
        --weights yolov8n-pose.pt \
        --seq_len 30 \
        --device cpu \
        --display
"""

import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import pickle

# === Custom Focal Loss (for loading models trained with it) ===
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return loss


# === Padding helper ===
def pad_sequence(X, length, n_features):
    """Pad or trim sequence to fixed length."""
    if len(X) == 0:
        return np.zeros((length, n_features))
    X = np.array(X)
    if X.shape[0] < length:
        pad = np.zeros((length - X.shape[0], X.shape[1]))
        return np.vstack([X, pad])
    return X[:length]


def main(video_path, model_dir, yolo_weights, seq_len, device, display):
    posture_names = ['standing', 'chair', 'bed']

    posture_indices = {
        'posture': [5, 6, 7, 8, 11, 12, 13, 14, 15, 16],  # shoulders, elbows, hips, knees, ankles
        'fall': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # full limb joints
    }

    # === Load YOLOv8 Pose model ===
    print(f"📦 Loading YOLOv8 Pose model on device: {device}")
    pose_model = YOLO(yolo_weights)

    # === Load Posture Classifier ===
    posture_model_path = os.path.join(model_dir, "posture_classifier.h5")
    posture_scaler_path = os.path.join(model_dir, "posture_scaler.pkl")
    posture_model = load_model(posture_model_path)
    with open(posture_scaler_path, 'rb') as f:
        posture_scaler = pickle.load(f)

    # === Extract Keypoints from Video ===
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 Processing video: {video_path} ({total_frames} frames)")

    posture_keypoints, fall_keypoints = [], []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run YOLOv8 Pose inference
        result = pose_model.predict(source=frame, save=False, verbose=False, device=device)[0]

        if result.keypoints is not None and len(result.keypoints.data) > 0:
            person = result.keypoints.data.cpu().numpy()[0]
            l_hip, r_hip = person[11][:2], person[12][:2]
            mid_hip_x = (l_hip[0] + r_hip[0]) / 2
            mid_hip_y = (l_hip[1] + r_hip[1]) / 2

            coords_posture, coords_fall = [], []
            for i in range(17):
                x, y = person[i][:2]
                x -= mid_hip_x
                y -= mid_hip_y
                if i in posture_indices['posture']:
                    coords_posture.extend([x, y])
                if i in posture_indices['fall']:
                    coords_fall.extend([x, y])

            posture_keypoints.append(coords_posture)
            fall_keypoints.append(coords_fall)

        if display:
            cv2.putText(frame, f"Frame: {frame_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(posture_keypoints) == 0:
        print("⚠️ No poses detected. Please check input video.")
        return

    # === Posture Prediction ===
    X_posture = pad_sequence(posture_keypoints, seq_len, len(posture_indices['posture'])*2)
    X_posture_scaled = posture_scaler.transform(X_posture)
    X_posture_scaled = np.expand_dims(X_posture_scaled, axis=0)

    y_posture_pred = posture_model.predict(X_posture_scaled)
    predicted_posture = posture_names[np.argmax(y_posture_pred)]
    print(f"🧍 Detected posture: {predicted_posture}")

    # === Load Corresponding Fall Model & Scaler ===
    fall_model_path = os.path.join(model_dir, f"{predicted_posture}_fall_model.h5")
    fall_scaler_path = os.path.join(model_dir, f"{predicted_posture}_scaler.pkl")

    print(f"📦 Loading {predicted_posture} fall model...")
    fall_model = load_model(fall_model_path, custom_objects={'loss': focal_loss()})
    with open(fall_scaler_path, 'rb') as f:
        fall_scaler = pickle.load(f)

    # === Fall Detection ===
    X_fall = pad_sequence(fall_keypoints, seq_len, len(posture_indices['fall'])*2)
    X_fall_scaled = fall_scaler.transform(X_fall)
    X_fall_scaled = np.expand_dims(X_fall_scaled, axis=0)

    y_fall_pred = fall_model.predict(X_fall_scaled)
    fall_prob = float(y_fall_pred[0][0])
    fall_label = "FALL" if fall_prob > 0.42 else "NO FALL"
    print(f"🛑 Fall Detection: {fall_label} (p={fall_prob:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Posture-Aware Fall Detection on a video")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model_dir', type=str, default='models/', help='Directory with models and scalers')
    parser.add_argument('--weights', type=str, default='yolov8n-pose.pt', help='Path to YOLOv8 pose weights')
    parser.add_argument('--seq_len', type=int, default=30, help='Number of frames to use in sequence')
    parser.add_argument('--device', type=str, default='cpu', help='Computation device (cpu or cuda)')
    parser.add_argument('--display', action='store_true', help='Display annotated video frames during inference')
    args = parser.parse_args()

    main(args.video, args.model_dir, args.weights, args.seq_len, args.device, args.display)
