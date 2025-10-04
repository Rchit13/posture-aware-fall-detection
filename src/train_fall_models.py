import os
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === Constants ===
KEYPOINTS_OF_INTEREST = [
    "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow",
    "Left Hip", "Right Hip",
    "Left Knee", "Right Knee",
    "Left Ankle", "Right Ankle"
]
SEQUENCE_LENGTH = 30

POSTURE_CONFIG = {
    "standing": {
        "folder": "train_keypoints_csv",
        "mask": "_mask_s",
        "model_name": "standing_fall_model.h5",
        "scaler_name": "standing_scaler.pkl",
        "output_prefix": "standing"
    },
    "chair": {
        "folder": "train_keypoints_csv_chair",
        "mask": "_mask_c",
        "model_name": "chair_fall_model.h5",
        "scaler_name": "chair_scaler.pkl",
        "output_prefix": "chair"
    },
    "bed": {
        "folder": "train_keypoints_csv_bed",
        "mask": "_mask_b",
        "model_name": "bed_fall_model.h5",
        "scaler_name": "bed_scaler.pkl",
        "output_prefix": "bed"
    }
}

def load_sequences(folder_path):
    sequences, labels = [], []
    for file in os.listdir(folder_path):
        if not file.endswith(".csv"): continue
        label = 1 if file.startswith("f_") else 0
        df = pd.read_csv(os.path.join(folder_path, file))
        df = df[df['Keypoint'].isin(KEYPOINTS_OF_INTEREST)]
        df = df[df['Confidence'] > 0.2]
        frames = sorted(df['Frame'].unique())[:SEQUENCE_LENGTH]
        if len(frames) < SEQUENCE_LENGTH: continue

        sequence = []
        for f in frames:
            frame_data = df[df['Frame'] == f]
            kp_dict = {row['Keypoint']: (row['X'], row['Y']) for _, row in frame_data.iterrows()}
            coords = []
            for kp in KEYPOINTS_OF_INTEREST:
                x, y = kp_dict.get(kp, (0, 0))
                coords.extend([x, y])

            left_hip = kp_dict.get("Left Hip", (0, 0))
            right_hip = kp_dict.get("Right Hip", (0, 0))
            mid_x = (left_hip[0] + right_hip[0]) / 2
            mid_y = (left_hip[1] + right_hip[1]) / 2
            coords = [(val - mid_x if i % 2 == 0 else val - mid_y) for i, val in enumerate(coords)]
            sequence.append(coords)

        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def plot_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(path, dpi=300)
    plt.close()

def plot_training_curves(history, prefix):
    for metric in ['loss', 'accuracy']:
        plt.plot(history.history[metric], label=f"Train {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"Val {metric}")
        plt.title(f"{metric.capitalize()} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(f"{prefix}_{metric}_curve.png", dpi=300)
        plt.close()

def train(posture, output_dir=".", prefix=None):
    cfg = POSTURE_CONFIG[posture]
    folder = os.path.join("data/fall_keypoints_split", cfg['folder'])
    print(f"📂 Loading data from {folder}...")
    X, y = load_sequences(folder)
    print("✅ Data shape:", X.shape)

    # Create output directories
    models_dir = os.path.join(output_dir, "models")
    reports_dir = os.path.join(output_dir, "reports")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    scaler = StandardScaler()
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_flat)
    X = X_scaled.reshape(X.shape)

    scaler_name = f"{prefix}_{cfg['scaler_name']}" if prefix else cfg['scaler_name']
    with open(os.path.join(models_dir, scaler_name), "wb") as f:
        pickle.dump(scaler, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X.shape[-1])),
        Dropout(0.3),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop])

    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    report = classification_report(y_test, y_pred, digits=4)

    model_name = f"{prefix}_{cfg['model_name']}" if prefix else cfg['model_name']
    model.save(os.path.join(models_dir, model_name))
    print(f"✅ Saved model: {model_name}")

    report_path = os.path.join(reports_dir, f"{prefix+'_' if prefix else ''}{cfg['output_prefix']}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    plot_confusion_matrix(y_test, y_pred, os.path.join(figures_dir, f"{prefix+'_' if prefix else ''}{cfg['output_prefix']}_confusion_matrix.png"),
                          title=f"{cfg['output_prefix'].capitalize()} Fall Detection")
    plot_training_curves(history, os.path.join(figures_dir, f"{prefix+'_' if prefix else ''}{cfg['output_prefix']}"))
    print("✅ Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--posture', choices=['standing', 'chair', 'bed'], required=True, help="Type of posture")
    parser.add_argument('--output_dir', default='.', help="Base output directory (default: current dir)")
    parser.add_argument('--prefix', default=None, help="Optional prefix for model/report filenames")
    args = parser.parse_args()
    train(args.posture, args.output_dir, args.prefix)
