import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === Config ===
DATA_DIRS = {
    0: "data/train_keypoints_csv",         # Standing
    1: "data/train_keypoints_csv_chair",   # Chair
    2: "data/train_keypoints_csv_bed"      # Bed
}
KEYPOINTS = [
    "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow",
    "Left Hip", "Right Hip",
    "Left Knee", "Right Knee",
    "Left Ankle", "Right Ankle"
]
SEQUENCE_LEN = 30
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/posture_clf.h5"


# === Load and process sequences ===
def load_sequences(folder_path, label):
    sequences, labels = [], []
    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(folder_path, file))
        df = df[df["Keypoint"].isin(KEYPOINTS)]
        df = df[df["Confidence"] > 0.2]

        frames = sorted(df["Frame"].unique())[:SEQUENCE_LEN]
        if len(frames) < SEQUENCE_LEN:
            continue

        seq = []
        for f in frames:
            frame_data = df[df["Frame"] == f]
            kp_dict = {row["Keypoint"]: (row["X"], row["Y"]) for _, row in frame_data.iterrows()}

            coords = []
            for kp in KEYPOINTS:
                x, y = kp_dict.get(kp, (0, 0))
                coords.extend([x, y])

            # Normalize by mid-hip
            l_hip = kp_dict.get("Left Hip", (0, 0))
            r_hip = kp_dict.get("Right Hip", (0, 0))
            mid_hip_x = (l_hip[0] + r_hip[0]) / 2
            mid_hip_y = (l_hip[1] + r_hip[1]) / 2
            coords = [(c - mid_hip_x if i % 2 == 0 else c - mid_hip_y) for i, c in enumerate(coords)]
            seq.append(coords)

        sequences.append(seq)
        labels.append(label)
    return sequences, labels


# === Aggregate Data ===
X_all, y_all = [], []
for label, path in DATA_DIRS.items():
    X, y = load_sequences(path, label)
    X_all.extend(X)
    y_all.extend(y)

X_all = np.array(X_all)
y_all = np.array(y_all)
print(f"✅ Loaded {X_all.shape[0]} samples, shape: {X_all.shape}")

# === Normalize and save scaler ===
scaler = StandardScaler()
X_flat = X_all.reshape(-1, X_all.shape[-1])
X_scaled = scaler.fit_transform(X_flat)
X_all = X_scaled.reshape(X_all.shape)

os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"📦 Saved scaler → {SCALER_PATH}")

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

# === Build Model ===
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LEN, X_all.shape[-1])),
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# === Train ===
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# === Evaluate ===
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred))

# === Save Model ===
model.save(MODEL_PATH)
print(f"✅ Saved model → {MODEL_PATH}")
