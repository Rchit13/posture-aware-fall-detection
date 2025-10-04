import argparse
import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


def load_data(data_path: str, sequence_length: int = 30):
    X, y = [], []
    for filename in os.listdir(data_path):
        if not filename.endswith(".csv"):
            continue

        label = 0 if filename.startswith("bed_") or filename.startswith("chair_") else 1
        filepath = os.path.join(data_path, filename)
        df = pd.read_csv(filepath)

        if df.shape[0] < sequence_length:
            pad = np.zeros((sequence_length - df.shape[0], df.shape[1]))
            df_padded = np.vstack([df.values, pad])
        else:
            df_padded = df.values[:sequence_length]

        X.append(df_padded)
        y.append(label)

    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train posture classifier.")
    parser.add_argument('--data', required=True, help='Path to CSV folder')
    parser.add_argument('--output-model', default='posture_classifier.h5', help='Output model filename')
    parser.add_argument('--output-scaler', default='posture_scaler.pkl', help='Output scaler filename')
    args = parser.parse_args()

    # === Load Data ===
    print("🔄 Loading data...")
    X, y = load_data(args.data)
    print(f"✅ Loaded {X.shape[0]} samples.")

    # === Preprocessing ===
    num_samples, seq_len, num_features = X.shape
    X_reshaped = X.reshape(num_samples * seq_len, num_features)
    scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped).reshape(num_samples, seq_len, num_features)

    # === Save Scaler ===
    with open(args.output_scaler, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Saved scaler to {args.output_scaler}")

    # === Train-Test Split ===
    y_cat = to_categorical(y)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_cat, test_size=0.2, stratify=y, random_state=42)

    # === Build & Train Model ===
    model = build_model(input_shape=(seq_len, num_features))
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    print("🚀 Training model...")
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])

    # === Evaluate ===
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    print("📊 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

    acc = accuracy_score(y_true_classes, y_pred_classes)
    print(f"🎯 Final Accuracy: {acc:.4f}")

    # === Save Model ===
    model.save(args.output_model)
    print(f"✅ Saved model to {args.output_model}")


if __name__ == "__main__":
    main()
