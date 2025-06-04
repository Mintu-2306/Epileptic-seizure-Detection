#bonn stft 4fold

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Constants
DATASET_PATH = '/content/drive/MyDrive/data set/Bonn Univeristy Dataset'
FOLDERS = ['F', 'N', 'O', 'S', 'Z']
LABELS = {'F': 0, 'N': 0, 'O': 0, 'Z': 0, 'S': 1}
ORIGINAL_LENGTH = 4097
NEW_LENGTH = 4096
SEGMENT_SIZE = 256
SEGMENTS_PER_FILE = NEW_LENGTH // SEGMENT_SIZE

# Storage
raw_segments, spectrograms, labels = [], [], []

# Function to create spectrogram
def create_spectrogram(segment):
    f, t, Sxx = spectrogram(segment, fs=256)
    Sxx = Sxx[:77, :75]
    if Sxx.shape != (77, 75):
        Sxx = np.resize(Sxx, (77, 75))
    return Sxx

# Load and preprocess data
for folder in FOLDERS:
    folder_path = os.path.join(DATASET_PATH, folder)
    label = LABELS[folder]

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.txt'):
            signal = np.loadtxt(os.path.join(folder_path, filename))

            if len(signal) == ORIGINAL_LENGTH:
                signal = signal[:-1]

            segments = np.split(signal, SEGMENTS_PER_FILE)

            for seg in segments:
                scaler = StandardScaler()
                seg_norm = scaler.fit_transform(seg.reshape(-1, 1)).flatten()
                spec = create_spectrogram(seg_norm)

                raw_segments.append(seg_norm)
                spectrograms.append(spec)
                labels.append(label)

print("Preprocessing done!")

# Convert to arrays
X_raw = np.array(raw_segments)                    # (N, 256)
X_spec = np.expand_dims(np.array(spectrograms), -1)  # (N, 77, 75, 1)
y = np.array(labels)

print(f"X_raw: {X_raw.shape}, X_spec: {X_spec.shape}, y: {y.shape}")

# Train-Test Split (85%-15%): Hold out test data
X_raw_remain, X_raw_test, X_spec_remain, X_spec_test, y_remain, y_test = train_test_split(
    X_raw, X_spec, y, test_size=0.15, stratify=y, random_state=42
)

# Model builder
def build_combined_model():
    # LSTM for raw EEG input
    input_raw = Input(shape=(256,))
    x1 = layers.Reshape((256, 1))(input_raw)
    x1 = layers.LSTM(64, return_sequences=True)(x1)
    x1 = layers.Dropout(0.2)(x1)
    x1 = layers.Dense(32, activation='sigmoid')(x1)
    x1 = layers.Dropout(0.2)(x1)
    x1 = layers.Dense(16, activation='relu')(x1)
    x1 = layers.Dropout(0.2)(x1)
    x1 = layers.Dense(8, activation='relu')(x1)
    x1 = layers.Dropout(0.2)(x1)
    x1 = layers.Flatten()(x1)

    # CNN for spectrogram input
    input_spec = Input(shape=(77, 75, 1))
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_spec)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)

    x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)

    x2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)

    x2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)

    x2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)

    x2 = layers.Flatten()(x2)

    # Combine features
    combined = layers.concatenate([x1, x2])
    combined = layers.Dense(64, activation='sigmoid')(combined)
    combined = layers.Dense(32, activation='sigmoid')(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    return models.Model(inputs=[input_raw, input_spec], outputs=output)


# Cross-validation
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw_remain, y_remain)):
    print(f"\n--- Fold {fold + 1} ---")
    Xr_train, Xr_val = X_raw_remain[train_idx], X_raw_remain[val_idx]
    Xs_train, Xs_val = X_spec_remain[train_idx], X_spec_remain[val_idx]
    y_train, y_val = y_remain[train_idx], y_remain[val_idx]

    model = build_combined_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(patience=3, restore_best_weights=True)

    history = model.fit(
        [Xr_train, Xs_train], y_train,
        validation_data=([Xr_val, Xs_val], y_val),
        epochs=10,
        batch_size=4,
        callbacks=[es],
        verbose=1
    )

    val_loss, val_acc = model.evaluate([Xr_val, Xs_val], y_val)
    print(f"Validation Accuracy for Fold {fold + 1}: {val_acc * 100:.2f}%")
    fold_accuracies.append(val_acc)

# Final model on full train data
print("\nTraining final model on full training data...")
final_model = build_combined_model()
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
final_model.fit([X_raw_remain, X_spec_remain], y_remain, epochs=10, batch_size=4, verbose=1)

# Evaluate on test set
test_loss, test_acc = final_model.evaluate([X_raw_test, X_spec_test], y_test)
print(f"\n✅ Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"Average Cross-Validation Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")

# Save model
final_model.save('hybrid_eeg_model.h5')

# Prediction Function
def predict_eeg(filepath):
    signal = np.loadtxt(filepath)
    if len(signal) == ORIGINAL_LENGTH:
        signal = signal[:-1]

    segments = np.split(signal, SEGMENTS_PER_FILE)
    all_preds = []

    for seg in segments:
        scaler = StandardScaler()
        seg_norm = scaler.fit_transform(seg.reshape(-1, 1)).flatten()
        spec = create_spectrogram(seg_norm)

        seg_input = np.expand_dims(seg_norm, axis=0)
        spec_input = np.expand_dims(spec, axis=(0, -1))

        pred = final_model.predict([seg_input, spec_input], verbose=0)
        all_preds.append(pred[0][0])

    final_score = np.mean(all_preds)
    print(f"\nPrediction Score: {final_score:.4f}")
    if final_score > 0.5:
        print("🚨 Epileptic Seizure Detected")
    else:
        print("✅ Normal Activity")

    return final_score

# Example usage
predict_eeg('/content/drive/MyDrive/data set/Bonn Univeristy Dataset/S/S036.txt')
