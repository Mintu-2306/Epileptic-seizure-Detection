# Hybrid model using CWT scalograms (Bonn dataset)

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import pywt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Constants
DATASET_PATH = '/content/drive/MyDrive/data set/Bonn Univeristy Dataset'
FOLDERS = ['F', 'N', 'O', 'S', 'Z']
LABELS = {'F': 0, 'N': 0, 'O': 0, 'Z': 0, 'S': 1}
ORIGINAL_LENGTH = 4097
NEW_LENGTH = 4096
SEGMENT_SIZE = 256
SEGMENTS_PER_FILE = NEW_LENGTH // SEGMENT_SIZE

# Storage
raw_segments = []
scalograms = []
labels = []

# ✅ Replaced STFT with CWT scalogram
def create_scalogram(segment):
    widths = np.arange(1, 129)
    cwtmatr, _ = pywt.cwt(segment, widths, 'morl')

    fig = plt.figure(figsize=(1.5, 1.5), dpi=50)
    canvas = FigureCanvas(fig)
    plt.axis('off')
    plt.imshow(np.abs(cwtmatr), aspect='auto', cmap='jet')
    canvas.draw()

    # Get width and height of the figure in pixels
    width, height = fig.canvas.get_width_height()

    X = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
    #X = X.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    X = X.reshape(height, width, 4)
    plt.close(fig)

    gray = np.mean(X, axis=-1)
    if gray.shape != (77, 75):
        gray = np.resize(gray, (77, 75))
    return gray

# Preprocessing
for folder in FOLDERS:
    folder_path = os.path.join(DATASET_PATH, folder)
    label = LABELS[folder]

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') or filename.endswith('.TXT'):
            file_path = os.path.join(folder_path, filename)
            signal = np.loadtxt(file_path)

            if len(signal) == ORIGINAL_LENGTH:
                signal = signal[:-1]

            segments = np.split(signal, SEGMENTS_PER_FILE)

            for seg in segments:
                scaler = StandardScaler()
                seg_normalized = scaler.fit_transform(seg.reshape(-1, 1)).flatten()

                # ✅ Using scalogram instead of spectrogram
                scalogram = create_scalogram(seg_normalized)

                raw_segments.append(seg_normalized)
                scalograms.append(scalogram)
                labels.append(label)

print("Preprocessing done!")

# Convert to arrays
X_raw = np.array(raw_segments)
X_spec = np.array(scalograms)  # Same variable name retained
y = np.array(labels)

X_spec = np.expand_dims(X_spec, axis=-1)

print(f"X_raw shape: {X_raw.shape}, X_spec shape: {X_spec.shape}, y shape: {y.shape}")

# Split data
X_raw_train, X_raw_temp, X_spec_train, X_spec_temp, y_train, y_temp = train_test_split(
    X_raw, X_spec, y, test_size=0.3, random_state=42, stratify=y)
X_raw_val, X_raw_test, X_spec_val, X_spec_test, y_val, y_test = train_test_split(
    X_raw_temp, X_spec_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {X_raw_train.shape}, Val: {X_raw_val.shape}, Test: {X_raw_test.shape}")

# Model definition
def build_combined_model():
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

    combined = layers.concatenate([x1, x2])
    combined = layers.Dense(64, activation='sigmoid')(combined)
    combined = layers.Dense(32, activation='sigmoid')(combined)
    combined = layers.Dense(1, activation='sigmoid')(combined)

    model = models.Model(inputs=[input_raw, input_spec], outputs=combined)
    return model

# Build and compile
model = build_combined_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train
history = model.fit(
    [X_raw_train, X_spec_train], y_train,
    validation_data=([X_raw_val, X_spec_val], y_val),
    epochs=10,
    batch_size=4
)

# Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training History')
plt.show()

# Evaluate
test_loss, test_acc = model.evaluate([X_raw_test, X_spec_test], y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Save model
model.save('my_eeg_model.h5')

# Load model (if needed)
model = tf.keras.models.load_model('/content/my_eeg_model.h5')

# Predict Function
def predict_eeg(filepath):
    signal = np.loadtxt(filepath)

    if len(signal) == ORIGINAL_LENGTH:
        signal = signal[:-1]

    segments = np.split(signal, SEGMENTS_PER_FILE)
    all_preds = []

    for seg in segments:
        scaler = StandardScaler()
        seg_norm = scaler.fit_transform(seg.reshape(-1, 1)).flatten()

        spec = create_scalogram(seg_norm)  # ✅ Updated to use scalogram

        seg_norm_input = np.expand_dims(seg_norm, axis=0)
        spec_input = np.expand_dims(spec, axis=(0, -1))

        pred = model.predict([seg_norm_input, spec_input], verbose=0)
        all_preds.append(pred[0][0])

    final_score = np.mean(all_preds)

    print(f"Average prediction score: {final_score:.4f}")
    if final_score > 0.5:
        print("Prediction: 🚨 Epileptic Seizure Detected")
    else:
        print("Prediction: ✅ Non-Seizure (Normal Activity)")

    return final_score

# Example
predict_eeg('/content/drive/MyDrive/data set/Bonn Univeristy Dataset/S/S036.txt')
