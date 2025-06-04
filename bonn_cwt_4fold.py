import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, models, Input
import tensorflow as tf

# Constants
DATASET_PATH = '/content/drive/MyDrive/data set/Bonn Univeristy Dataset'
FOLDERS = ['F', 'N', 'O', 'S', 'Z']
LABELS = {'F': 0, 'N': 0, 'O': 0, 'Z': 0, 'S': 1}
ORIGINAL_LENGTH = 4097
NEW_LENGTH = 4096
SEGMENT_SIZE = 256
SEGMENTS_PER_FILE = NEW_LENGTH // SEGMENT_SIZE

# ✅ Create scalogram from a segment
def create_scalogram(segment):
    widths = np.arange(1, 129)
    cwtmatr, _ = pywt.cwt(segment, widths, 'morl')

    fig = plt.figure(figsize=(1.5, 1.5), dpi=50)
    canvas = FigureCanvas(fig)
    plt.axis('off')
    plt.imshow(np.abs(cwtmatr), aspect='auto', cmap='jet')
    canvas.draw()
    width, height = fig.canvas.get_width_height()
    X = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
    X = X.reshape(height, width, 4)
    plt.close(fig)
    gray = np.mean(X, axis=-1)
    if gray.shape != (77, 75):
        gray = np.resize(gray, (77, 75))
    return gray

# ✅ Load data
raw_segments, scalograms, labels = [], [], []

for folder in FOLDERS:
    label = LABELS[folder]
    folder_path = os.path.join(DATASET_PATH, folder)
    for file in os.listdir(folder_path):
        if file.endswith('.txt') or file.endswith('.TXT'):
            signal = np.loadtxt(os.path.join(folder_path, file))
            if len(signal) == ORIGINAL_LENGTH:
                signal = signal[:-1]
            segments = np.split(signal, SEGMENTS_PER_FILE)
            for seg in segments:
                seg = StandardScaler().fit_transform(seg.reshape(-1, 1)).flatten()
                spec = create_scalogram(seg)
                raw_segments.append(seg)
                scalograms.append(spec)
                labels.append(label)

print("Preprocessing done!")

# ✅ Convert to arrays
X_raw = np.array(raw_segments)
X_spec = np.expand_dims(np.array(scalograms), axis=-1)
y = np.array(labels)

print(f"X_raw: {X_raw.shape}, X_spec: {X_spec.shape}, y: {y.shape}")

# ✅ Define model
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

# ✅ 4-Fold CV
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
fold = 1
acc_per_fold = []

for train_idx, test_idx in kf.split(X_raw, y):
    print(f"\n--- Fold {fold} ---")

    X_raw_train, X_raw_test = X_raw[train_idx], X_raw[test_idx]
    X_spec_train, X_spec_test = X_spec[train_idx], X_spec[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = build_combined_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        [X_raw_train, X_spec_train], y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=4,
        verbose=1
    )

    scores = model.evaluate([X_raw_test, X_spec_test], y_test, verbose=0)
    acc_per_fold.append(scores[1])
    print(f"✅ Fold {fold} Accuracy: {scores[1]*100:.2f}%")

    fold += 1

print(f"\nAverage Accuracy: {np.mean(acc_per_fold)*100:.2f}%")

# ✅ Save final model (last fold)
model.save('my_eeg_model.h5')

# ✅ Predict function (unchanged)
def predict_eeg(filepath):
    signal = np.loadtxt(filepath)
    if len(signal) == ORIGINAL_LENGTH:
        signal = signal[:-1]
    segments = np.split(signal, SEGMENTS_PER_FILE)
    all_preds = []
    for seg in segments:
        seg_norm = StandardScaler().fit_transform(seg.reshape(-1, 1)).flatten()
        spec = create_scalogram(seg_norm)
        seg_input = np.expand_dims(seg_norm, axis=0)
        spec_input = np.expand_dims(spec, axis=(0, -1))
        pred = model.predict([seg_input, spec_input], verbose=0)
        all_preds.append(pred[0][0])
    final_score = np.mean(all_preds)
    print(f"Average prediction score: {final_score:.4f}")
    if final_score > 0.5:
        print("Prediction: 🚨 Epileptic Seizure Detected")
    else:
        print("Prediction: ✅ Non-Seizure (Normal Activity)")
    return final_score

# ✅ Example
predict_eeg('/content/drive/MyDrive/data set/Bonn Univeristy Dataset/S/S036.txt')
