# EEG-Based Epileptic Seizure Detection Using PCNN–LSTM

A hybrid deep learning framework combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks for automatic detection of epileptic seizures from EEG signals. This project explores dual-path learning using both time-frequency representations (spectrograms, scalograms) and raw time-series data, enabling robust and accurate classification.

## 📘 Overview

Epilepsy affects over 70 million people worldwide, and EEG (electroencephalogram) signals are essential for its diagnosis. Manual interpretation is time-consuming and prone to errors. This study presents a PCNN–LSTM architecture that automates seizure detection using both spatial and temporal patterns in EEG data.

**Key Features:**
- Processes both raw 1D EEG and 2D spectrogram/scalogram representations.
- Integrates STFT and CWT techniques for time-frequency analysis.
- Parallel CNN–LSTM branches capture spatial and temporal dependencies.
- Achieved up to **95.83% test accuracy** and **AUC ≈ 0.99** on the Bonn EEG dataset.

---

## 🧠 Model Architecture

```csharp
         Raw EEG 1D Signal                 EEG as Spectrograms/Scalograms
                │                                      │
            [LSTM]                                 [CNN]
                │                                      │
        Temporal Features                    Spatial Features
                └────────────┬───────────────┘
                             ▼
                 Concatenated Feature Vector
                             ▼
                     Fully Connected Layers
                             ▼
                  Binary Output: Seizure / Non-Seizure
```


---

## 🗂 Dataset

- **Source**: [Bonn EEG Database](https://epilepsy.uni-freiburg.de/freiburg-seizure-prediction-project/eeg-database)
- 5 subsets: A (healthy awake), B (healthy closed eyes), C/D (epileptic, interictal), E (epileptic, ictal)
- 4096 samples per segment, sampled at 173.61 Hz
- Transformed to 256-sample segments → 6400 total EEG inputs

---

## ⚙️ Methodology

- **Preprocessing**: Segmentation, normalization (0–1 scale)
- **Transforms**:
  - **STFT** → Spectrograms (32x32 grayscale images)
  - **CWT** → Scalograms using Morlet wavelets (32x32)
- **Model**: Parallel CNN–LSTM fusion
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Evaluation**: 4-fold Cross-Validation and Hold-Out Split

---

## 💡 Future Work

- Replace LSTM with Transformer-based architectures
- Real-time deployment via MobileNet or TinyML
- Multiclass classification for various seizure types
- Integration with other modalities (e.g., MRI, heart rate)
- Improved explainability using SHAP, Grad-CAM

## 📎 Resources

- 🔗 [Final Paper PDF](./Epilepsy_Final.pdf)
- 📚 [Bonn EEG Dataset](https://epilepsy.uni-freiburg.de/freiburg-seizure-prediction-project/eeg-database)

---

## 📬 Contact

Feel free to reach out for collaboration or queries:

**Mintu Thomas**  
📧 mintuthomas.230023@gmail.com  
🔗 [GitHub](https://github.com/Mintu-2306)  
🔗 [LinkedIn](https://www.linkedin.com/in/mintu-thomas-v230623)

---

**Note**: This repository is intended for academic and research purposes only.
```
