Contactless Breathing Rate Prediction Using Wi-Fi CSI

This repository contains the modular, production-ready implementation of a
deep learning system that estimates a person's respiratory rate — in
**Breaths Per Minute (BPM)** — from **Wi-Fi Channel State Information (CSI)**,
with no physical contact with the subject. It reframes respiratory
monitoring as a **continuous regression problem** (rather than discrete
BPM classification), allowing the model to interpolate fractional
breathing rates (e.g. 14.5 BPM) instead of being limited to the exact BPM
values seen during training.

## Motivation

Contact-based respiratory monitors can be uncomfortable and can alter a
patient's natural breathing baseline. Camera- and microphone-based
contactless alternatives are sensitive to lighting, line-of-sight, and
background noise. Radio-frequency sensing over Commercial Off-The-Shelf
(COTS) Wi-Fi hardware avoids these constraints: as a person's chest
expands and contracts while breathing, it produces measurable perturbations
in the amplitude and phase of Wi-Fi signals reflected through the
environment. This project extracts those perturbations from CSI and maps
them directly to BPM with a hybrid deep learning model.

## Dataset

The model is trained and evaluated on a public dataset released by the
**National Institute of Standards and Technology (NIST)**, collected using a
calibrated **RespiPro** breathing manikin programmed to reproduce a range of
clinical breathing rates and tidal volumes.

| Pattern | BPM | Max Inspiratory/Expiratory TV (mL) |
|---------|-----|--------------------------------------|
| #1 | 25 | 319 |
| #2 | 6  | 705 |
| #3 | 18 | 488 |
| #4 | 18 | 171 |
| #5 | 22 | 651 |
| #6 | 15 | 587 |
| #7 | 28 | 442 |
| #8 | 19 | 513 |
| #9 | 25 | 463 |

This dataset is publicly available online: **https://doi.org/10.18434/mds2-2963**
See the [References](#references) section below for the full citation.

## Results

Performance of the trained model on the 20% held-out test set:

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | 2.0564 BPM |
| Root Mean Squared Error (RMSE) | 2.9872 |
| Mean Squared Error (MSE) | 8.9233 |
| Coefficient of Determination (R²) | 0.8195 |

**Tolerance-based accuracy** — the percentage of predictions within a given
margin of the true BPM:

| Tolerance | Accuracy |
|-----------|----------|
| ±1 BPM | 41.66% |
| ±2 BPM | 62.20% |
| ±3 BPM | 75.38% |
| ±4 BPM | 83.90% |
| ±5 BPM | 90.73% |

An R² of 0.8195 means the extracted spatio-temporal CSI features account for
roughly 82% of the variance in true respiratory rate, and over 90% of
predictions fall within a clinically acceptable ±5 BPM margin — indicating
strong reliability for trend-level vital sign monitoring. Residual analysis
shows errors are approximately normally distributed around zero, i.e. the
model does not systematically over- or under-predict BPM.

## Project Structure

```
csi-breathing-rate/
├── src/
│   ├── data_processing.py   # CSI loading, amplitude calc, filtering, normalization, PCA, windowing
│   ├── model.py              # CNN-BiLSTM model definition
│   ├── train.py               # Training loop (callbacks + model.fit)
│   └── evaluate.py            # Metrics + diagnostic plots
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data Processing Pipeline

Raw CSI logs (real and imaginary components) for each experiment are
converted into fixed-shape training windows through the following steps:

1. **Amplitude extraction** — the CSI amplitude is computed from the real
   and imaginary tensors using the standard magnitude formula
   `X = sqrt(real² + imag²)`.
2. **Filtering and normalization** — sub-carriers with near-zero variance
   (standard deviation ≤ 1e-6), corresponding to dead channels or static
   reflections, are removed. Remaining features are Z-score normalized
   (zero mean, unit variance) to stabilize gradient descent.
3. **Dimensionality reduction (PCA)** — Principal Component Analysis
   reduces the normalized amplitude matrix to at most 20 principal
   components, capturing the dominant variance associated with human
   respiration. Observations with fewer than 20 valid features are
   zero-padded so every sample has a consistent shape.
4. **Temporal windowing** — the continuous time series is sliced into
   fixed-length windows (size 100, step 15) suitable for a recurrent
   network, using a sliding-window algorithm.
5. **Dataset splitting** — all windows and their ground-truth BPM labels are
   stacked and split into an 80% training set and a 20% hold-out test set.

### 2. Hybrid CNN-BiLSTM Architecture

The model treats BPM estimation as continuous regression, combining local
feature extraction with long-range temporal modeling:

- **CNN block** (spatial feature extraction):
  - `Conv1D` — 256 filters, kernel size 5, ReLU → `BatchNormalization` → `MaxPooling1D` (pool size 2)
  - `Conv1D` — 128 filters, kernel size 3, ReLU → `BatchNormalization` → `MaxPooling1D` (pool size 2)
- **RNN block** (bidirectional temporal modeling):
  - `Bidirectional(LSTM(128))`, returning full sequence → `Dropout(0.3)`
  - `Bidirectional(LSTM(64))`, returning final hidden state → `Dropout(0.3)`
- **Dense regression head**:
  - `Dense(64, relu)` → `BatchNormalization` → `Dense(32, relu)` → `Dense(1, linear)` (predicted BPM)

### 3. Training Configuration

- **Optimizer / loss**: Adam, initial learning rate `1e-4`; MSE loss with MAE tracked as a metric.
- **ReduceLROnPlateau**: halves the learning rate if validation loss does not improve for 5 epochs (lower bound `1e-6`).
- **EarlyStopping**: stops training if validation loss plateaus for 25 epochs, restoring the best-performing weights.
- Max 200 epochs, batch size 32, 15% validation split.

## Setup

Requires Python 3.9+.

```bash
git clone <this-repo-url>
cd breathesmart
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Layout

Point the scripts at the root of the NIST CSI dataset, arranged as:

```
<data_dir>/
└── Figure_XX/
    └── BreathingPattern_YY/
        └── config0001/
            ├── config0001.csv
            ├── config0001_csi_real_log.csv
            └── config0001_csi_imag_log.csv
```

## Usage

### Training

```bash
python -m src.train \
    --data_dir /path/to/dataset \
    --output_dir artifacts \
    --epochs 200 \
    --batch_size 32
```

Saves to `--output_dir`:
- `breathesmart_model.keras` — the trained model
- `history.json` — per-epoch training/validation loss and MAE
- `test_data.npz` — the held-out `X_test`, `y_test` arrays

### Evaluation

```bash
python -m src.evaluate --output_dir artifacts
```

Prints MAE / MSE / RMSE / R² and tolerance-based accuracy (±1–±5 BPM), and
saves diagnostic plots to `--output_dir`:
- `loss_curve.png` — training vs. validation loss over epochs
- `actual_vs_predicted.png` — scatter plot against the ideal `y = x` line
- `error_distribution.png` — histogram of residual errors

Pass `--show_plots` to also display the plots interactively.

## Discussion

Framing BPM estimation as regression rather than classification lets the
model predict fractional breathing rates it never saw as an explicit label
during training, giving a smoother picture of a person transitioning
between physiological states (e.g. rest to mild exertion). The ~9% of
predictions falling outside the ±5 BPM tolerance likely correspond to edge
cases in the dataset — e.g. windows captured under heavy simulated signal
attenuation or highly irregular breathing patterns with a degraded
signal-to-noise ratio.

## References

This implementation builds on the CSI-based respiratory sensing approach
described in [1], and is trained on the publicly available dataset in [2].

[1] S. Mosleh, J. B. Coder, C. G. Scully, K. Forsyth and M. O. A. Kalaa,
"Monitoring Respiratory Motion With Wi-Fi CSI: Characterizing Performance
and the BreatheSmart Algorithm," in *IEEE Access*, vol. 10,
pp. 131932–131951, 2022, doi: 10.1109/ACCESS.2022.3230003.

[2] J. Coder, S. Mosleh, M. K. Forsyth, C. Scully, M. O. Al Kalaa,
"Raw data used to investigate the use of Wi-Fi signals to monitor human
respiratory motion," National Institute of Standards and Technology, 2023.
https://doi.org/10.18434/mds2-2963 (Version 1.0)
