# Epilepsy ML Pipeline - Implementation Guide

A complete, production-ready machine learning pipeline for epilepsy detection using both classical and deep learning models.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Results & Evaluation](#results--evaluation)

## 🏗️ Architecture Overview

This pipeline implements a complete end-to-end machine learning workflow:

```
Data Loading
    ↓
Feature Engineering (Advanced Feature Extraction)
    ↓
Preprocessing (Filtering, Scaling, Selection)
    ↓
├─→ Classical ML Models (RF, GB, SVM, KNN, etc.)
│   └─→ Evaluation & Metrics
│
└─→ Deep Learning Models (MLP, CNN, Attention)
    └─→ Evaluation & Metrics
```

## ✨ Features

### 1. **Data Loading & Management**
- Support for public biomedical datasets (Breast Cancer, Wine)
- Automatic feature extraction from tabular data
- Dataset metadata and information utilities

### 2. **Advanced Feature Engineering**
- **Time-Domain Features**: Mean, std, skewness, kurtosis, RMS, energy, entropy
- **Frequency-Domain Features**: FFT, power spectral density, spectral centroid
- **Wavelet Features**: Multi-level wavelet decomposition
- **Statistical Features**: Hjorth parameters, zero crossings
- **Interaction Features**: Polynomial and multiplicative interactions
- **Ratio Features**: Feature engineering through domain-specific ratios

### 3. **EEG-Specific Preprocessing**
- Bandpass filtering (customizable frequency bands)
- Notch filtering for power line interference (50/60 Hz)
- Artifact removal using threshold/IQR methods
- Common Average Reference (CAR)
- Adaptive feature selection based on data characteristics

### 4. **Classical ML Models**
- Logistic Regression (standard & SAGA solver)
- Decision Trees
- Random Forests (200 estimators, optimized)
- Gradient Boosting (150 estimators)
- Support Vector Machines (RBF, Linear, Polynomial kernels)
- K-Nearest Neighbors (uniform & distance-weighted)
- Naive Bayes
- AdaBoost
- **Ensemble Methods**: Voting Classifier, Stacking Classifier

### 5. **Deep Learning Models**
- Simple MLP (128→64→classes)
- Deep MLP with batch normalization (256→128→64→32)
- 1D Convolutional Neural Network
- Attention-based MLP with self-attention blocks
- Residual MLP with skip connections
- Early stopping and learning rate optimization

### 6. **Comprehensive Evaluation**
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score, MCC, Kappa
- **Medical Metrics**: Sensitivity, Specificity, PPV, NPV
- **ROC-AUC & PR Curves** for binary classification
- Confusion matrices with visualization
- Cross-validation evaluation
- Per-class and weighted metrics

### 7. **Visualization Suite**
- Correlation heatmaps
- PCA & t-SNE embeddings
- Feature importance plots
- Class distribution charts
- Training history plots
- ROC and Precision-Recall curves
- EEG signal visualization

## 🚀 Installation

### Requirements
- Python 3.8+
- Virtual environment recommended

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd epilepsy-ml-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
numpy>=1.20
pandas>=1.3
scikit-learn>=1.0
scipy>=1.7
matplotlib>=3.4
seaborn>=0.11
plotly>=5.0
torch>=1.9
pywavelets>=1.1
```

## 🎯 Quick Start

### Run Full Pipeline

```bash
# Run complete ML pipeline (classical + deep learning)
python experiments/run_pipeline.py
```

### Run Classical ML Only

```bash
# Evaluate multiple classical ML models
python experiments/run_classical.py
```

### Python API

```python
from src.data.loader import load_public_dataset, split_dataset
from src.features.engineer import enrich_features
from src.models.classical import make_classical_model
from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.training.train import train_sklearn
from src.evaluation.metrics import classification_summary

# Load data
df, metadata = load_public_dataset("breast_cancer")
X_train, X_test, y_train, y_test = split_dataset(df)

# Feature engineering
X_train = enrich_features(X_train, include_statistical=True)
X_test = enrich_features(X_test, include_statistical=True)

# Preprocessing
preprocess = build_preprocessing_pipeline(k_best=20)
preprocess.fit(X_train, y_train)
X_train = preprocess.transform(X_train)
X_test = preprocess.transform(X_test)

# Train model
model = make_classical_model("random_forest")
model = train_sklearn(model, X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
summary = classification_summary(y_test, y_pred)
print(f"Accuracy: {summary['accuracy']:.4f}")
```

## 📦 Pipeline Components

### 1. Data Loading (`src/data/loader.py`)

```python
# Load public dataset
df, metadata = load_public_dataset("breast_cancer")

# Train/val/test split
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    df, test_size=0.2, val_size=0.1
)
```

### 2. Feature Engineering (`src/features/engineer.py`)

```python
from src.features.engineer import enrich_features, extract_eeg_features_tabular
from src.features.extractor import SignalFeatureExtractor

# Basic feature enrichment
X_engineered = enrich_features(X, include_statistical=True, include_interactions=False)

# Advanced feature extraction from signals
extractor = SignalFeatureExtractor()
features = extractor.extract_all_features(signal, fs=256, include_wavelets=True)

# EEG feature extraction
df_features, feature_names = extract_eeg_features_tabular(eeg_data, fs=256)
```

### 3. Preprocessing (`src/preprocessing/pipeline.py`)

```python
from src.preprocessing.pipeline import build_preprocessing_pipeline, EEGPreprocessor

# Standard preprocessing with feature selection
pipeline = build_preprocessing_pipeline(k_best=20, scale_method="standard")
X_transformed = pipeline.fit_transform(X_train, y_train)

# PCA-based preprocessing
pca_pipeline = build_pca_pipeline(n_components=20)
X_pca = pca_pipeline.fit_transform(X_train)

# EEG-specific preprocessing
eeg_preprocessor = EEGPreprocessor(fs=256, lowcut=1.0, highcut=50.0)
signal_clean = eeg_preprocessor.preprocess(raw_signal)
```

### 4. Classical Models (`src/models/classical.py`)

```python
from src.models.classical import make_classical_model, make_ensemble_model

# Single models
rf_model = make_classical_model("random_forest")
svm_model = make_classical_model("svm")
gb_model = make_classical_model("gradient_boosting")

# Ensemble
ensemble = make_ensemble_model()
```

### 5. Deep Learning (`src/models/deep.py`)

```python
from src.models.deep import SimpleMLP, DeepMLP, ConvNet1D, AttentionMLP

# Simple MLP
model = SimpleMLP(input_dim=100, n_classes=2)

# Advanced architectures
model = DeepMLP(input_dim=100, n_classes=2)
model = AttentionMLP(input_dim=100, n_classes=2)
model = ConvNet1D(input_channels=22, n_classes=2, seq_length=256)
```

### 6. Training (`src/training/train.py`)

```python
from src.training.train import train_sklearn, train_torch, hyperparameter_tuning

# Classical ML training
model = train_sklearn(model, X_train, y_train)

# Deep learning training
model, history = train_torch(
    model, X_train, y_train, X_val, y_val,
    epochs=50, lr=1e-3, batch_size=32,
    early_stopping=True, patience=5
)

# Hyperparameter tuning
best_model, best_params, cv_results = hyperparameter_tuning(
    model, param_grid, X_train, y_train, cv=5
)
```

### 7. Evaluation (`src/evaluation/metrics.py`)

```python
from src.evaluation.metrics import (
    classification_summary,
    binary_classification_summary,
    evaluate_model_performance,
    plot_confusion_matrix,
    plot_roc_curve
)

# Classification summary
summary = classification_summary(y_true, y_pred, y_proba)

# Binary classification with medical metrics
summary = binary_classification_summary(y_true, y_pred, y_proba)
print(f"Sensitivity: {summary['sensitivity']}")
print(f"Specificity: {summary['specificity']}")

# Visualization
fig = plot_confusion_matrix(summary['confusion_matrix'])
fig = plot_roc_curve(y_true, y_proba)
```

## 💡 Usage Examples

### Example 1: Complete Pipeline with Breast Cancer Dataset

```python
from experiments.run_pipeline import run_full_pipeline

# Run complete pipeline
results = run_full_pipeline(
    dataset_name="breast_cancer",
    use_eeg=False
)

# Results contain classical ML and deep learning results
print("Classical ML Results:", results['classical_ml'])
print("Deep Learning Results:", results['deep_learning'])
```

### Example 3: Advanced Feature Extraction

```python
from src.features.extractor import SignalFeatureExtractor
import numpy as np

# Initialize extractor
extractor = SignalFeatureExtractor()

# Extract features
signal = np.random.randn(256)  # 256-point signal
features = extractor.extract_all_features(
    signal,
    fs=256,  # 256 Hz sampling rate
    include_wavelets=True
)

# Get feature names
feature_names = extractor.get_feature_names(include_wavelets=True)

print(f"Extracted {len(features)} features")
print(f"First 5 features: {features[:5]}")
```

### Example 4: Hyperparameter Tuning

```python
from src.training.train import hyperparameter_tuning
from src.models.classical import make_classical_model

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
}

# Create base model
model = make_classical_model("random_forest")

# Tune hyperparameters
best_model, best_params, cv_results = hyperparameter_tuning(
    model, param_grid, X_train, y_train, cv=5
)

print(f"Best parameters: {best_params}")
```

## 🔬 Advanced Features

### 1. EEG Signal Filtering

```python
from src.preprocessing.eeg_filters import apply_eeg_filter, apply_notch_filter

# Bandpass filter (1-50 Hz)
filtered = apply_eeg_filter(
    signal,
    fs=256,
    filter_type="bandpass",
    lowcut=1.0,
    highcut=50.0,
    order=4
)

# Notch filter (50 Hz power line)
notched = apply_notch_filter(signal, fs=256, freq=50.0)
```

### 2. Cross-Validation

```python
from src.training.train import cross_val_evaluate

cv_results = cross_val_evaluate(model, X_train, y_train, cv=5)
print(f"Mean CV Accuracy: {cv_results['test_accuracy'].mean():.4f}")
```

### 3. Model Comparison

```python
import pandas as pd

results_summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['test_accuracy'] for r in results.values()],
    'F1-Score': [r['test_f1'] for r in results.values()],
})

print(results_summary.sort_values('F1-Score', ascending=False))
```

## 📊 Results & Evaluation

### Expected Performance

On public datasets (e.g., Breast Cancer Wisconsin):
- **Classical ML**: 95-98% accuracy
- **Deep Learning**: 94-97% accuracy
- **Ensemble**: 96-99% accuracy

### Medical-Specific Metrics

For epilepsy detection, focus on:
- **Sensitivity** (True Positive Rate): Detection of actual seizures
- **Specificity** (True Negative Rate): Avoiding false alarms
- **PPV/NPV**: Clinical relevance
- **ROC-AUC**: Overall discriminative ability

### Output Files

Results are saved to `experiments/results/`:
- `pipeline_results_*.json` - Complete pipeline results
- `classical_results_*.json` - Classical ML results

Figures saved to `experiments/figures/`:
- `correlation_matrix_*.png` - Feature correlations
- `pca_embedding_*.png` - PCA visualization
- `class_distribution_*.png` - Target distribution
- `feature_importance_*.png` - Feature rankings

## 🛠️ Customization

### Add New Dataset

```python
# In src/data/loader.py
DATASETS["my_dataset"] = my_dataset_loader_function
```

### Add New Model

```python
# In src/models/classical.py
def make_classical_model(name="my_model"):
    if name == "my_model":
        return MyCustomModel()
```

### Custom Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline

custom_pipeline = Pipeline([
    ("custom_step_1", MyTransformer1()),
    ("custom_step_2", MyTransformer2()),
])
```

## 📝 References

- Research paper: `papers/The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy.pdf`
- Scikit-learn Documentation: https://scikit-learn.org/
- PyTorch Documentation: https://pytorch.org/
- SciPy Signal Processing: https://docs.scipy.org/doc/scipy/reference/signal.html

## 📄 License

[Add your license information here]

## 👥 Contributors

- Saniya Vaish (3017061@gmail.com)

---

**Last Updated**: May 2026
**Status**: Production-Ready ✅