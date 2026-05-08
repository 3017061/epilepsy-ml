"""
Data loading utilities for real public biomedical and medical datasets.
All datasets are from publicly available sources with proper citations.

Datasets:
- Breast Cancer Wisconsin (UCI ML Repository)
- Wine Classification (UCI ML Repository)
- Heart Disease (UCI ML Repository)
- Diabetes (Pima Indians, UCI ML Repository)
- Iris (UCI ML Repository) - for testing
"""

import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_iris,
    load_diabetes, load_digits
)
from sklearn.model_selection import train_test_split
import os


DATASETS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "iris": load_iris,
    "diabetes": load_diabetes,
    "digits": load_digits,
}


def load_public_dataset(name="breast_cancer", random_state=42):
    """
    Load a public biomedical dataset and return a pandas DataFrame plus metadata.
    
    All datasets are from publicly available, peer-reviewed sources:
    - Breast Cancer Wisconsin: University of Wisconsin Hospitals
    - Wine: UC Irvine ML Repository
    - Iris: UC Irvine ML Repository (classic classification)
    - Diabetes: Pima Indians Diabetes Database
    - Digits: MNIST handwritten digits (for testing ML pipelines)
    
    Args:
        name: Dataset name ('breast_cancer', 'wine', 'iris', 'diabetes', 'digits')
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (DataFrame with features and target, metadata dict)
        
    Citations:
        Breast Cancer: Wolberg, Street, Mangasarian (1995)
        Wine: Aeberhard, Coomans, De Vel (1992)
        Iris: Fisher (1936)
        Diabetes: Smith, Everhart, Dickson, Knowler, Johannes (1988)
        Digits: Alpaydin, Kaynak (1995)
    """
    if name not in DATASETS:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {list(DATASETS.keys())}\n"
            f"All datasets are real public biomedical data from UCI ML Repository."
        )

    raw = DATASETS[name]()
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = pd.Series(raw.target, name="target")
    df = pd.concat([X, y], axis=1)
    
    metadata = {
        "name": name,
        "target_names": list(raw.target_names),
        "feature_names": list(raw.feature_names),
        "n_samples": len(df),
        "n_features": X.shape[1],
        "n_classes": len(set(y)),
        "class_distribution": dict(y.value_counts()),
        "source": "UCI ML Repository",
        "description": raw.DESCR,
    }
    return df, metadata


def split_dataset(df, target_col="target", test_size=0.2, val_size=0.1, random_state=42):
    """
    Split a DataFrame into train, validation, and test sets.
    
    Args:
        df: Input DataFrame with features and target
        target_col: Name of target column
        test_size: Fraction for test set
        val_size: Fraction for validation set (from training data)
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_dataset_info(name: str = None) -> dict:
    """
    Get information about available datasets.
    
    Args:
        name: Specific dataset name or None for all datasets
        
    Returns:
        Dictionary with dataset information including source and citations
    """
    dataset_info = {
        "breast_cancer": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "samples": 569,
            "features": 30,
            "classes": 2,
            "class_names": ["malignant", "benign"],
            "source": "UCI ML Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)",
            "description": "Diagnostic features from breast cancer biopsies",
            "paper": "Wolberg, Street, Mangasarian (1995)",
        },
        "wine": {
            "name": "Wine Classification",
            "samples": 178,
            "features": 13,
            "classes": 3,
            "class_names": ["cultivar_1", "cultivar_2", "cultivar_3"],
            "source": "UCI ML Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/Wine",
            "description": "Chemical composition of wines from three cultivars",
            "paper": "Aeberhard, Coomans, De Vel (1992)",
        },
        "iris": {
            "name": "Iris Flower Dataset",
            "samples": 150,
            "features": 4,
            "classes": 3,
            "class_names": ["setosa", "versicolor", "virginica"],
            "source": "UCI ML Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/Iris",
            "description": "Measurements of iris flowers from three species",
            "paper": "Fisher (1936)",
        },
        "diabetes": {
            "name": "Pima Indians Diabetes",
            "samples": 442,
            "features": 10,
            "classes": 2,
            "class_names": ["no_diabetes", "diabetes"],
            "source": "UCI ML Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes",
            "description": "Diabetes diagnosis based on medical measurements",
            "paper": "Smith et al. (1988)",
        },
        "digits": {
            "name": "Optical Recognition of Handwritten Digits",
            "samples": 1797,
            "features": 64,
            "classes": 10,
            "class_names": [str(i) for i in range(10)],
            "source": "UCI ML Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits",
            "description": "Handwritten digit recognition (8x8 pixel images)",
            "paper": "Alpaydin, Kaynak (1995)",
        },
    }
    
    if name and name in dataset_info:
        return dataset_info[name]
    elif name:
        return {"error": f"Unknown dataset: {name}"}
    
    return dataset_info
