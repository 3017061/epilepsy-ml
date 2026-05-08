"""Unit tests for data loading and preprocessing."""

import pytest
import pandas as pd
from src.data.loader import load_public_dataset, split_dataset


def test_load_breast_cancer_dataset():
    """Test loading the breast cancer dataset."""
    df, metadata = load_public_dataset("breast_cancer")
    assert df.shape[0] > 0
    assert "target" in df.columns
    assert metadata["name"] == "breast_cancer"
    assert "malignant" in metadata["target_names"]


def test_load_wine_dataset():
    """Test loading the wine dataset."""
    df, metadata = load_public_dataset("wine")
    assert df.shape[0] > 0
    assert metadata["name"] == "wine"


def test_load_invalid_dataset():
    """Test that invalid dataset name raises error."""
    with pytest.raises(ValueError):
        load_public_dataset("invalid_name")


def test_split_dataset():
    """Test train-test split functionality."""
    df, _ = load_public_dataset("breast_cancer")
    X_train, X_test, y_train, y_test = split_dataset(df)
    
    # Check shapes
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    
    # Check stratification (roughly equal class distribution)
    train_ratio = y_train.sum() / len(y_train)
    test_ratio = y_test.sum() / len(y_test)
    assert abs(train_ratio - test_ratio) < 0.1


def test_split_dataset_custom_ratio():
    """Test train-test split with custom ratio."""
    df, _ = load_public_dataset("breast_cancer")
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.3)
    
    # Check that test size is approximately 30%
    test_ratio = len(X_test) / (len(X_train) + len(X_test))
    assert abs(test_ratio - 0.3) < 0.05
