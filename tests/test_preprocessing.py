"""Unit tests for preprocessing pipelines."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.data.loader import load_public_dataset, split_dataset


def test_preprocessing_pipeline_creation():
    """Test that preprocessing pipeline is created correctly."""
    pipeline = build_preprocessing_pipeline(k_best=10)
    assert pipeline is not None
    assert len(pipeline.steps) == 4


def test_preprocessing_pipeline_fit():
    """Test fitting the preprocessing pipeline."""
    df, _ = load_public_dataset("breast_cancer")
    X_train, X_test, y_train, _ = split_dataset(df)
    
    pipeline = build_preprocessing_pipeline(k_best=10)
    X_transformed = pipeline.fit_transform(X_train, y_train)
    
    # Check output shape
    assert X_transformed.shape[0] == X_train.shape[0]
    assert X_transformed.shape[1] == 10  # k_best=10


def test_preprocessing_pipeline_transform():
    """Test transforming new data with fitted pipeline."""
    df, _ = load_public_dataset("breast_cancer")
    X_train, X_test, y_train, _ = split_dataset(df)
    
    pipeline = build_preprocessing_pipeline(k_best=15)
    pipeline.fit(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)
    
    # Check output shape matches test set size
    assert X_test_transformed.shape[0] == X_test.shape[0]
    assert X_test_transformed.shape[1] == 15


def test_preprocessing_output_type():
    """Test that preprocessing output is numeric."""
    df, _ = load_public_dataset("breast_cancer")
    X_train, _, y_train, _ = split_dataset(df)
    
    pipeline = build_preprocessing_pipeline(k_best=10)
    X_transformed = pipeline.fit_transform(X_train, y_train)
    
    # Check that output is numeric
    assert isinstance(X_transformed, np.ndarray)
    assert np.issubdtype(X_transformed.dtype, np.number)
