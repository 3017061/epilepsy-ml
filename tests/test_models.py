"""Unit tests for model creation and training."""

import pytest
from src.models.classical import make_classical_model, make_clustering_model
from src.models.deep import SimpleMLP
from src.data.loader import load_public_dataset, split_dataset


def test_make_random_forest():
    """Test creating a random forest classifier."""
    model = make_classical_model("random_forest")
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_make_logistic_regression():
    """Test creating a logistic regression classifier."""
    model = make_classical_model("logistic_regression")
    assert model is not None
    assert hasattr(model, "fit")


def test_make_gradient_boosting():
    """Test creating a gradient boosting classifier."""
    model = make_classical_model("gradient_boosting")
    assert model is not None
    assert hasattr(model, "fit")


def test_make_invalid_model():
    """Test that invalid model name raises error."""
    with pytest.raises(ValueError):
        make_classical_model("invalid_model")


def test_make_clustering_model():
    """Test creating a KMeans clustering model."""
    model = make_clustering_model(n_clusters=3)
    assert model is not None
    assert hasattr(model, "fit_predict")


def test_simple_mlp_creation():
    """Test creating a simple MLP."""
    model = SimpleMLP(input_dim=30, n_classes=2)
    assert model is not None
    assert hasattr(model, "forward")


def test_random_forest_training():
    """Test training a random forest on real data."""
    df, _ = load_public_dataset("breast_cancer")
    X_train, X_test, y_train, y_test = split_dataset(df)
    
    model = make_classical_model("random_forest")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    # Check that model achieves reasonable accuracy
    assert score > 0.8
