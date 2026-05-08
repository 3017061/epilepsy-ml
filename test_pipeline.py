"""
Pipeline validation and quick test script.
Verifies all components are working correctly.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...", end=" ")
    try:
        import numpy
        import pandas
        import sklearn
        import scipy
        import matplotlib
        import torch
        import pywt
        print("✓")
        return True
    except ImportError as e:
        print(f"✗ {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...", end=" ")
    try:
        from src.data.loader import load_public_dataset, split_dataset, get_dataset_info
        
        # Test public dataset
        df, metadata = load_public_dataset("breast_cancer")
        assert df.shape[0] > 0
        assert "target" in df.columns
        
        # Test dataset info
        info = get_dataset_info("breast_cancer")
        assert "n_samples" in info
        
        # Test split
        X_train, X_test, y_train, y_test = split_dataset(df)
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_feature_extraction():
    """Test feature extraction functionality."""
    print("Testing feature extraction...", end=" ")
    try:
        from src.features.extractor import SignalFeatureExtractor
        
        extractor = SignalFeatureExtractor()
        signal = np.random.randn(256)
        
        # Test time domain
        features = extractor.time_domain_features(signal)
        assert len(features) > 0
        
        # Test all features
        all_features = extractor.extract_all_features(signal, include_wavelets=False)
        assert len(all_features) > 0
        
        # Test feature names
        names = extractor.get_feature_names(include_wavelets=False)
        assert len(names) > 0
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("Testing preprocessing...", end=" ")
    try:
        from src.preprocessing.pipeline import build_preprocessing_pipeline
        from src.data.loader import load_public_dataset, split_dataset
        from src.features.engineer import enrich_features
        
        df, _ = load_public_dataset("breast_cancer")
        X_train, X_test, y_train, _ = split_dataset(df)
        
        X_train = enrich_features(X_train)
        X_test = enrich_features(X_test)
        
        pipeline = build_preprocessing_pipeline(k_best=10)
        X_transformed = pipeline.fit_transform(X_train, y_train)
        
        assert X_transformed.shape[0] == X_train.shape[0]
        assert X_transformed.shape[1] == 10
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_classical_models():
    """Test classical ML models."""
    print("Testing classical models...", end=" ")
    try:
        from src.models.classical import make_classical_model, get_model_list
        from src.training.train import train_sklearn
        from src.data.loader import load_public_dataset, split_dataset
        from src.features.engineer import enrich_features
        from src.preprocessing.pipeline import build_preprocessing_pipeline
        
        # Load and preprocess
        df, _ = load_public_dataset("breast_cancer")
        X_train, X_test, y_train, y_test = split_dataset(df)
        X_train = enrich_features(X_train)
        X_test = enrich_features(X_test)
        
        pipeline = build_preprocessing_pipeline(k_best=15)
        pipeline.fit(X_train, y_train)
        X_train_t = pipeline.transform(X_train)
        X_test_t = pipeline.transform(X_test)
        
        # Test a few models
        for model_name in ["random_forest", "svm", "logistic_regression"]:
            model = make_classical_model(model_name)
            model = train_sklearn(model, X_train_t, y_train)
            y_pred = model.predict(X_test_t)
            assert len(y_pred) == len(y_test)
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_deep_learning():
    """Test deep learning models."""
    print("Testing deep learning...", end=" ")
    try:
        import torch
        from src.models.deep import SimpleMLP, DeepMLP, get_device, count_parameters
        
        device = get_device()
        
        # Create models
        model1 = SimpleMLP(input_dim=100, n_classes=2)
        model2 = DeepMLP(input_dim=100, n_classes=2)
        
        # Count parameters
        params1 = count_parameters(model1)
        params2 = count_parameters(model2)
        
        assert params1 > 0
        assert params2 > 0
        
        # Test forward pass
        x = torch.randn(32, 100).to(device)
        model1 = model1.to(device)
        model2 = model2.to(device)
        
        out1 = model1(x)
        out2 = model2(x)
        
        assert out1.shape == (32, 2)
        assert out2.shape == (32, 2)
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_evaluation():
    """Test evaluation metrics."""
    print("Testing evaluation metrics...", end=" ")
    try:
        from src.evaluation.metrics import (
            classification_summary,
            binary_classification_summary,
            evaluate_model_performance
        )
        import numpy as np
        
        # Create dummy data
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        
        # Test classification summary
        summary = classification_summary(y_true, y_pred)
        assert "accuracy" in summary
        assert "f1_score" in summary
        
        # Test binary classification summary
        summary_binary = binary_classification_summary(y_true, y_pred)
        assert "sensitivity" in summary_binary
        assert "specificity" in summary_binary
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_visualization():
    """Test visualization functions."""
    print("Testing visualization...", end=" ")
    try:
        from src.visualization.plots import (
            plot_correlation_matrix,
            plot_pca_embedding,
            plot_class_distribution
        )
        from src.data.loader import load_public_dataset, split_dataset
        import matplotlib.pyplot as plt
        
        df, _ = load_public_dataset("breast_cancer")
        X_train, _, y_train, _ = split_dataset(df)
        
        # Test plots (they create matplotlib figures)
        fig1 = plot_correlation_matrix(X_train.iloc[:, :10])
        fig2 = plot_pca_embedding(X_train.values, y_train.values)
        fig3 = plot_class_distribution(y_train.values)
        
        plt.close('all')
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_full_pipeline():
    """Test complete pipeline."""
    print("Testing full pipeline...", end=" ")
    try:
        from src.data.loader import load_public_dataset, split_dataset
        from src.features.engineer import enrich_features
        from src.models.classical import make_classical_model
        from src.preprocessing.pipeline import build_preprocessing_pipeline
        from src.training.train import train_sklearn
        from src.evaluation.metrics import classification_summary
        
        # Load and preprocess
        df, metadata = load_public_dataset("breast_cancer")
        X_train, X_test, y_train, y_test = split_dataset(df)
        
        X_train = enrich_features(X_train, include_statistical=True)
        X_test = enrich_features(X_test, include_statistical=True)
        
        pipeline = build_preprocessing_pipeline(k_best=20)
        pipeline.fit(X_train, y_train)
        X_train = pipeline.transform(X_train)
        X_test = pipeline.transform(X_test)
        
        # Train and evaluate
        model = make_classical_model("random_forest")
        model = train_sklearn(model, X_train, y_train)
        y_pred = model.predict(X_test)
        
        summary = classification_summary(y_test, y_pred, metadata.get("target_names"))
        
        assert summary["accuracy"] > 0
        assert summary["f1_score"] > 0
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("EPILEPSY ML PIPELINE - VALIDATION TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        test_imports,
        test_data_loading,
        test_feature_extraction,
        test_preprocessing,
        test_classical_models,
        test_deep_learning,
        test_evaluation,
        test_visualization,
        test_full_pipeline,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("="*70)
        print("\n✓ Pipeline is ready to use!")
        print("\nNext steps:")
        print("  1. Run full pipeline: python experiments/run_pipeline.py")
        print("  2. Run classical ML only: python experiments/run_classical.py")
        print("  3. Read PIPELINE_GUIDE.md for detailed usage examples")
        return 0
    else:
        print(f"✗ {total - passed} TEST(S) FAILED ({passed}/{total})")
        print("="*70)
        print("\nPlease check the errors above and install missing dependencies.")
        print(f"Install dependencies: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
