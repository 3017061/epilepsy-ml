"""
Complete end-to-end ML pipeline for biomedical classification.
Uses real public datasets from UCI ML Repository.
Combines data loading, feature engineering, classical ML, and deep learning.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import matplotlib.pyplot as plt

from src.data.loader import load_public_dataset, split_dataset, get_dataset_info
from src.features.engineer import enrich_features
from src.models.classical import make_classical_model, make_ensemble_model
from src.models.deep import SimpleMLP, DeepMLP, get_device, count_parameters
from src.preprocessing.pipeline import build_preprocessing_pipeline, adaptive_k_best
from src.training.train import (
    train_sklearn,
    train_torch,
    predict_torch,
    split_data,
)
from src.evaluation.metrics import (
    classification_summary,
    binary_classification_summary,
    evaluate_model_performance,
)
from src.visualization.plots import (
    plot_correlation_matrix,
    plot_pca_embedding,
    plot_training_history,
    plot_class_distribution,
)
from src.utils.seed import set_seed


def ensure_directories():
    """Create output directories."""
    os.makedirs("experiments/figures", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)


def run_classical_pipeline(X_train, X_val, X_test, y_train, y_val, y_test, metadata):
    """
    Run classical ML models.
    
    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Labels
        metadata: Dataset metadata
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*70)
    print("CLASSICAL MACHINE LEARNING MODELS")
    print("="*70)
    
    results = {}
    
    models_to_test = ["random_forest", "gradient_boosting", "svm", "knn"]
    
    for model_name in models_to_test:
        print(f"\nTraining {model_name.upper()}...", end=" ")
        
        try:
            model = make_classical_model(model_name)
            model = train_sklearn(model, X_train, y_train)
            
            # Evaluate on validation set
            y_val_pred = model.predict(X_val)
            val_summary = classification_summary(y_val, y_val_pred)
            
            # Evaluate on test set
            y_test_pred = model.predict(X_test)
            y_test_proba = None
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_test_proba = model.predict_proba(X_test)
            except Exception:
                pass
            
            if len(np.unique(y_test)) == 2:
                test_summary = binary_classification_summary(y_test, y_test_pred, y_test_proba)
            else:
                test_summary = classification_summary(y_test, y_test_pred, y_test_proba)
            
            results[model_name] = {
                "val_accuracy": float(val_summary["accuracy"]),
                "test_accuracy": float(test_summary["accuracy"]),
                "test_f1": float(test_summary["f1_score"]),
                "test_precision": float(test_summary["precision"]),
                "test_recall": float(test_summary["recall"]),
                "model": model,
            }
            
            print(f"✓ Test Acc: {test_summary['accuracy']:.4f}, F1: {test_summary['f1_score']:.4f}")
        
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
    
    return results


def run_deep_learning_pipeline(X_train, X_val, X_test, y_train, y_val, y_test, metadata):
    """
    Run deep learning models.
    
    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Labels
        metadata: Dataset metadata
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*70)
    print("DEEP LEARNING MODELS")
    print("="*70)
    
    device = get_device()
    print(f"\nUsing device: {device}")
    
    results = {}
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    model_configs = [
        ("SimpleMLP", SimpleMLP, {"input_dim": input_dim, "n_classes": n_classes}),
        ("DeepMLP", DeepMLP, {"input_dim": input_dim, "n_classes": n_classes}),
    ]
    
    for model_name, ModelClass, kwargs in model_configs:
        print(f"\nTraining {model_name}...", end=" ")
        
        try:
            model = ModelClass(**kwargs)
            n_params = count_parameters(model)
            
            # Train model
            model, history = train_torch(
                model,
                X_train, y_train,
                X_val, y_val,
                epochs=50,
                lr=1e-3,
                batch_size=32,
                device=device,
                early_stopping=True,
                patience=5,
                verbose=False,
            )
            
            # Evaluate on test set
            X_test_t = torch.tensor(X_test.values, dtype=torch.float32, device=device)
            y_test_pred, y_test_proba = predict_torch(model, X_test_t, device)
            
            test_summary = classification_summary(
                y_test.values, y_test_pred, y_test_proba
            )
            
            results[model_name] = {
                "test_accuracy": float(test_summary["accuracy"]),
                "test_f1": float(test_summary["f1_score"]),
                "test_precision": float(test_summary["precision"]),
                "test_recall": float(test_summary["recall"]),
                "n_parameters": int(n_params),
                "history": history,
                "model": model,
            }
            
            print(f"✓ Test Acc: {test_summary['accuracy']:.4f}, F1: {test_summary['f1_score']:.4f}")
        
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
    
    return results


def run_full_pipeline(dataset_name="breast_cancer"):
    """
    Run complete ML pipeline: data loading, preprocessing, and model evaluation.
    
    Args:
        dataset_name: Dataset to use
    """
    ensure_directories()
    set_seed(42)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/results/pipeline_results_{timestamp}.json"
    
    print("\n" + "="*70)
    print("EPILEPSY DETECTION ML PIPELINE")
    print("="*70)
    
    # Load data
    print(f"\nLoading dataset: {dataset_name}...", end=" ")
    df, metadata = load_public_dataset(dataset_name)
    print("✓")
    
    print(f"  Samples: {df.shape[0]}, Features: {df.shape[1]-1}, Classes: {metadata['n_classes']}")
    
    # Split data (train, val, test)
    print(f"\nSplitting data...", end=" ")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, test_size=0.2, val_size=0.1)
    print(f"✓ (Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]})")
    
    # Feature engineering
    print(f"Feature engineering...", end=" ")
    X_train = enrich_features(X_train, include_statistical=True)
    X_val = enrich_features(X_val, include_statistical=True)
    X_test = enrich_features(X_test, include_statistical=True)
    print(f"✓ ({X_train.shape[1]} features)")
    
    # Preprocessing
    print(f"Preprocessing...", end=" ")
    k_best = adaptive_k_best(X_train, percentile=0.8, min_k=15)
    preprocess = build_preprocessing_pipeline(k_best=k_best)
    preprocess.fit(X_train, y_train)
    X_train_transformed = preprocess.transform(X_train)
    X_val_transformed = preprocess.transform(X_val)
    X_test_transformed = preprocess.transform(X_test)
    print(f"✓ ({X_train_transformed.shape[1]} features after selection)")
    
    # Visualizations
    print(f"\nGenerating visualizations...")
    
    fig = plot_class_distribution(y_train, metadata.get("target_names"))
    fig.savefig(f"experiments/figures/class_distribution_{timestamp}.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ Class distribution saved")
    
    fig = plot_pca_embedding(X_train_transformed, y_train)
    fig.savefig(f"experiments/figures/pca_embedding_{timestamp}.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ PCA embedding saved")
    
    # Run classical ML models
    classical_results = run_classical_pipeline(
        X_train_transformed, X_val_transformed, X_test_transformed,
        y_train, y_val, y_test,
        metadata
    )
    
    # Convert pandas Series to numpy arrays if needed
    X_train_transformed_np = X_train_transformed.values if isinstance(X_train_transformed, pd.DataFrame) else X_train_transformed
    X_val_transformed_np = X_val_transformed.values if isinstance(X_val_transformed, pd.DataFrame) else X_val_transformed
    X_test_transformed_np = X_test_transformed.values if isinstance(X_test_transformed, pd.DataFrame) else X_test_transformed
    
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    # Convert back to pandas for torch training
    X_train_pd = pd.DataFrame(X_train_transformed_np)
    X_val_pd = pd.DataFrame(X_val_transformed_np)
    X_test_pd = pd.DataFrame(X_test_transformed_np)
    y_train_pd = pd.Series(y_train_np)
    y_val_pd = pd.Series(y_val_np)
    y_test_pd = pd.Series(y_test_np)
    
    # Run deep learning models
    dl_results = run_deep_learning_pipeline(
        X_train_pd, X_val_pd, X_test_pd,
        y_train_pd, y_val_pd, y_test_pd,
        metadata
    )
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    summary = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "metadata": metadata,
        "classical_ml": {k: {kk: vv for kk, vv in v.items() if kk != "model"} 
                        for k, v in classical_results.items()},
        "deep_learning": {k: {kk: vv for kk, vv in v.items() if kk not in ["model", "history"]} 
                         for k, v in dl_results.items()},
    }
    
    print("\nClassical ML Results:")
    for model_name, results in classical_results.items():
        print(f"  {model_name}: Test Acc={results['test_accuracy']:.4f}, F1={results['test_f1']:.4f}")
    
    print("\nDeep Learning Results:")
    for model_name, results in dl_results.items():
        print(f"  {model_name}: Test Acc={results['test_accuracy']:.4f}, F1={results['test_f1']:.4f}")
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return summary


if __name__ == "__main__":
    # Run full pipeline with breast cancer dataset
    results = run_full_pipeline(dataset_name="breast_cancer")
