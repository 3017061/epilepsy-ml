"""
Classical ML experiment runner for biomedical classification.
Uses real public datasets from UCI ML Repository.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from src.data.loader import load_public_dataset, split_dataset, get_dataset_info
from src.features.engineer import enrich_features
from src.models.classical import make_classical_model, get_model_list
from src.preprocessing.pipeline import build_preprocessing_pipeline, adaptive_k_best
from src.training.train import train_sklearn, cross_val_evaluate
from src.evaluation.metrics import (
    binary_classification_summary,
    classification_summary,
    plot_confusion_matrix,
    evaluate_model_performance,
)
from src.visualization.plots import (
    plot_correlation_matrix,
    plot_pca_embedding,
    plot_feature_importance,
    plot_class_distribution,
)


def ensure_experiment_directory():
    """Create experiment output directory."""
    os.makedirs("experiments/figures", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)


def run_single_model_evaluation(
    X_train, X_test, y_train, y_test,
    model_name="random_forest",
    dataset_name="dataset",
    metadata=None,
):
    """
    Train and evaluate a single model.
    
    Args:
        X_train, X_test, y_train, y_test: Training/test data
        model_name: Name of model to train
        dataset_name: Name of dataset
        metadata: Optional metadata
        
    Returns:
        Dictionary with results
    """
    print(f"\n  Training {model_name}...", end=" ")
    
    # Create and train model
    model = make_classical_model(model_name)
    model = train_sklearn(model, X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = None
    
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
    except Exception:
        pass
    
    # Evaluate
    target_names = metadata.get("target_names") if metadata else None
    
    if len(np.unique(y_test)) == 2:
        summary = binary_classification_summary(y_test, y_pred, y_proba, target_names)
    else:
        summary = classification_summary(y_test, y_pred, y_proba, target_names)
    
    print(f"✓ Accuracy: {summary['accuracy']:.4f}")
    
    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    
    return {
        "model": model,
        "summary": summary,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_importance": feature_importance,
    }


def run_classical_workflow(dataset_name="breast_cancer"):
    """
    Run full classical ML evaluation workflow on real public datasets.
    
    Args:
        dataset_name: Real dataset from UCI ML Repository ('breast_cancer', 'wine')
    """
    ensure_experiment_directory()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/results/classical_results_{timestamp}.json"
    
    print(f"\n{'='*70}")
    print(f"Classical ML Evaluation - Real Public Dataset")
    print(f"{'='*70}")
    
    # Load real public dataset
    print(f"\nLoading {dataset_name} dataset...", end=" ")
    df, metadata = load_public_dataset(dataset_name)
    print(f"✓")
    
    dataset_info = get_dataset_info(dataset_name)
    print(f"  Source: {dataset_info['source']}")
    print(f"  Samples: {metadata['n_samples']}, Features: {metadata['n_features']}, Classes: {metadata['n_classes']}")
    print(f"  Classes: {', '.join(metadata['target_names'])}")
    print(f"  Class Distribution: {metadata['class_distribution']}")
    
    # Split data
    print(f"\nSplitting data...", end=" ")
    X_train, X_test, y_train, y_test = split_dataset(df)
    print(f"✓ (Train: {X_train.shape[0]}, Test: {X_test.shape[0]})")
    
    # Feature engineering
    print(f"Feature engineering...", end=" ")
    X_train = enrich_features(X_train, include_statistical=True)
    X_test = enrich_features(X_test, include_statistical=True)
    print(f"✓ (Features: {X_train.shape[1]})")
    
    # Preprocessing
    print(f"Preprocessing...", end=" ")
    k_best = adaptive_k_best(X_train, percentile=0.8, min_k=10)
    preprocess = build_preprocessing_pipeline(k_best=k_best)
    preprocess.fit(X_train, y_train)
    X_train_transformed = preprocess.transform(X_train)
    X_test_transformed = preprocess.transform(X_test)
    print(f"✓ (Features after selection: {X_train_transformed.shape[1]})")
    
    # Visualizations
    print(f"\nGenerating visualizations...")
    
    fig = plot_class_distribution(y_train, metadata.get("target_names"))
    fig.savefig(f"experiments/figures/class_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Class distribution")
    
    fig = plot_correlation_matrix(df.drop(columns=["target"]))
    fig.savefig(f"experiments/figures/correlation_matrix_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Correlation matrix")
    
    fig = plot_pca_embedding(X_train_transformed, y_train, labels=metadata.get("target_names"))
    fig.savefig(f"experiments/figures/pca_embedding_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ PCA embedding")
    
    # Model evaluation
    print(f"\nTraining and evaluating models...")
    
    results = {}
    model_list = get_model_list()
    
    for model_name in model_list:
        try:
            result = run_single_model_evaluation(
                X_train_transformed, X_test_transformed,
                y_train, y_test,
                model_name=model_name,
                dataset_name=dataset_name,
                metadata=metadata,
            )
            results[model_name] = {
                "accuracy": float(result["summary"]["accuracy"]),
                "f1_score": float(result["summary"]["f1_score"]),
                "precision": float(result["summary"]["precision"]),
                "recall": float(result["summary"]["recall"]),
            }
            
            # Save feature importance if available
            if result["feature_importance"] is not None and len(result["feature_importance"]) <= 50:
                try:
                    feature_names = preprocess.get_feature_names_out() if hasattr(preprocess, 'get_feature_names_out') else [f"Feature_{i}" for i in range(X_train_transformed.shape[1])]
                    fig = plot_feature_importance(feature_names, result["feature_importance"], top_n=10)
                    fig.savefig(f"experiments/figures/feature_importance_{model_name}_{timestamp}.png", dpi=300, bbox_inches='tight')
                    plt.close(fig)
                except Exception:
                    pass
        
        except Exception as e:
            print(f"  ✗ {model_name}: {str(e)[:50]}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("Model Performance Summary")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("f1_score", ascending=False)
    
    print(results_df.to_string())
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "dataset": dataset_name,
            "metadata": metadata,
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results_df


def run_hyperparameter_tuning(dataset_name="breast_cancer"):
    """Run hyperparameter tuning for top models."""
    print(f"\n{'='*70}")
    print("Hyperparameter Tuning")
    print(f"{'='*70}")
    
    from src.training.train import hyperparameter_tuning
    
    # Load and preprocess data
    df, metadata = load_public_dataset(dataset_name)
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train = enrich_features(X_train, include_statistical=True)
    X_test = enrich_features(X_test, include_statistical=True)
    
    k_best = adaptive_k_best(X_train, percentile=0.8, min_k=10)
    preprocess = build_preprocessing_pipeline(k_best=k_best)
    preprocess.fit(X_train, y_train)
    X_train_transformed = preprocess.transform(X_train)
    X_test_transformed = preprocess.transform(X_test)
    
    # Define parameter grids for key models
    param_grids = {
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 15, 20],
            "min_samples_split": [2, 5],
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ['rbf', 'linear'],
        },
    }
    
    tuning_results = {}
    
    for model_name, param_grid in param_grids.items():
        print(f"\nTuning {model_name}...")
        model = make_classical_model(model_name)
        
        best_model, best_params, cv_results = hyperparameter_tuning(
            model, param_grid, X_train_transformed, y_train, cv=3
        )
        
        y_pred = best_model.predict(X_test_transformed)
        summary = classification_summary(y_test, y_pred)
        
        tuning_results[model_name] = {
            "best_params": best_params,
            "accuracy": float(summary["accuracy"]),
            "f1_score": float(summary["f1_score"]),
        }
        
        print(f"  Best params: {best_params}")
        print(f"  Accuracy: {summary['accuracy']:.4f}, F1: {summary['f1_score']:.4f}")
    
    return tuning_results


if __name__ == "__main__":
    # Run classical ML evaluation
    results = run_classical_workflow(dataset_name="breast_cancer")
    
    # Optional: Hyperparameter tuning
    # tuning_results = run_hyperparameter_tuning()
