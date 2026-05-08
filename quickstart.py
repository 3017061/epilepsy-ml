#!/usr/bin/env python3
"""
Quick Start Script for Epilepsy ML Pipeline
Provides easy access to common pipeline operations.
"""

import argparse
import sys
from pathlib import Path


def setup_environment():
    """Ensure working directory is set correctly."""
    import os
    os.chdir(Path(__file__).parent)


def run_tests():
    """Run pipeline validation tests."""
    print("\n" + "="*70)
    print("Running Pipeline Validation Tests...")
    print("="*70)
    import test_pipeline
    return test_pipeline.main()


def run_full_pipeline(dataset="breast_cancer"):
    """Run complete ML pipeline."""
    print("\n" + "="*70)
    print("Running Full ML Pipeline")
    print("="*70)
    
    from experiments.run_pipeline import run_full_pipeline
    results = run_full_pipeline(dataset_name=dataset)
    return 0


def run_classical_models(dataset="breast_cancer"):
    """Run classical ML model evaluation."""
    print("\n" + "="*70)
    print("Running Classical ML Model Evaluation")
    print("="*70)
    
    from experiments.run_classical import run_classical_workflow
    results = run_classical_workflow(dataset_name=dataset)
    return 0


def run_quick_demo():
    """Run a quick demonstration."""
    print("\n" + "="*70)
    print("Quick Demonstration - Basic Pipeline")
    print("="*70)
    
    try:
        from src.data.loader import load_public_dataset, split_dataset
        from src.features.engineer import enrich_features
        from src.models.classical import make_classical_model
        from src.preprocessing.pipeline import build_preprocessing_pipeline
        from src.training.train import train_sklearn
        from src.evaluation.metrics import classification_summary
        
        print("\n1. Loading Breast Cancer dataset...", end=" ")
        df, metadata = load_public_dataset("breast_cancer")
        print(f"✓ Loaded {df.shape[0]} samples with {df.shape[1]-1} features")
        
        print("2. Splitting data...", end=" ")
        X_train, X_test, y_train, y_test = split_dataset(df)
        print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        print("3. Feature engineering...", end=" ")
        X_train = enrich_features(X_train, include_statistical=True)
        X_test = enrich_features(X_test, include_statistical=True)
        print(f"✓ Generated {X_train.shape[1]} features")
        
        print("4. Preprocessing...", end=" ")
        pipeline = build_preprocessing_pipeline(k_best=20)
        pipeline.fit(X_train, y_train)
        X_train_t = pipeline.transform(X_train)
        X_test_t = pipeline.transform(X_test)
        print(f"✓ Selected {X_train_t.shape[1]} best features")
        
        print("5. Training Random Forest model...", end=" ")
        model = make_classical_model("random_forest")
        model = train_sklearn(model, X_train_t, y_train)
        print("✓ Model trained")
        
        print("6. Evaluating model...", end=" ")
        y_pred = model.predict(X_test_t)
        summary = classification_summary(y_test, y_pred, metadata.get("target_names"))
        print(f"✓ Accuracy: {summary['accuracy']:.4f}")
        
        print("\n" + "="*70)
        print("RESULTS:")
        print("="*70)
        print(f"Accuracy:  {summary['accuracy']:.4f}")
        print(f"F1-Score:  {summary['f1_score']:.4f}")
        print(f"Precision: {summary['precision']:.4f}")
        print(f"Recall:    {summary['recall']:.4f}")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def show_help():
    """Show detailed help information."""
    help_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   EPILEPSY ML PIPELINE - QUICK START                       ║
╚════════════════════════════════════════════════════════════════════════════╝

USAGE:
    python quickstart.py [COMMAND] [OPTIONS]

COMMANDS:
    test                Run pipeline validation tests (9 tests)
    demo                Run quick demonstration with Breast Cancer dataset
    full                Run complete ML pipeline
    classical           Run classical ML model evaluation
    help                Show this help message

OPTIONS:
    --dataset DATASET   Dataset to use: breast_cancer, wine
    --help             Show this message

EXAMPLES:
    # Run validation tests (recommended first)
    python quickstart.py test
    
    # Run quick demo
    python quickstart.py demo
    
    # Run complete pipeline with breast cancer data
    python quickstart.py full --dataset breast_cancer
    
    # Run classical ML evaluation only
    python quickstart.py classical

FILES & DOCUMENTATION:
    PIPELINE_GUIDE.md          → Complete user guide (2000+ lines)
    IMPLEMENTATION_SUMMARY.md  → Technical implementation details
    README.md                  → Project overview
    test_pipeline.py           → Validation test suite
    requirements.txt           → Python dependencies

KEY MODULES:
    src/data/loader.py         → Data loading and management
    src/features/extractor.py  → Advanced feature extraction (50+ features)
    src/models/classical.py    → 12+ classical ML models
    src/models/deep.py         → 5 deep learning architectures
    src/evaluation/metrics.py  → 14+ evaluation metrics
    src/visualization/plots.py → 10+ publication-ready plots

QUICK PYTHON API:
    from src.data.loader import load_public_dataset, split_dataset
    from src.features.engineer import enrich_features
    from src.models.classical import make_classical_model
    from src.training.train import train_sklearn
    
    # Load and split data
    df, metadata = load_public_dataset("breast_cancer")
    X_train, X_test, y_train, y_test = split_dataset(df)
    
    # Feature engineering
    X_train = enrich_features(X_train, include_statistical=True)
    X_test = enrich_features(X_test, include_statistical=True)
    
    # Train model
    model = make_classical_model("random_forest")
    model = train_sklearn(model, X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)

INSTALLATION:
    pip install -r requirements.txt

FIRST-TIME SETUP:
    1. python quickstart.py test      # Validate installation
    2. python quickstart.py demo      # Run quick example
    3. Read PIPELINE_GUIDE.md         # Understand the pipeline

For detailed documentation, see: PIPELINE_GUIDE.md
For technical details, see: IMPLEMENTATION_SUMMARY.md
"""
    print(help_text)


def main():
    """Main entry point."""
    setup_environment()
    
    parser = argparse.ArgumentParser(
        description="Epilepsy ML Pipeline Quick Start",
        add_help=False,  # Use custom help
    )
    
    parser.add_argument("command", nargs="?", default="help",
                       choices=["test", "demo", "full", "classical", "help"],
                       help="Command to run")
    parser.add_argument("--dataset", default="breast_cancer",
                       choices=["breast_cancer", "wine"],
                       help="Dataset to use")
    parser.add_argument("--help", "-h", action="store_true",
                       help="Show help message")
    
    try:
        args = parser.parse_args()
    except SystemExit:
        show_help()
        return 1
    
    # Handle help
    if args.help or args.command == "help":
        show_help()
        return 0
    
    # Execute command
    if args.command == "test":
        return run_tests()
    elif args.command == "demo":
        return run_quick_demo()
    elif args.command == "full":
        return run_full_pipeline(dataset=args.dataset)
    elif args.command == "classical":
        return run_classical_models(dataset=args.dataset)
    else:
        show_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
