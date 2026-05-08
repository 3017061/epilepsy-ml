
# The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy


## Features

✨ **Data Science Workflow**
- Public biomedical dataset loading with reproducible preprocessing
- Exploratory data analysis (EDA) with publication-ready visualizations
- Feature engineering and dimensionality reduction (PCA)

🤖 **ML Models**
- Classical models: Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, SVM, KNN, Naive Bayes
- Deep learning: PyTorch multi-layer perceptron (MLP) classifiers
- Ensemble methods and unsupervised clustering

📊 **Research Tools**
- Hyperparameter tuning via GridSearchCV and cross-validation
- Interactive Jupyter notebook dashboard with `ipywidgets` controls
- Publication-ready metrics and evaluation summaries
- Feature importance visualization

## Quick Start

### Installation

```bash
git clone https://github.com/3017061/epilepsy-ml-pipeline.git
cd epilepsy-ml-pipeline
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Pipeline

**Headless experiment runner:**
```bash
python experiments/run_pipeline.py
```

**Classical model evaluation:**
```bash
python experiments/run_classical.py
```

**Interactive research notebook:**
```bash
jupyter notebook notebooks/ml_pipeline.ipynb
```

## Project Structure

```
epilepsy-ml-pipeline/
├── src/
│   ├── data/              # Dataset loading and splitting
│   ├── preprocessing/     # Data preprocessing pipelines
│   ├── features/          # Feature engineering utilities
│   ├── models/            # Classical and deep learning models
│   ├── training/          # Training utilities for sklearn and PyTorch
│   ├── evaluation/        # Metrics and evaluation functions
│   └── visualization/     # Plotting and visualization tools
├── experiments/
│   ├── run_pipeline.py    # End-to-end research workflow
│   ├── run_classical.py   # Classical ML evaluation
│   └── figures/           # Generated output visualizations
├── notebooks/
│   └── ml_pipeline.ipynb  # Interactive research notebook
├── tests/                 # Unit tests (pytest)
├── publications.md        # Publication metadata template
├── requirements.txt       # Python dependencies
└── README.md
```

## Dependencies

- **Data & ML**: numpy, pandas, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Deep Learning**: PyTorch
- **Notebooks**: Jupyter, ipywidgets
- See `requirements.txt` for versions

## Usage Examples

### Load and preprocess biomedical data:
```python
from src.data.loader import load_public_dataset, split_dataset
from src.preprocessing.pipeline import build_preprocessing_pipeline

df, metadata = load_public_dataset("breast_cancer")
X_train, X_test, y_train, y_test = split_dataset(df)

preprocess = build_preprocessing_pipeline(k_best=20)
X_train = preprocess.fit_transform(X_train, y_train)
X_test = preprocess.transform(X_test)
```

### Train a classical model:
```python
from src.models.classical import make_classical_model
from src.training.train import train_sklearn
from src.evaluation.metrics import classification_summary

model = make_classical_model("random_forest")
model = train_sklearn(model, X_train, y_train)
summary = classification_summary(y_test, model.predict(X_test))
print(summary['report'])
```

## Research & Publication

This repository implements the computational methods from:

**Vaish, S. (2024).** *The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy.* International Journal for Multidisciplinary Research (IJFMR), 6(3), May-June.

📄 **Publication Details**:
- **Journal**: International Journal for Multidisciplinary Research (IJFMR)
- **Volume**: 6, Issue 3 (May-June 2024)
- **ISSN**: 2582-2160  
- **DOI**: [10.36948/ijfmr.2024.v06i03.22870](https://doi.org/10.36948/ijfmr.2024.v06i03.22870)
- **Research Paper**: See [papers/](papers/) directory

### Manuscript Abstract

Epilepsy affects over 50 million people worldwide. This research investigates machine learning approaches for automated epilepsy prediction and neurological disorder detection. We implement and compare multiple classical ML models and deep neural networks with comprehensive preprocessing and feature engineering. Ensemble methods (Random Forest, Gradient Boosting) achieve superior performance and provide interpretable decision support for clinical applications.

For full publication metadata, supplementary materials, and research details, see [publications.md](publications.md).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- Fork the repository
- Create a feature branch
- Submit a pull request with clear commit messages

## Citation

If you use this code or methodology in your research, please cite the original paper and/or this repository:

**Original Research Paper**:
```bibtex
@article{vaish2024epilepsy,
  author = {Vaish, Saniya},
  title = {The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy},
  journal = {International Journal for Multidisciplinary Research (IJFMR)},
  volume = {6},
  number = {3},
  year = {2024},
  issn = {2582-2160},
  doi = {10.36948/ijfmr.2024.v06i03.22870},
  url = {https://www.ijfmr.com}
}
```

**This Repository**:
```bibtex
@software{vaish2026pipeline,
  author = {Vaish, Saniya},
  title = {Biomedical ML Research: Epilepsy Prediction Pipeline},
  year = {2026},
  url = {https://github.com/3017061/epilepsy-ml-pipeline},
  note = {Open-source implementation accompanying IJFMR publication}
}
```

## License

This project is licensed under the MIT License—see [LICENSE](LICENSE) for details.

## Contact

For questions or feedback, please open an [issue](https://github.com/3017061/epilepsy-ml-pipeline/issues) or contact via your preferred channel.

---

**Made with ❤️ Saniya Vaish :)**
