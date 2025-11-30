# Prosodic Feature-Based Behaviour Classification

A complete, production-quality implementation for academic research on prosodic feature-based behavioral/emotional classification using interpretable, low-compute speech analytics.

## Overview

This project implements a full experimental pipeline for classifying behavioral and emotional states from speech using prosodic features. The system extracts utterance-level prosodic features (F0, jitter, shimmer, intensity, speaking rate, pauses) and optional spectral features (MFCCs, formants) from audio recordings, then trains interpretable machine learning models for classification.

## Features

- **Multi-Dataset Support**: Integrated loaders for RAVDESS, CREMA-D, and SAVEE datasets
- **Robust Preprocessing**: Resampling, loudness normalization, VAD-based silence trimming, noise reduction
- **Comprehensive Feature Extraction**: Prosodic features using Praat (parselmouth) and spectral features using librosa
- **Multiple Models**: Logistic Regression, Linear SVM, Random Forest, and XGBoost
- **Speaker-Independent Evaluation**: Stratified k-fold cross-validation with speaker-independent splits
- **Interpretability**: SHAP explanations for model interpretability
- **Publication-Ready Visualizations**: Confusion matrices, ROC curves, feature importance plots

## Project Structure

```
Prosodic-Feature-Based-Behaviour-Analysis/
├── src/
│   ├── data_loader/          # Dataset loaders (RAVDESS, CREMA-D, SAVEE)
│   ├── preprocessing/        # Audio preprocessing pipeline
│   ├── feature_extraction/   # Prosodic and spectral feature extraction
│   ├── models/               # Model training and evaluation
│   ├── evaluation/           # Metrics, visualizations, interpretability
│   ├── visualization/        # Plotting utilities
│   └── utils/                # Configuration, logging utilities
├── configs/                  # YAML configuration files
├── notebooks/                # Jupyter notebooks for analysis
├── results/                  # Experiment results and outputs
├── data/                     # Dataset storage (created automatically)
├── main.py                   # Main CLI script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
cd Prosodic-Feature-Based-Behaviour-Analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install parselmouth separately (due to dependency issues with Python 3.12):
```bash
# Option 1: Use the installation script (recommended)
./install_parselmouth.sh

# Option 2: Manual installation (ensure setuptools is installed first)
pip install setuptools wheel build
pip install parselmouth --no-deps
```

**Note**: The `--no-deps` flag skips the problematic `googleads` dependency which is not actually needed for parselmouth to function.

**Note**: 
- **Parselmouth**: Has compatibility issues with Python 3.12. If installation fails, the code will automatically use librosa-based F0 extraction as a fallback. All prosodic features will still be extracted, though some advanced Praat-specific features (like precise jitter/shimmer calculations) will use approximations. The code is fully functional without parselmouth.
- **XGBoost**: On macOS, XGBoost requires OpenMP runtime. If you want to use XGBoost, install it with: `brew install libomp`. However, XGBoost is optional - you can use other models (logistic_regression, svm, random_forest) without it.

### Dataset Download

The project supports three datasets. You need to download them manually and place them in the `data/` directory:

1. **RAVDESS**: Download from [Zenodo](https://zenodo.org/record/1188976)
   - Place `Audio_Speech_Actors_01-24.zip` in `data/ravdess/`

2. **CREMA-D**: Download from [GitHub](https://github.com/CheyneyComputerScience/CREMA-D)
   - Place `AudioWAV.zip` in `data/crema_d/`

3. **SAVEE**: Download from [SAVEE website](http://kahlan.eps.surrey.ac.uk/savee/Download.html)
   - Place `AudioData.zip` in `data/savee/`

The loaders will automatically extract the zip files on first run.

## Usage

### Running Experiments

Run a complete experiment using the main CLI script:

```bash
python main.py --config configs/experiment.yaml
```

This will:
1. Load the specified datasets
2. Preprocess audio files
3. Extract prosodic and spectral features
4. Train the specified model
5. Evaluate performance with cross-validation
6. Generate visualizations and SHAP interpretability plots
7. Save all results to the `results/` directory

### Configuration

Edit `configs/experiment.yaml` to customize:
- Which datasets to use
- Preprocessing parameters
- Feature extraction settings
- Model type and hyperparameters
- Evaluation settings

Example configuration:
```yaml
experiment_name: "prosodic_behavior_classification"
datasets:
  - "ravdess"
  - "crema_d"
  - "savee"

model:
  model_type: "logistic_regression"  # or "svm", "random_forest", "xgboost"
  cv_folds: 5
  speaker_independent: true
```

### Using in Python

You can also use the modules programmatically:

```python
from src.data_loader import load_ravdess
from src.feature_extraction import ProsodicFeatureExtractor
from src.models import ModelTrainer

# Load dataset
df = load_ravdess(data_dir="./data/ravdess")

# Extract features
extractor = ProsodicFeatureExtractor()
features = extractor.extract_from_file("path/to/audio.wav")

# Train model
trainer = ModelTrainer(model_type="logistic_regression")
# ... training code ...
```

## Features Extracted

### Prosodic Features
- **F0 (Fundamental Frequency)**: mean, median, std, min, max, range
- **Jitter**: local, local absolute, RAP, PPQ5
- **Shimmer**: local, local dB, APQ3, APQ5
- **Voicing**: voicing ratio
- **Intensity**: mean, std
- **Speaking Rate**: approximate using voiced frames
- **Pauses**: count and average duration

### Spectral Features (Optional)
- **MFCCs**: 13 coefficients with deltas
- **Formants**: F1 and F2 means

## Models

The project supports four model types:

1. **Logistic Regression**: Fast, interpretable baseline
2. **Linear SVM**: Good for high-dimensional features
3. **Random Forest**: Non-linear, feature importance available
4. **XGBoost**: Gradient boosting with early stopping

All models support:
- Class weighting for imbalanced datasets
- Cross-validation
- Speaker-independent evaluation

## Evaluation Metrics

The evaluation module computes:
- **Accuracy**: Overall classification accuracy
- **Macro F1**: Unweighted mean of per-class F1 scores
- **UAR (Unweighted Average Recall)**: Mean recall across classes
- **Per-Class Metrics**: Precision, recall, F1 for each class
- **Confusion Matrix**: Visual representation of classification performance
- **ROC Curves**: Receiver Operating Characteristic curves per class

## Interpretability

SHAP (SHapley Additive exPlanations) is used for model interpretability:
- **Summary Plots**: Global feature importance
- **Waterfall Plots**: Local explanations for individual predictions
- **Misclassified Analysis**: Explanations for incorrectly classified examples

## Results

All results are saved to the `results/` directory:
- `{experiment_name}_metrics.json`: Evaluation metrics
- `{experiment_name}_report.txt`: Text summary report
- `{experiment_name}_confusion_matrix.png`: Confusion matrix visualization
- `{experiment_name}_roc_curves.png`: ROC curves
- `{experiment_name}_model.pkl`: Trained model
- `{experiment_name}_features.csv`: Extracted features
- `{experiment_name}_shap_*.png`: SHAP interpretability plots
- `{experiment_name}.log`: Experiment log file

## Notebooks

See `notebooks/analysis.ipynb` for an interactive demonstration of the full workflow.

## Extending the Project

### Adding New Datasets

1. Create a new loader class in `src/data_loader/` inheriting from `BaseDatasetLoader`
2. Implement `download()` and `load()` methods
3. Add loader function to `src/data_loader/__init__.py`

### Adding New Features

1. Extend `ProsodicFeatureExtractor` in `src/feature_extraction/extractor.py`
2. Add feature extraction methods
3. Update configuration schema if needed

### Adding New Models

1. Add model creation logic to `src/models/model_factory.py`
2. Update hyperparameter configuration in YAML
3. Ensure model supports `fit()`, `predict()`, and `predict_proba()` methods

## Requirements

See `requirements.txt` for complete dependency list. Key dependencies:
- numpy, pandas, scipy
- scikit-learn
- librosa, soundfile
- parselmouth (Praat Python interface)
- matplotlib, seaborn
- xgboost (optional)
- shap (optional, for interpretability)

## Citation

If you use this code in your research, please cite appropriately and acknowledge the datasets used:
- RAVDESS: Livingstone & Russo (2018)
- CREMA-D: Cao et al. (2014)
- SAVEE: Jackson & Haq (2014)

## License

This project is provided for academic research purposes.

## Troubleshooting

### Common Issues

1. **Parselmouth/Praat errors**: Ensure Praat is installed. Parselmouth includes Praat, but some systems may need additional setup.

2. **Memory errors with large datasets**: Reduce the number of files processed or increase system memory.

3. **SHAP errors**: SHAP can be slow for large datasets. The code automatically uses subsets for faster computation.

4. **Dataset not found**: Ensure datasets are downloaded and placed in the correct directories as specified in the Installation section.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

