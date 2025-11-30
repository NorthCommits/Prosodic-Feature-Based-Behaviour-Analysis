# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install parselmouth (for prosodic features):
```bash
# Quick method:
pip install setuptools wheel build
pip install parselmouth --no-deps

# Or use the script:
./install_parselmouth.sh
```

3. Download datasets (see README.md for links):
   - Place RAVDESS zip in `data/ravdess/`
   - Place CREMA-D zip in `data/crema_d/`
   - Place SAVEE zip in `data/savee/`

## Running an Experiment

```bash
python main.py --config configs/experiment.yaml
```

## Changing Model Type

Edit `configs/experiment.yaml` and change:
```yaml
model:
  model_type: "logistic_regression"  # or "svm", "random_forest", "xgboost"
```

## Viewing Results

Results are saved in `results/` directory:
- Metrics: `{experiment_name}_metrics.json`
- Confusion matrix: `{experiment_name}_confusion_matrix.png`
- ROC curves: `{experiment_name}_roc_curves.png`
- SHAP plots: `{experiment_name}_shap_*.png`

## Using the Notebook

Open `notebooks/analysis.ipynb` in Jupyter for interactive analysis.

