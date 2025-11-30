# How to Run the Program

## Quick Start (If you have datasets)

1. **Ensure you're in the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run the main script:**
   ```bash
   python3 main.py --config configs/experiment.yaml
   ```

## Full Setup (First Time)

### Step 1: Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install main dependencies
pip install -r requirements.txt

# (Optional) Install parselmouth for Praat features
# Note: May not work on Python 3.12, but code will use librosa fallback
./install_parselmouth.sh
```

### Step 2: Download Datasets

You need to download at least one of these datasets:

**RAVDESS:**
- Download from: https://zenodo.org/record/1188976
- Place `Audio_Speech_Actors_01-24.zip` in `data/ravdess/`

**CREMA-D:**
- Download from: https://github.com/CheyneyComputerScience/CREMA-D
- Place `AudioWAV.zip` in `data/crema_d/`

**SAVEE:**
- Download from: http://kahlan.eps.surrey.ac.uk/savee/Download.html
- Place `AudioData.zip` in `data/savee/`

The program will automatically extract the zip files on first run.

### Step 3: Configure Your Experiment

Edit `configs/experiment.yaml` to:
- Choose which datasets to use
- Select model type (logistic_regression, svm, random_forest, or xgboost)
- Adjust preprocessing and feature extraction settings

### Step 4: Run the Experiment

```bash
python3 main.py --config configs/experiment.yaml
```

## What Happens When You Run

1. **Loads datasets** - Reads audio files and metadata
2. **Preprocesses audio** - Resamples, normalizes, removes silence
3. **Extracts features** - Computes prosodic and spectral features
4. **Trains model** - Trains the selected classifier
5. **Evaluates** - Computes metrics and generates visualizations
6. **Saves results** - All outputs saved to `results/` directory

## Results Location

All results are saved in `results/` directory:
- `{experiment_name}_metrics.json` - Evaluation metrics
- `{experiment_name}_confusion_matrix.png` - Confusion matrix
- `{experiment_name}_roc_curves.png` - ROC curves
- `{experiment_name}_model.pkl` - Trained model
- `{experiment_name}_features.csv` - Extracted features
- `{experiment_name}.log` - Experiment log

## Using Different Models

Edit `configs/experiment.yaml` and change:
```yaml
model:
  model_type: "logistic_regression"  # or "svm", "random_forest", "xgboost"
```

**Note:** XGBoost requires OpenMP on macOS:
```bash
brew install libomp
```

## Using the Jupyter Notebook

For interactive analysis:
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Troubleshooting

**No datasets found:**
- Make sure you've downloaded at least one dataset
- Check that zip files are in the correct `data/` subdirectories

**Parselmouth errors:**
- This is normal on Python 3.12
- The code automatically uses librosa-based F0 extraction
- All features will still be extracted

**XGBoost errors:**
- Use a different model type (logistic_regression, svm, random_forest)
- Or install OpenMP: `brew install libomp`

**Memory errors:**
- Process fewer files at once
- Reduce feature extraction options in config
- Use a smaller dataset

