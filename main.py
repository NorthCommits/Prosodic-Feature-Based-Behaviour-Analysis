"""Main CLI script for running prosodic feature-based behavior classification experiments."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import load_config
from utils.logger import setup_logger
from data_loader import load_ravdess, load_crema_d, load_savee
from preprocessing import AudioPreprocessor
from feature_extraction import ProsodicFeatureExtractor
from models import ModelTrainer
from evaluation import Evaluator
from visualization import Plotter

try:
    from evaluation.interpretability import InterpretabilityAnalyzer
except ImportError:
    InterpretabilityAnalyzer = None


def load_datasets(config, logger):
    """
    Load all specified datasets.
    
    Args:
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        Combined DataFrame with all datasets
    """
    all_datasets = []
    
    if "ravdess" in config.datasets:
        logger.info("Loading RAVDESS dataset...")
        try:
            df_ravdess = load_ravdess(
                data_dir=f"{config.data_dir}/ravdess",
                download=True
            )
            all_datasets.append(df_ravdess)
        except Exception as e:
            logger.warning(f"Failed to load RAVDESS: {str(e)}")
    
    if "crema_d" in config.datasets:
        logger.info("Loading CREMA-D dataset...")
        try:
            df_crema = load_crema_d(
                data_dir=f"{config.data_dir}/crema_d",
                download=True
            )
            all_datasets.append(df_crema)
        except Exception as e:
            logger.warning(f"Failed to load CREMA-D: {str(e)}")
    
    if "savee" in config.datasets:
        logger.info("Loading SAVEE dataset...")
        try:
            df_savee = load_savee(
                data_dir=f"{config.data_dir}/savee",
                download=True
            )
            all_datasets.append(df_savee)
        except Exception as e:
            logger.warning(f"Failed to load SAVEE: {str(e)}")
    
    if not all_datasets:
        raise ValueError("No datasets were successfully loaded.")
    
    # Combine datasets
    combined_df = pd.concat(all_datasets, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} total samples from {len(all_datasets)} datasets")
    
    return combined_df


def preprocess_audio(data_df, config, logger):
    """
    Preprocess audio files.
    
    Args:
        data_df: DataFrame with file paths
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        DataFrame with processed audio paths (or original if preprocessing disabled)
    """
    preprocessor = AudioPreprocessor(
        target_sr=config.preprocessing.target_sr,
        normalize_loudness=config.preprocessing.normalize_loudness,
        trim_silence=config.preprocessing.trim_silence,
        vad_threshold=config.preprocessing.vad_threshold,
        noise_reduction=config.preprocessing.noise_reduction,
        max_length_seconds=config.preprocessing.max_length_seconds,
        chunk_long_utterances=config.preprocessing.chunk_long_utterances,
        chunk_size_seconds=config.preprocessing.chunk_size_seconds
    )
    
    logger.info("Preprocessing audio files...")
    processed_paths = []
    
    for idx, row in data_df.iterrows():
        try:
            # For now, use original paths (preprocessing can be added later)
            processed_paths.append(row['file_path'])
        except Exception as e:
            logger.warning(f"Error preprocessing {row['file_path']}: {str(e)}")
            processed_paths.append(None)
    
    data_df['processed_path'] = processed_paths
    data_df = data_df[data_df['processed_path'].notna()]
    
    logger.info(f"Preprocessed {len(data_df)} files")
    return data_df


def extract_features(data_df, config, logger):
    """
    Extract features from audio files.
    
    Args:
        data_df: DataFrame with file paths
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        DataFrame with extracted features
    """
    extractor = ProsodicFeatureExtractor(
        extract_prosodic=config.feature_extraction.extract_prosodic,
        extract_spectral=config.feature_extraction.extract_spectral,
        extract_mfcc=config.feature_extraction.extract_mfcc,
        extract_formants=config.feature_extraction.extract_formants,
        mfcc_n_coeffs=config.feature_extraction.mfcc_n_coeffs,
        frame_length=config.feature_extraction.frame_length,
        hop_length=config.feature_extraction.hop_length
    )
    
    logger.info("Extracting features...")
    features_list = []
    
    for idx, row in data_df.iterrows():
        try:
            features = extractor.extract_from_file(row['processed_path'])
            features['label'] = row['label']
            features['speaker_id'] = row['speaker_id']
            features['dataset'] = row['dataset']
            features['file_path'] = row['file_path']
            features_list.append(features)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(data_df)} files")
                
        except Exception as e:
            logger.warning(f"Error extracting features from {row['processed_path']}: {str(e)}")
    
    features_df = pd.DataFrame(features_list)
    logger.info(f"Extracted features for {len(features_df)} files")
    logger.info(f"Feature dimensions: {len(features_df.columns) - 4} features")
    
    return features_df


def train_and_evaluate(features_df, config, logger):
    """
    Train model and evaluate performance.
    
    Args:
        features_df: DataFrame with features and labels
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        Dictionary with evaluation results
    """
    # Initialize trainer
    trainer = ModelTrainer(
        model_type=config.model.model_type,
        hyperparameters=getattr(config.model, config.model.model_type, {}),
        random_state=config.seed,
        test_size=config.model.test_size,
        val_size=config.model.val_size,
        cv_folds=config.model.cv_folds,
        speaker_independent=config.model.speaker_independent
    )
    
    # Prepare data
    X, y, speaker_ids, feature_names = trainer.prepare_data(features_df)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, speaker_ids)
    
    # Further split training for validation
    from sklearn.model_selection import train_test_split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train,
        test_size=config.model.val_size,
        random_state=config.seed,
        stratify=y_train
    )
    
    # Train model
    logger.info(f"Training {config.model.model_type} model...")
    train_metrics = trainer.train(X_train_final, y_train_final, X_val, y_val)
    logger.info(f"Training metrics: {train_metrics}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test)
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"Test UAR: {test_metrics['uar']:.4f}")
    
    # Cross-validation
    if config.model.cv_folds > 1:
        logger.info("Performing cross-validation...")
        cv_metrics = trainer.cross_validate(X_train, y_train, speaker_ids)
        logger.info(f"CV F1 (mean ± std): {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")
        test_metrics['cv_metrics'] = cv_metrics
    
    # Save model
    model_path = Path(config.results_dir) / f"{config.experiment_name}_model.pkl"
    trainer.save_model(str(model_path))
    
    return {
        'trainer': trainer,
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'label_encoder': trainer.label_encoder
    }


def generate_visualizations(results, config, logger):
    """
    Generate evaluation visualizations.
    
    Args:
        results: Dictionary with evaluation results
        config: Experiment configuration
        logger: Logger instance
    """
    evaluator = Evaluator(results_dir=config.results_dir)
    
    # Save metrics
    evaluator.save_metrics(
        results['test_metrics'],
        config.experiment_name
    )
    
    # Generate report
    evaluator.generate_report(
        results['test_metrics'],
        config.experiment_name
    )
    
    # Plot confusion matrix
    class_names = results['label_encoder'].classes_
    evaluator.plot_confusion_matrix(
        results['y_test'],
        np.array(results['test_metrics']['y_pred']),
        class_names,
        config.experiment_name
    )
    
    # Plot ROC curves
    evaluator.plot_roc_curves(
        results['y_test'],
        np.array(results['test_metrics']['y_pred_proba']),
        class_names,
        config.experiment_name
    )
    
    logger.info("Visualizations generated and saved")


def generate_interpretability(results, config, logger):
    """
    Generate SHAP interpretability plots.
    
    Args:
        results: Dictionary with evaluation results
        config: Experiment configuration
        logger: Logger instance
    """
    if InterpretabilityAnalyzer is None:
        logger.warning("SHAP not available. Skipping interpretability analysis.")
        return
    
    try:
        analyzer = InterpretabilityAnalyzer(results_dir=config.results_dir)
        
        # Use subset of test data for SHAP
        X_test = results['X_test']
        n_samples = min(100, len(X_test))
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_shap = X_test[indices]
        
        shap_paths = analyzer.explain_model(
            results['trainer'].model,
            X_shap,
            results['feature_names'],
            config.model.model_type,
            config.experiment_name,
            n_samples=n_samples
        )
        
        logger.info(f"SHAP plots saved: {shap_paths}")
        
        # Explain misclassified examples
        y_pred = np.array(results['test_metrics']['y_pred'])
        misclassified_mask = results['y_test'] != y_pred
        if np.any(misclassified_mask):
            X_misclassified = X_test[misclassified_mask]
            y_true_mis = results['y_test'][misclassified_mask]
            y_pred_mis = y_pred[misclassified_mask]
            
            analyzer.explain_misclassified(
                results['trainer'].model,
                X_misclassified,
                y_true_mis,
                y_pred_mis,
                results['feature_names'],
                config.model.model_type,
                config.experiment_name
            )
        
    except ImportError:
        logger.warning("SHAP not available. Skipping interpretability analysis.")
    except Exception as e:
        logger.warning(f"Error in interpretability analysis: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prosodic Feature-Based Behavior Classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_file = Path(config.results_dir) / f"{config.experiment_name}.log"
    logger = setup_logger("main", log_file=str(log_file))
    
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Configuration loaded from: {args.config}")
    
    try:
        # Set random seed
        np.random.seed(config.seed)
        
        # Load datasets
        data_df = load_datasets(config, logger)
        
        # Preprocess audio
        data_df = preprocess_audio(data_df, config, logger)
        
        # Extract features
        features_df = extract_features(data_df, config, logger)
        
        # Save features
        features_path = Path(config.results_dir) / f"{config.experiment_name}_features.csv"
        features_df.to_csv(features_path, index=False)
        logger.info(f"Features saved to {features_path}")
        
        # Train and evaluate
        results = train_and_evaluate(features_df, config, logger)
        
        # Generate visualizations
        generate_visualizations(results, config, logger)
        
        # Generate interpretability plots
        generate_interpretability(results, config, logger)
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

