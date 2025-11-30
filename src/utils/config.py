"""Configuration loading and validation utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
import os


@dataclass
class PreprocessingConfig:
    """Configuration for audio preprocessing."""
    target_sr: int = 16000
    normalize_loudness: bool = True
    trim_silence: bool = True
    vad_threshold: float = 0.01
    noise_reduction: bool = True
    max_length_seconds: float = 10.0
    chunk_long_utterances: bool = False
    chunk_size_seconds: float = 5.0


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""
    extract_prosodic: bool = True
    extract_spectral: bool = True
    extract_mfcc: bool = True
    extract_formants: bool = True
    mfcc_n_coeffs: int = 13
    frame_length: int = 2048
    hop_length: int = 512


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str = "logistic_regression"
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    cv_folds: int = 5
    speaker_independent: bool = True
    cross_dataset_eval: bool = False
    
    # Model-specific hyperparameters
    logistic_regression: Dict[str, Any] = field(default_factory=lambda: {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000
    })
    svm: Dict[str, Any] = field(default_factory=lambda: {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000
    })
    random_forest: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced"
    })
    xgboost: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "early_stopping_rounds": 10
    })


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    experiment_name: str = "default_experiment"
    datasets: list = field(default_factory=lambda: ["ravdess", "crema_d", "savee"])
    data_dir: str = "./data"
    results_dir: str = "./results"
    seed: int = 42
    
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ExperimentConfig object with loaded settings
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Build nested config objects
    preprocessing_config = PreprocessingConfig(**config_dict.get("preprocessing", {}))
    feature_config = FeatureExtractionConfig(**config_dict.get("feature_extraction", {}))
    model_config = ModelConfig(**config_dict.get("model", {}))
    
    # Create main config
    main_config = config_dict.copy()
    main_config.pop("preprocessing", None)
    main_config.pop("feature_extraction", None)
    main_config.pop("model", None)
    
    experiment_config = ExperimentConfig(
        **main_config,
        preprocessing=preprocessing_config,
        feature_extraction=feature_config,
        model=model_config
    )
    
    return experiment_config


def get_openai_api_key() -> str:
    """Load OpenAI API key from .env file."""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    
    # Fallback to environment variable
    return os.getenv("OPENAI_API_KEY", "")

