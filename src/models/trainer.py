"""Model training pipeline with cross-validation and speaker-independent splits."""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)
from typing import Dict, Any, Tuple, Optional, List
import logging
import pickle
from pathlib import Path

from .model_factory import create_model

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate models with cross-validation."""
    
    def __init__(
        self,
        model_type: str = "logistic_regression",
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        test_size: float = 0.2,
        val_size: float = 0.1,
        cv_folds: int = 5,
        speaker_independent: bool = True
    ):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model to train
            hyperparameters: Model hyperparameters
            random_state: Random seed
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            cv_folds: Number of CV folds
            speaker_independent: Whether to use speaker-independent splits
        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        self.cv_folds = cv_folds
        self.speaker_independent = speaker_independent
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
    
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        label_column: str = "label",
        speaker_column: str = "speaker_id"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            features_df: DataFrame with features and labels
            label_column: Name of label column
            speaker_column: Name of speaker ID column
            
        Returns:
            Tuple of (X, y, speaker_ids, feature_names)
        """
        # Separate features and labels
        feature_columns = [col for col in features_df.columns 
                          if col not in [label_column, speaker_column, 'file_path', 'dataset']]
        self.feature_columns = feature_columns
        
        X = features_df[feature_columns].values
        y = features_df[label_column].values
        speaker_ids = features_df[speaker_column].values if speaker_column in features_df.columns else None
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded, speaker_ids, feature_columns
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        speaker_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Labels
            speaker_ids: Optional speaker IDs for speaker-independent split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.speaker_independent and speaker_ids is not None:
            # Speaker-independent split
            unique_speakers = np.unique(speaker_ids)
            np.random.seed(self.random_state)
            np.random.shuffle(unique_speakers)
            
            n_test_speakers = int(len(unique_speakers) * self.test_size)
            test_speakers = set(unique_speakers[:n_test_speakers])
            
            test_mask = np.array([sp in test_speakers for sp in speaker_ids])
            train_mask = ~test_mask
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Dictionary with training metrics
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create model
        self.model = create_model(
            self.model_type,
            self.hyperparameters,
            self.random_state
        )
        
        # Train with early stopping if applicable
        if self.model_type == "xgboost" and X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        # Training metrics
        y_pred_train = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train, average='macro')
        
        metrics = {
            'train_accuracy': train_acc,
            'train_f1_macro': train_f1
        }
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_pred_val = self.model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, y_pred_val)
            val_f1 = f1_score(y_val, y_pred_val, average='macro')
            metrics['val_accuracy'] = val_acc
            metrics['val_f1_macro'] = val_f1
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        speaker_ids: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            speaker_ids: Optional speaker IDs
            
        Returns:
            Dictionary with CV metrics
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model
        model = create_model(
            self.model_type,
            self.hyperparameters,
            self.random_state
        )
        
        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        cv_scores = cross_val_score(
            model, X_scaled, y,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        return {
            'cv_f1_mean': np.mean(cv_scores),
            'cv_f1_std': np.std(cv_scores),
            'cv_scores': cv_scores.tolist()
        }
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Unweighted average recall (UAR)
        from sklearn.metrics import recall_score
        uar = recall_score(y_test, y_pred, average='macro')
        
        # Per-class metrics
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'uar': uar,
            'f1_per_class': f1_per_class.tolist(),
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
        
        return metrics
    
    def save_model(self, save_path: str) -> None:
        """
        Save trained model and preprocessors.
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load trained model and preprocessors.
        
        Args:
            load_path: Path to load model from
        """
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.hyperparameters = model_data['hyperparameters']
        
        logger.info(f"Model loaded from {load_path}")

