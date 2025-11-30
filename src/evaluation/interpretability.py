"""Interpretability module using SHAP for model explanations."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class InterpretabilityAnalyzer:
    """Analyze model interpretability using SHAP."""
    
    def __init__(self, results_dir: str = "./results"):
        """
        Initialize interpretability analyzer.
        
        Args:
            results_dir: Directory to save SHAP plots
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for interpretability analysis. Install with: pip install shap")
    
    def explain_model(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        model_type: str,
        experiment_name: str,
        n_samples: int = 100
    ) -> Dict[str, str]:
        """
        Generate SHAP explanations for a model.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            model_type: Type of model
            experiment_name: Name of experiment
            n_samples: Number of samples for SHAP (use subset for speed)
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        # Sample subset for faster computation
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create appropriate explainer based on model type
        if model_type in ["random_forest", "xgboost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # For linear models, use KernelExplainer
            explainer = shap.KernelExplainer(
                model.predict_proba,
                X_sample[:10]  # Background data
            )
            shap_values = explainer.shap_values(X_sample)
        
        saved_paths = {}
        
        # Summary plot
        summary_path = self._plot_summary(
            shap_values, X_sample, feature_names, experiment_name
        )
        saved_paths['summary'] = summary_path
        
        # Feature importance plot
        importance_path = self._plot_feature_importance(
            shap_values, feature_names, experiment_name
        )
        saved_paths['importance'] = importance_path
        
        # Waterfall plot for a sample (skip if multi-class to avoid shape issues)
        try:
            if isinstance(shap_values, list):
                # For multi-class, use first class's values for first sample
                shap_values_single = shap_values[0][0]  # First class, first sample
                expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else (explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
            else:
                shap_values_single = shap_values[0]  # First sample
                expected_val = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0]
            
            waterfall_path = self._plot_waterfall(
                explainer, X_sample[0], shap_values_single,
                expected_val, feature_names, experiment_name
            )
            saved_paths['waterfall'] = waterfall_path
        except Exception as e:
            logger.warning(f"Could not generate waterfall plot: {str(e)}")
            saved_paths['waterfall'] = None
        
        return saved_paths
    
    def _plot_summary(
        self,
        shap_values: Any,
        X: np.ndarray,
        feature_names: List[str],
        experiment_name: str
    ) -> str:
        """Plot SHAP summary plot."""
        plt.figure()
        
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
        else:
            shap_values_plot = shap_values
        
        shap.summary_plot(
            shap_values_plot,
            X,
            feature_names=feature_names,
            show=False,
            plot_size=(10, 8)
        )
        
        fig_path = self.results_dir / f"{experiment_name}_shap_summary.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP summary plot saved to {fig_path}")
        return str(fig_path)
    
    def _plot_feature_importance(
        self,
        shap_values: Any,
        feature_names: List[str],
        experiment_name: str
    ) -> str:
        """Plot SHAP feature importance."""
        if isinstance(shap_values, list):
            shap_values_mean = np.abs(shap_values[0]).mean(0)
        else:
            shap_values_mean = np.abs(shap_values).mean(0)
        
        # Sort by importance
        indices = np.argsort(shap_values_mean)[::-1][:20]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), shap_values_mean[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'SHAP Feature Importance - {experiment_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        fig_path = self.results_dir / f"{experiment_name}_shap_importance.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP importance plot saved to {fig_path}")
        return str(fig_path)
    
    def _plot_waterfall(
        self,
        explainer: Any,
        instance: np.ndarray,
        shap_values_instance: np.ndarray,
        expected_value: Any,
        feature_names: List[str],
        experiment_name: str
    ) -> str:
        """Plot SHAP waterfall plot for a single instance."""
        plt.figure(figsize=(10, 8))
        
        # Ensure shap_values_instance is 1D
        if shap_values_instance.ndim > 1:
            shap_values_instance = shap_values_instance.flatten()
        
        # Ensure instance is 1D
        if instance.ndim > 1:
            instance = instance.flatten()
        
        # Handle expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            if len(expected_value) > 0:
                expected_value = expected_value[0] if isinstance(expected_value, list) else float(expected_value.flat[0])
            else:
                expected_value = 0.0
        else:
            expected_value = float(expected_value)
        
        # Create SHAP Explanation object
        try:
            shap_explanation = shap.Explanation(
                values=shap_values_instance,
                base_values=expected_value,
                data=instance,
                feature_names=feature_names
            )
            
            shap.waterfall_plot(shap_explanation, show=False)
        except Exception as e:
            # Fallback: use bar plot if waterfall fails
            logger.warning(f"Waterfall plot failed, using bar plot instead: {str(e)}")
            plt.clf()
            indices = np.argsort(np.abs(shap_values_instance))[::-1][:20]
            plt.barh(range(len(indices)), shap_values_instance[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('SHAP Value')
            plt.title(f'SHAP Values (Top 20) - {experiment_name}')
            plt.gca().invert_yaxis()
        
        fig_path = self.results_dir / f"{experiment_name}_shap_waterfall.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP waterfall plot saved to {fig_path}")
        return str(fig_path)
    
    def explain_misclassified(
        self,
        model: Any,
        X_misclassified: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str],
        model_type: str,
        experiment_name: str,
        n_examples: int = 5
    ) -> str:
        """
        Explain misclassified examples.
        
        Args:
            model: Trained model
            X_misclassified: Feature matrix for misclassified examples
            y_true: True labels
            y_pred: Predicted labels
            feature_names: List of feature names
            model_type: Type of model
            experiment_name: Name of experiment
            n_examples: Number of examples to explain
            
        Returns:
            Path to saved plot
        """
        n_examples = min(n_examples, len(X_misclassified))
        
        if model_type in ["random_forest", "xgboost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_misclassified[:n_examples])
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba,
                X_misclassified[:10]
            )
            shap_values = explainer.shap_values(X_misclassified[:n_examples])
        
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
        else:
            shap_values_plot = shap_values
        
        plt.figure()
        shap.summary_plot(
            shap_values_plot,
            X_misclassified[:n_examples],
            feature_names=feature_names,
            show=False,
            plot_size=(10, 8)
        )
        
        fig_path = self.results_dir / f"{experiment_name}_shap_misclassified.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP misclassified examples plot saved to {fig_path}")
        return str(fig_path)

