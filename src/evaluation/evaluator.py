"""Comprehensive evaluation module with metrics and visualizations."""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate models and generate metrics and visualizations."""
    
    def __init__(self, results_dir: str = "./results"):
        """
        Initialize evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        experiment_name: str,
        fold: Optional[int] = None
    ) -> str:
        """
        Save evaluation metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics
            experiment_name: Name of experiment
            fold: Optional fold number for CV
            
        Returns:
            Path to saved metrics file
        """
        filename = f"{experiment_name}_metrics"
        if fold is not None:
            filename += f"_fold{fold}"
        filename += ".json"
        
        metrics_path = self.results_dir / filename
        
        # Convert numpy types to native Python types for JSON
        metrics_serializable = self._make_serializable(metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_path}")
        return str(metrics_path)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list,
        experiment_name: str,
        fold: Optional[int] = None,
        normalize: bool = True
    ) -> str:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            experiment_name: Name of experiment
            fold: Optional fold number
            normalize: Whether to normalize confusion matrix
            
        Returns:
            Path to saved figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )
        plt.title(f'Confusion Matrix - {experiment_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = f"{experiment_name}_confusion_matrix"
        if fold is not None:
            filename += f"_fold{fold}"
        filename += ".png"
        
        fig_path = self.results_dir / filename
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {fig_path}")
        return str(fig_path)
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: list,
        experiment_name: str,
        fold: Optional[int] = None
    ) -> str:
        """
        Plot ROC curves for each class.
        
        Args:
            y_true: True labels (encoded)
            y_pred_proba: Predicted probabilities
            class_names: List of class names
            experiment_name: Name of experiment
            fold: Optional fold number
            
        Returns:
            Path to saved figure
        """
        n_classes = len(class_names)
        
        plt.figure(figsize=(10, 8))
        
        # Compute ROC curve for each class
        for i, class_name in enumerate(class_names):
            # Binarize labels for this class
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred_proba[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.2f})',
                linewidth=2
            )
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {experiment_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = f"{experiment_name}_roc_curves"
        if fold is not None:
            filename += f"_fold{fold}"
        filename += ".png"
        
        fig_path = self.results_dir / filename
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {fig_path}")
        return str(fig_path)
    
    def generate_report(
        self,
        metrics: Dict[str, Any],
        experiment_name: str,
        fold: Optional[int] = None
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: Dictionary of metrics
            experiment_name: Name of experiment
            fold: Optional fold number
            
        Returns:
            Path to saved report
        """
        report_lines = [
            f"Evaluation Report: {experiment_name}",
            "=" * 60,
            ""
        ]
        
        if fold is not None:
            report_lines.append(f"Fold: {fold}\n")
        
        # Overall metrics
        report_lines.extend([
            "Overall Metrics:",
            f"  Accuracy: {metrics.get('accuracy', 0):.4f}",
            f"  Macro F1: {metrics.get('f1_macro', 0):.4f}",
            f"  UAR (Unweighted Average Recall): {metrics.get('uar', 0):.4f}",
            ""
        ])
        
        # Per-class metrics
        if 'classification_report' in metrics:
            report_lines.append("Per-Class Metrics:")
            report = metrics['classification_report']
            
            for class_name in metrics.get('f1_per_class', []):
                if class_name in report:
                    prec = report[class_name]['precision']
                    rec = report[class_name]['recall']
                    f1 = report[class_name]['f1-score']
                    report_lines.append(
                        f"  {class_name}: Precision={prec:.4f}, "
                        f"Recall={rec:.4f}, F1={f1:.4f}"
                    )
            report_lines.append("")
        
        # Save report
        filename = f"{experiment_name}_report"
        if fold is not None:
            filename += f"_fold{fold}"
        filename += ".txt"
        
        report_path = self.results_dir / filename
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to {report_path}")
        return str(report_path)
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object to make serializable
            
        Returns:
            Serializable object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

