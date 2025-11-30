"""Visualization utilities for feature analysis and model interpretation."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


class Plotter:
    """Utility class for creating publication-ready visualizations."""
    
    def __init__(self, results_dir: str = "./results"):
        """
        Initialize plotter.
        
        Args:
            results_dir: Directory to save plots
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_feature_distributions(
        self,
        features_df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str = "label",
        n_features: int = 12,
        experiment_name: str = "feature_distributions"
    ) -> str:
        """
        Plot distributions of features by class.
        
        Args:
            features_df: DataFrame with features and labels
            feature_columns: List of feature column names
            label_column: Name of label column
            n_features: Number of features to plot
            experiment_name: Name for saving figure
            
        Returns:
            Path to saved figure
        """
        n_features = min(n_features, len(feature_columns))
        selected_features = feature_columns[:n_features]
        
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(selected_features):
            ax = axes[i]
            
            # Plot distribution for each class
            for label in features_df[label_column].unique():
                data = features_df[features_df[label_column] == label][feature]
                ax.hist(data, alpha=0.5, label=label, bins=30)
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(feature)
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        fig_path = self.results_dir / f"{experiment_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature distributions plot saved to {fig_path}")
        return str(fig_path)
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 20,
        experiment_name: str = "feature_importance"
    ) -> str:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            top_n: Number of top features to show
            experiment_name: Name for saving figure
            
        Returns:
            Path to saved figure
        """
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance - {experiment_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        fig_path = self.results_dir / f"{experiment_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {fig_path}")
        return str(fig_path)
    
    def plot_correlation_matrix(
        self,
        features_df: pd.DataFrame,
        feature_columns: List[str],
        experiment_name: str = "correlation_matrix"
    ) -> str:
        """
        Plot correlation matrix of features.
        
        Args:
            features_df: DataFrame with features
            feature_columns: List of feature column names
            experiment_name: Name for saving figure
            
        Returns:
            Path to saved figure
        """
        # Select subset if too many features
        if len(feature_columns) > 30:
            feature_columns = feature_columns[:30]
        
        corr_matrix = features_df[feature_columns].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'label': 'Correlation'}
        )
        plt.title(f'Feature Correlation Matrix - {experiment_name}')
        plt.tight_layout()
        
        fig_path = self.results_dir / f"{experiment_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlation matrix saved to {fig_path}")
        return str(fig_path)

