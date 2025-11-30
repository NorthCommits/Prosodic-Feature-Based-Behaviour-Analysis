"""
Generate a radar (spider) plot of prosodic features for IEEE publication.

This script loads prosodic features from a CSV file, normalizes them,
and creates a publication-quality radar chart visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set matplotlib parameters for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5

# Path to CSV file
csv_path = Path("results/prosodic_behavior_classification_features.csv")

# Load the CSV file
print(f"Loading features from {csv_path}...")
df = pd.read_csv(csv_path)

# Select the first sample (first row)
sample = df.iloc[0].copy()
print(f"Selected sample: {sample.get('file_path', 'Unknown')}")

# Define prosodic feature columns to extract
prosodic_features = {
    # F0 statistics
    'f0_mean': 'F0 Mean',
    'f0_std': 'F0 Std',
    'f0_min': 'F0 Min',
    'f0_max': 'F0 Max',
    'f0_range': 'F0 Range',
    # Voice quality
    'jitter_local': 'Jitter Local',
    'jitter_local_abs': 'Jitter Local Abs',
    'shimmer_local': 'Shimmer Local',
    # Energy (using intensity as it's the available feature)
    'intensity_mean': 'Intensity Mean',
    'intensity_std': 'Intensity Std',
    # Timing
    'speaking_rate': 'Speaking Rate',
    'pause_count': 'Pause Count',
    'avg_pause_duration': 'Pause Duration',
    # Voicing
    'voicing_ratio': 'Voicing Ratio'
}

# Extract feature values
feature_values = []
feature_labels = []

for col_name, display_name in prosodic_features.items():
    if col_name in sample.index:
        feature_values.append(sample[col_name])
        feature_labels.append(display_name)
    else:
        print(f"Warning: Column '{col_name}' not found in CSV")

# Convert to numpy array
feature_values = np.array(feature_values)

# Normalize features to 0-1 range for visualization
# Using min-max normalization: (x - min) / (max - min)
# For each feature, we normalize based on the range in the entire dataset
normalized_values = np.zeros_like(feature_values)

for i, col_name in enumerate(prosodic_features.keys()):
    if col_name in df.columns:
        col_data = df[col_name].values
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        
        # Handle case where min == max (constant feature)
        if max_val == min_val:
            normalized_values[i] = 0.5  # Set to middle value
        else:
            normalized_values[i] = (feature_values[i] - min_val) / (max_val - min_val)
    else:
        normalized_values[i] = 0.0

# Number of features
num_features = len(feature_labels)

# Create angles for radar chart
angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()

# Close the polygon by repeating the first value
normalized_values = np.concatenate((normalized_values, [normalized_values[0]]))
angles += angles[:1]

# Create figure with white background
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot the radar chart
ax.plot(angles, normalized_values, 'o-', linewidth=2, color='#2E86AB', markersize=6)
ax.fill(angles, normalized_values, alpha=0.25, color='#2E86AB')

# Set the labels for each angle
ax.set_xticks(angles[:-1])
ax.set_xticklabels(feature_labels, fontsize=9)

# Set radial limits (0 to 1 for normalized values)
ax.set_ylim(0, 1)

# Add grid lines
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')

# Remove radial labels for cleaner look
ax.set_ylabel('')

# Note: No title added as IEEE papers use captions instead of figure titles

# Create figures directory if it doesn't exist
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# Save the figure
output_path = figures_dir / "prosodic_features_speech.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nRadar plot saved to: {output_path}")

# Display summary statistics
print("\nFeature values (original):")
for i, (col_name, display_name) in enumerate(prosodic_features.items()):
    if col_name in sample.index:
        print(f"  {display_name}: {feature_values[i]:.4f} (normalized: {normalized_values[i]:.4f})")

plt.close()

