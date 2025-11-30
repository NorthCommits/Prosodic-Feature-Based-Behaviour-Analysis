#!/bin/bash
# Installation script for parselmouth to avoid dependency issues
# Note: Parselmouth has known compatibility issues with Python 3.12
# The code will work fine without it, using librosa-based fallbacks

echo "Attempting to install parselmouth (may fail on Python 3.12)..."

# First, ensure setuptools and wheel are installed
echo "Installing build dependencies..."
pip install setuptools wheel build stopit

# Install parselmouth without dependencies to skip problematic googleads package
echo "Installing parselmouth (skipping problematic dependencies)..."
pip install parselmouth --no-deps 2>&1 | grep -v "ERROR: pip's dependency resolver"

# Verify installation
python -c "import parselmouth; print('✓ Parselmouth installed successfully')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ Parselmouth installation complete!"
else
    echo "⚠ Parselmouth installation failed (expected on Python 3.12)."
    echo "  The code will automatically use librosa-based F0 extraction instead."
    echo "  All features will still be extracted - no functionality is lost!"
fi

