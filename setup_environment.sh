#!/bin/bash

# Setup script for py-ftf environment
echo "Setting up Python environment for py-ftf..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To run the py-ftf code, use: python py-ftf.py"
