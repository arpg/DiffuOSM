#!/bin/bash

# Create a Conda virtual environment
conda create --name scene_completion python=3.9 -y

# Activate the Conda virtual environment
conda activate scene_completion

# Install PyTorch with CUDA support
conda install torch

# Install other required packages
conda install numpy

# Deactivate the Conda virtual environment
conda deactivate

echo "Installation completed successfully."