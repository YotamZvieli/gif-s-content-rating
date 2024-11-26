#!/bin/bash

# Install required Python packages
pip install torch
pip install pillow
pip install numpy
pip install open_clip_torch

# Notify the user
echo "Running...."

# Run the Python script
python gifs-rate-server.py
