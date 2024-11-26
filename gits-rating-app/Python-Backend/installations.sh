#!/bin/bash

pip install torch
pip install pillow
pip install numpy
pip install open_clip_torch
pip install flask
pip install flask-cors
pip install imageio

echo "Running...."

python gifs-rate-server.py
