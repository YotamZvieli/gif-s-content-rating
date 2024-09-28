import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


def extract_frames(gif_path, max_frames=10):
    frames = []
    with Image.open(gif_path) as img:
        skip_frames_window = max(img.n_frames // max_frames, 1)
        for i in range(0, img.n_frames, skip_frames_window):
            img.seek(i)
            frame = img.convert('RGB')
            frame = frame.resize((128, 128))
            frame_array = img_to_array(frame)
            frames.append(frame_array)
    
    # Pad with zeros if there are fewer than max_frames
    while len(frames) < max_frames:
        frames.append(np.zeros((128, 128, 3)))
    
    return np.array(frames)