import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


def extract_frames(gif_path, width, height, num_frames=10):
    frames = []
    with Image.open(gif_path) as img:
        skip_frames_window = max(img.n_frames // num_frames, 1)
        for i in range(0, img.n_frames, skip_frames_window):
            img.seek(i)
            frame = img.convert('RGB')
            frame = frame.resize((width, height))
            frame_array = img_to_array(frame)
            frames.append(frame_array)

    img.close()
    
    # Pad with zeros if there are fewer than max_frames
    while len(frames) < num_frames:
        frames.append(np.zeros((width, height, 3)))
    
    return np.array(frames[:num_frames])
