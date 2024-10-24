import glob
import os
import numpy as np
import tensorflow as tf
import torch
from decord import VideoReader, cpu
from tqdm import tqdm as t
from ipywidgets import Video
from transformers import XCLIPProcessor, XCLIPModel


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len < seg_len:
        end_idx = np.random.randint(converted_len, seg_len)
    else:
        end_idx = seg_len
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


if __name__ == '__main__':
    model_name = "microsoft/xclip-base-patch32"
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)

    # process the data
    train_path = "gifs/DL project - gifs/train_avi/"
    classes = ['g', 'r']
    gpus = tf.config.experimental.list_physical_devices('GPU')
    videos = []
    labels = []
    for c in classes:
        for gif in t(glob.glob(os.path.join(train_path + c, "*.avi"))):
            vr = VideoReader(gif, num_threads=1, ctx=cpu(0))
            vr.seek(0)
            indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=len(vr))
            try:
                video = vr.get_batch(indices).asnumpy()
            except Exception:
                print("skipped video")
                continue
            videos.append(video)
            labels.append(c)

    for i in range(len(videos)):
        videos[i] = list(videos[i])

    inputs = processor(text=["sex", "violence", "Suitable for children"], videos=videos, return_tensors="pt",
                       padding=True)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_video.softmax(dim=1)

    print(len(probs), len(labels), len(videos))

    success = 0
    probs = list(probs)
    print(probs)
    for i in range(len(probs)):
        probs[i] = list(probs[i])
        ind = probs[i].index(max(probs[i]))
        if ind < 2 and (labels[i] == 'pg-13' or labels[i] == 'r'):
            success += 1
        elif ind == 2 and (labels[i] == 'g' or labels[i] == 'pg'):
            success += 1

    print((success / len(labels)) * 100)