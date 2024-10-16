import pathlib
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import pytorchvideo.data
import os
import imageio
import numpy as np
from IPython.display import Image
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

if __name__ == '__main__':
    dataset_root_path = "gifs/DL project - gifs"
    dataset_root_path = pathlib.Path(dataset_root_path)
    video_count_train = len(list(dataset_root_path.glob("train_avi/*/*.gif")))

    video_file_paths_class_g = list(dataset_root_path.glob("train_avi/g/*.gif"))
    video_file_paths_class_pg = list(dataset_root_path.glob("train_avi/pg/*.gif"))
    video_file_paths_class_pg13 = list(dataset_root_path.glob("train_avi/pg-13/*.gif"))
    video_file_paths_class_r = list(dataset_root_path.glob("train_avi/r/*.gif"))

    label_to_id = {"g": 0, "pg": 1, "pg-13": 2, "r": 3}
    id_to_label = {0: "g", 1: "pg", 2: "pg-13", 3: "r"}

    model_ckpt = "MCG-NJU/videomae-base"
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label_to_id,
        id2label=id_to_label,
        ignore_mismatched_sizes=True)

    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "train_avi"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )








