from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from .labeled_gif_dataset import LabeledGIFDataset
import evaluate
import torch
import numpy as np
import pytorchvideo.data
import copy
import pathlib
import os

os.environ["WANDB_DISABLED"] = 'true'

def preparing_datasets(datasets_root_path, image_processor, num_frames_to_sample,
                       sample_rate, fps):
    mean = image_processor.image_mean
    std = image_processor.image_std

    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    clip_duration = num_frames_to_sample * sample_rate / fps

    # Training dataset transformations.
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        # RandomShortSideScale(min_size=224, max_size=280),
                        # RandomCrop(resize_to),
                        Resize(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    # Training dataset.
    train_dataset = LabeledGIFDataset(
        data_path=os.path.join(datasets_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    # Validation and evaluation datasets' transformations.
    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    # Validation and evaluation datasets.
    val_dataset = LabeledGIFDataset(
        data_path=os.path.join(datasets_root_path, "validation"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    test_dataset = LabeledGIFDataset(
        data_path=os.path.join(datasets_root_path, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    return train_dataset, val_dataset, test_dataset

def compute_metrics(eval_preds):
    metric = evaluate.load("mae")
    predictions = np.argmax(eval_preds.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_preds.label_ids)

def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


if __name__ == '__main__':
    label2id = {'g': 0, 'pg': 1, 'pg-13': 2, 'r': 3}
    id2label = {v: k for k, v in label2id.items()}

    model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics" # pre-trained model from which to fine-tune
    batch_size = 8 # batch size for training and evaluation

    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Create a new layer by copying the last layer
    last_layer_idx = len(model.videomae.encoder.layer) - 1
    new_layer = copy.deepcopy(model.videomae.encoder.layer[last_layer_idx])

    # Add the new layer
    model.videomae.encoder.layer.append(new_layer)

    # Update the config to reflect the new number of layers
    model.config.num_hidden_layers += 1
    model.config.num_attention_heads += 1

    # for name, param in model.named_parameters():
    #     if "encoder.layer.11" not in name and "encoder.layer.12" not in name and "classifier" not in name:
    #         param.requires_grad = False

    datasets_root_path = '/content/drive/MyDrive/DL project - gifs/learning'
    datasets_root_path = pathlib.Path(datasets_root_path)
    train_dataset, val_dataset, test_dataset = preparing_datasets(datasets_root_path, image_processor,
                                                                  model.config.num_frames, 4, 30)

    new_model_name = f"fully-finetuned"
    num_epochs = 4

    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_strategy = "steps",
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better = False,
        push_to_hub=False,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()