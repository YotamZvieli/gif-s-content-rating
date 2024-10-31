import glob
import os

import evaluate
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from tqdm import tqdm as t
from transformers import TrainingArguments
from transformers import XCLIPProcessor, XCLIPModel, Trainer


class GIFDataset(Dataset):
    def __init__(self, root_dir, classes, processor, clip_len=8):
        self.videos = []
        self.labels = []
        self.classes = classes
        self.processor = processor
        self.clip_len = clip_len
        self.class_to_label = {'g': 2, 'pg': 2, 'pg-13': 1, 'r': 1}

        for c in classes:
            class_videos = glob.glob(os.path.join(root_dir, c, "*.avi"))
            cnt = 0
            for gif in t(class_videos):
                cnt += 1
                if cnt >= 50:
                    break
                vr = VideoReader(gif, num_threads=1, ctx=cpu(0), width=224, height=224)
                vr.seek(0)
                frame_sample_rate = len(vr) // self.clip_len
                indices = sample_frame_indices(clip_len=self.clip_len, frame_sample_rate=frame_sample_rate,
                                               seg_len=len(vr))
                try:
                    video = vr.get_batch(indices).asnumpy()
                except Exception:
                    print("Skipped video:", gif)
                    continue

                # Convert frames to RGB PIL images
                video = [Image.fromarray(frame).convert('RGB') for frame in video]
                self.videos.append(video)
                self.labels.append(self.class_to_label[c])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # Process video with processor
        inputs = self.processor(
            text=["sex", "violence", "Suitable for children"],
            videos=[self.videos[idx]],
            return_tensors="pt",
            padding=True
        )

        # Extract tensors and prepare labels
        video_data = {k: v[0] for k, v in inputs.items() if k != 'text_embeds'}
        video_data["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        print(type(video_data['labels']), video_data['labels'].dtype)

        # Debugging output
        print(f"Video Data: {video_data}")
        print(f"Labels: {video_data['labels']}")

        return video_data


from torch.nn import CrossEntropyLoss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)

        # Get the logits from model output
        logits = outputs.get("logits_per_video")

        # Get the labels from inputs
        labels = inputs.get("labels")

        # Compute the loss with CrossEntropyLoss
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = seg_len #Changed to always sample the last frames, this way less likely to sample out of bounds
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64) #Changed to int64, as this is what decord expects
    return indices


if __name__ == '__main__':
    model_name = "microsoft/xclip-base-patch32"
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)

    train_path = "gifs/DL project - gifs/train_avi/"
    val_path = "gifs/DL project - gifs/test_avi/"
    classes = ['g', 'pg', 'pg-13', 'r']

    train_dataset = GIFDataset(train_path, classes, processor)
    val_dataset = GIFDataset(val_path, classes, processor)

    print(train_dataset[0])

    os.environ["WANDB_DISABLED"] = "true"

    metric = evaluate.load("accuracy")


    training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

