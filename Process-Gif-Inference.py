import os
from glob import glob
import torch.nn as nn
import torch
from PIL import Image
import imageio
import random
from torchvision import transforms
import open_clip

def process_gif_to_embeddings(gif_path, model, preprocess, device='cuda', num_frames=30):
    """
    Process a GIF to extract embeddings for all frames.

    Args:
        gif_path (str): Path to the GIF file.
        model (torch.nn.Module): Preloaded OpenCLIP model.
        preprocess (callable): Preprocessing function for the model.
        device (str): Device to use ('cuda' or 'cpu').
        num_frames (int): Number of frames to extract from the GIF.

    Returns:
        torch.Tensor: Combined embeddings for all frames as a single vector.
    """
    try:
        # Extract frames from the GIF
        reader = imageio.get_reader(gif_path, format='GIF')
        total_frames = len(reader)

        if total_frames < num_frames:
            # Repeat frames if fewer than required
            frame_indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
        else:
            frame_indices = sorted(random.sample(range(total_frames), num_frames))

        frames = [Image.fromarray(reader.get_data(idx)).convert("RGB") for idx in frame_indices]
        reader.close()

        # Preprocess frames
        images = torch.stack([preprocess(frame) for frame in frames]).to(device)

        # Extract embeddings using OpenCLIP
        with torch.no_grad():
            embeddings = model.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize

        # Flatten embeddings into a single vector (e.g., [30, 512] -> [15360])
        return embeddings.view(-1)
    except Exception as e:
        print(f"Error processing GIF {gif_path}: {e}")
        return None



class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)  # Batch Normalization
        self.activation = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU
        self.dropout = nn.Dropout(dropout_prob)

        # Projection layer to align dimensions if needed
        self.projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Apply projection if dimensions don't match
        residual = self.projection(x) if self.projection else x
        return residual + out  # Residual connection


class ImprovedVideoClassifier(nn.Module):
    def __init__(self, input_dim=512,
                 hidden_dim=512,  # Fixed hidden dimension for all layers
                 num_layers=2,   # Number of residual blocks
                 output_dim=4,
                 dropout_prob=0.1):
        super(ImprovedVideoClassifier, self).__init__()

        layers = []
        current_dim = input_dim

        # Add residual blocks dynamically
        for _ in range(num_layers):
            layers.append(ResidualBlock(current_dim, hidden_dim, dropout_prob))
            current_dim = hidden_dim  # Update current dimension after each block

        # Add the final output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Wrap the layers in nn.Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.network(x)

def process_gif(gif_path):
    # Load the trained classifier
    model_checkpoint_path = '/content/drive/MyDrive/data/learning_3/model_checkpoint.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the architecture matches the one used during training
    classifier = ImprovedVideoClassifier(
        input_dim=30 * 512,  # Match input dimensions from training
        hidden_dim=1024,  # Match hidden dimensions
        output_dim=2,  # Match output classes
        num_layers=10,  # Match number of layers
        dropout_prob=0.5  # Match dropout
    )
    classifier.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()

    # Process the GIF to extract embeddings
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    model.to(device)

    embeddings = process_gif_to_embeddings(gif_path, model, preprocess, device=device, num_frames=30)

    # Feed embeddings into the classifier
    if embeddings is not None:
        embeddings = embeddings.unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            output = classifier(embeddings)
            _, predicted_class = torch.max(output, 1)

        print("Model Output (Raw Scores):", output)
        print("Predicted Class:", predicted_class.item())
    else:
        print("Failed to process GIF.")