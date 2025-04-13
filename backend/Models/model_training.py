import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.nn import functional as F

# ============================ #
#       Simple Model           #
# ============================ #

class DeepfakeDetectionModel(nn.Module):
    def __init__(self):
        super(DeepfakeDetectionModel, self).__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove classification layer
        self.fc = nn.Linear(512, 2)  # Binary classification (real vs fake)

    def forward(self, frames):
        # frames: (Batch, Frames, C, H, W)
        batch_size, num_frames, C, H, W = frames.size()
        frames = frames.view(batch_size * num_frames, C, H, W)  # Flatten frames for ResNet
        features = self.feature_extractor(frames)  # Extract features
        features = features.view(batch_size, num_frames, -1)  # Reshape for temporal analysis
        features = torch.mean(features, dim=1)  # Average pooling across frames
        output = self.fc(features)
        return output

# ============================ #
#       Training Loop          #
# ============================ #

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Train a machine learning model.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer function.
        num_epochs: Number of training epochs.
        device: Device ('cuda' or 'cpu') to train on.

    Returns:
        Trained model.
    """
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            frames = batch['frames'].to(device)  # (Batch, Frames, C, H, W)
            label = batch['label'].to(device)  # (Batch,)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device)
                label = batch['label'].to(device)

                outputs = model(frames)
                loss = criterion(outputs, label)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return model