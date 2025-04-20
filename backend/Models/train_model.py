import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import DeepfakeDataset
from model import MultiModalDeepfakeDetector

# Paths
processed_data_path = r"D:\Bunny\Deepfake\backend\combined_data\processed_data.json"
model_save_path = r"D:\Bunny\Deepfake\backend\model\deepfake_detector.pth"

# Hyperparameters
batch_size = 16
learning_rate = 1e-4
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and Dataloader
dataset = DeepfakeDataset(processed_data_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = MultiModalDeepfakeDetector().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for batch in dataloader:
        video_frames = batch["video_frames"].to(device)  # Video input
        audio_features = batch["audio_features"].to(device)  # Audio input
        labels = batch["labels"].to(device)  # Labels
        
        optimizer.zero_grad()
        outputs = model(video_frames, audio_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")