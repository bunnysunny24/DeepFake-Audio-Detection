import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset_loader import DeepfakeDataset
from model import MultiModalDeepfakeDetector

# Paths
processed_data_path = r"D:\Bunny\Deepfake\backend\combined_data\processed_data.json"
model_path = r"D:\Bunny\Deepfake\backend\model\deepfake_detector.pth"

# Load Dataset
dataset = DeepfakeDataset(processed_data_path)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalDeepfakeDetector().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluate
preds, labels = [], []

with torch.no_grad():
    for batch in dataloader:
        video_frames = batch["video_frames"].to(device)
        audio_features = batch["audio_features"].to(device)
        labels.extend(batch["labels"].cpu().numpy())
        
        outputs = model(video_frames, audio_features)
        preds.extend((outputs.cpu().numpy() > 0.5).astype(int))

accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
f1 = f1_score(labels, preds)

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")