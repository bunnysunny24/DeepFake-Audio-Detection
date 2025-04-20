from dataset_loader import get_data_loaders
from model_training import DeepfakeDetectionModel, train_model
import torch

if __name__ == "__main__":
    # Paths
    json_path = r"D:\Bunny\Deepfake\backend\combined_data\processed_data.json"
    data_dir = r"D:\Bunny\Deepfake\backend\combined_data"

    # Data Loaders
    train_loader, val_loader = get_data_loaders(json_path, data_dir, batch_size=8)

    # Model
    model = DeepfakeDetectionModel()

    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)