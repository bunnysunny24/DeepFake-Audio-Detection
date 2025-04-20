from multi_modal_model import MultiModalDeepfakeModel
from dataset_loader import get_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Subset


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    Train and validate the multimodal deepfake detection model.
    """

    # Move model to device
    model.to(device)

    best_val_accuracy = 0.0
    best_model_state = None
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training Loop
        model.train()
        total_loss, total_correct, total_inconsistencies = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
            video_frames = batch["video_frames"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs, deepfake_check = model(video_frames, audio)

            # Calculate loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

            # Count deepfake inconsistencies (safeguard added here)
            if deepfake_check is not None and isinstance(deepfake_check, torch.Tensor):
                total_inconsistencies += deepfake_check.sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}, Deepfake Inconsistencies: {total_inconsistencies}"
        )

        # Validation Loop
        model.eval()
        total_val_loss, total_val_correct, total_val_inconsistencies = 0, 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}", leave=False):
                video_frames = batch["video_frames"].to(device)
                audio = batch["audio"].to(device)
                labels = batch["label"].to(device)

                outputs, deepfake_check = model(video_frames, audio)
                val_loss = criterion(outputs, labels)

                total_val_loss += val_loss.item()
                total_val_correct += (outputs.argmax(1) == labels).sum().item()

                # Count deepfake inconsistencies (already safeguarded)
                if deepfake_check is not None and isinstance(deepfake_check, torch.Tensor):
                    total_val_inconsistencies += deepfake_check.sum().item()

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = total_val_correct / len(val_loader.dataset)

        print(
            f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}, Deepfake Inconsistencies: {total_val_inconsistencies}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, "best_model.pth")

        # Check for overfitting
        if val_loss > best_val_loss:
            print(
                "Warning: Potential Overfitting Detected - Validation loss is increasing while training loss is decreasing."
            )

        best_val_loss = min(best_val_loss, val_loss)

    return best_val_accuracy



if __name__ == "__main__":
    # Paths
    json_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"
    data_dir = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loaders
    train_loader, val_loader = get_data_loaders(json_path, data_dir, batch_size=8)

    # Limit to the first 1000 samples
    train_subset = Subset(train_loader.dataset, list(range(min(len(train_loader.dataset), 1000))))
    val_subset = Subset(val_loader.dataset, list(range(min(len(val_loader.dataset), 1000))))

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=8, shuffle=False)

    # Model
    model = MultiModalDeepfakeModel(num_classes=2)  # Explicitly set num_classes

    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train and validate
    best_val_accuracy = train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)

    print("Training Complete.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}")