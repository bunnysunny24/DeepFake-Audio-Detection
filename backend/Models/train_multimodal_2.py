from multi_modal_model import MultiModalDeepfakeModel
from dataset_loader import get_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import warnings
import os
from tqdm import tqdm


def suppress_warnings():
    """
    Suppress all warnings and unnecessary logs.
    """
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def plot_roc_curve(y_true, y_probs, epoch):
    """
    Plot the ROC curve and calculate the AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for Epoch {epoch}")
    plt.legend(loc="lower right")
    plt.show()
    return auc_score


def calculate_metrics(y_true, y_pred, y_probs, epoch):
    """
    Calculate metrics like Precision, Recall, F1 Score, Confusion Matrix, and AUC.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    confusion = confusion_matrix(y_true, y_pred)
    auc_score = plot_roc_curve(y_true, y_probs, epoch)

    print(f"Epoch {epoch} Metrics:")
    print(f"- Precision: {precision:.2f}")
    print(f"- Recall   : {recall:.2f}")
    print(f"- F1 Score : {f1:.2f}")
    print(f"- AUC Score: {auc_score:.2f}")
    print(f"- Confusion Matrix:\n{confusion}")

    return precision, recall, f1, auc_score


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    """
    Train and validate the multimodal deepfake detection model.
    """
    model.to(device)
    best_val_accuracy = 0.0
    best_model_path = "best_model_2.pth"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Training phase
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        y_train_true, y_train_pred, y_train_probs = [], [], []

        for batch in tqdm(train_loader, desc="Training", leave=False):
            video_frames = batch["video_frames"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs, _ = model(video_frames, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            # Collect predictions and true labels for metrics
            y_train_true.extend(labels.cpu().numpy())
            y_train_pred.extend(outputs.argmax(1).cpu().numpy())
            y_train_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())  # Fixed here

        train_accuracy = train_correct / total
        train_loss /= len(train_loader)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}")

        # Calculate training metrics
        calculate_metrics(y_train_true, y_train_pred, y_train_probs, epoch)

        # Validation phase
        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        y_val_true, y_val_pred, y_val_probs = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                video_frames = batch["video_frames"].to(device)
                audio = batch["audio"].to(device)
                labels = batch["label"].to(device)

                outputs, _ = model(video_frames, audio)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

                # Collect predictions and true labels for metrics
                y_val_true.extend(labels.cpu().numpy())
                y_val_pred.extend(outputs.argmax(1).cpu().numpy())
                y_val_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())  # Fixed here

        val_accuracy = val_correct / total
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}")

        # Calculate validation metrics
        calculate_metrics(y_val_true, y_val_pred, y_val_probs, epoch)

        # Adjust learning rate
        scheduler.step(val_loss)

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"✔️ Best model saved with accuracy {best_val_accuracy:.2f}")

    # Print final classification report
    print("\nFinal Classification Report:")
    print(classification_report(y_val_true, y_val_pred))

    return best_val_accuracy


if __name__ == "__main__":
    suppress_warnings()

    # Paths
    json_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"
    data_dir = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = get_data_loaders(json_path, data_dir, batch_size=8)

    # Limit dataset size to the first 1000 samples for both training and validation
    train_subset = Subset(train_loader.dataset, list(range(min(len(train_loader.dataset), 1000))))
    val_subset = Subset(val_loader.dataset, list(range(min(len(val_loader.dataset), 1000))))

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=8, shuffle=False)

    # Initialize model
    model = MultiModalDeepfakeModel(num_classes=2)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Train and validate
    best_val_accuracy = train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10)

    print("\n🎯 Training Complete")
    print(f"🥇 Best Validation Accuracy: {best_val_accuracy:.2f}")
