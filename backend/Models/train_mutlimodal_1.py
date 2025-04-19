from multi_modal_model import MultiModalDeepfakeModel
from dataset_loader import get_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import warnings
import os
import numpy as np
from tqdm import tqdm


def suppress_warnings():
    """
    Suppress all warnings and unnecessary logs.
    """
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def plot_metrics(train_values, val_values, metric_name, epoch, save_dir="plots"):
    """
    Plot training and validation metrics on the same graph and save to file.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_values) + 1), train_values, label=f"Train {metric_name}", color="blue", marker="o")
    plt.plot(range(1, len(val_values) + 1), val_values, label=f"Validation {metric_name}", color="red", marker="o")
    plt.title(f"{metric_name.capitalize()} for Epoch {epoch}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{metric_name}_epoch_{epoch}.png")
    plt.close()


def calculate_metrics(y_true, y_pred, y_probs, epoch):
    """
    Calculate metrics like Precision, Recall, F1 Score, Confusion Matrix, and AUC.
    """
    # Convert lists to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    confusion = confusion_matrix(y_true, y_pred)
    
    # Handle case where we might have only one class in the batch
    try:
        auc_score = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc_score = 0.0

    print(f"Epoch {epoch} Metrics:")
    print(f"- Precision: {precision:.2f}")
    print(f"- Recall   : {recall:.2f}")
    print(f"- F1 Score : {f1:.2f}")
    print(f"- AUC Score: {auc_score:.2f}")
    print(f"- Confusion Matrix:\n{confusion}")

    return precision, recall, f1, auc_score


def move_batch_to_device(batch, device):
    """
    Safely move batch items to device, handling non-tensor items properly.
    """
    device_batch = {}
    for key, value in batch.items():
        if key == "label":
            # Handle labels separately
            continue
        
        if isinstance(value, torch.Tensor):
            device_batch[key] = value.to(device)
        elif value is None:
            device_batch[key] = None
        elif isinstance(value, list):
            # Keep lists as they are
            device_batch[key] = value
        else:
            # Try to convert other types to tensor if possible
            try:
                device_batch[key] = torch.tensor(value).to(device)
            except:
                device_batch[key] = value
    
    return device_batch


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    """
    Train and validate the multimodal deepfake detection model with updated metric plotting.
    """
    model.to(device)
    best_val_accuracy = 0.0
    best_model_path = "best_model_v3.pth"
    model_checkpoint_dir = "checkpoints"
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    # Initialize lists to store metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")

        # Training phase
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        y_train_true, y_train_pred, y_train_probs = [], [], []

        train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=True)
        for batch in train_progress:
            try:
                # Move batch to device safely
                inputs = move_batch_to_device(batch, device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

                # Update progress bar
                train_progress.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{train_correct/total:.4f}"
                })

                # Collect predictions for metrics
                y_train_true.extend(labels.cpu().numpy())
                y_train_pred.extend(outputs.argmax(1).cpu().numpy())
                y_train_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())

            except Exception as e:
                print(f"❌ Error during training: {e}. Skipping batch.")
                continue

        train_accuracy = train_correct / total if total > 0 else 0
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Calculate training metrics
        train_precision, train_recall, train_f1, train_auc = calculate_metrics(
            y_train_true, y_train_pred, y_train_probs, epoch
        )
        train_f1_scores.append(train_f1)
        
        print(f"📊 Training Summary - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")

        # Validation phase
        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        y_val_true, y_val_pred, y_val_probs = [], [], []
        
        val_progress = tqdm(val_loader, desc=f"Validating Epoch {epoch}", leave=True)
        with torch.no_grad():
            for batch in val_progress:
                try:
                    # Move batch to device safely
                    inputs = move_batch_to_device(batch, device)
                    labels = batch["label"].to(device)

                    outputs, deepfake_check = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)
                    
                    # Update progress bar
                    val_progress.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{val_correct/total:.4f}"
                    })

                    # Collect predictions for metrics
                    y_val_true.extend(labels.cpu().numpy())
                    y_val_pred.extend(outputs.argmax(1).cpu().numpy())
                    y_val_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())

                except Exception as e:
                    print(f"❌ Error during validation: {e}. Skipping batch.")
                    continue

        val_accuracy = val_correct / total if total > 0 else 0
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Calculate validation metrics
        val_precision, val_recall, val_f1, val_auc = calculate_metrics(
            y_val_true, y_val_pred, y_val_probs, epoch
        )
        val_f1_scores.append(val_f1)
        
        print(f"📊 Validation Summary - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

        # Plot metrics for this epoch
        plot_metrics(train_losses, val_losses, "loss", epoch)
        plot_metrics(train_accuracies, val_accuracies, "accuracy", epoch)
        plot_metrics(train_f1_scores, val_f1_scores, "f1_score", epoch)

        # Adjust learning rate
        scheduler.step(val_loss)

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, best_model_path))
            print(f"✅ Best model saved with accuracy {best_val_accuracy:.4f}")
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
        }, os.path.join(model_checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))

    return best_val_accuracy


if __name__ == "__main__":
    suppress_warnings()

    # Paths - Update these to your actual paths
    json_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"
    data_dir = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Parameters
    batch_size = 8
    max_samples = 1000  # Limit for testing, set to None to use all samples
    num_workers = 0 if device.type == 'cuda' else 0  # Use 0 for debugging
    num_epochs = 10
    learning_rate = 1e-4
    
    print(f"🔄 Loading data loaders (max_samples={max_samples}, batch_size={batch_size})...")
    
    # Load data
    train_loader, val_loader = get_data_loaders(
        json_path=json_path,
        data_dir=data_dir,
        batch_size=batch_size,
        max_samples=max_samples,
        num_workers=num_workers
    )

    print(f"📦 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model
    print("🔧 Initializing model...")
    model = MultiModalDeepfakeModel(
        num_classes=2,
        video_feature_dim=512,  # Reduced dimension for faster training
        audio_feature_dim=512,  # Reduced dimension for faster training
        transformer_dim=512,    # Reduced dimension for faster training
        num_transformer_layers=2,  # Fewer layers for faster training
        enable_face_mesh=False,  # Disable for faster training
        debug=False  # Disable debugging output
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # Train and validate
    print("🚀 Starting training...")
    best_val_accuracy = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs
    )

    print("\n🎯 Training Complete")
    print(f"🥇 Best Validation Accuracy: {best_val_accuracy:.4f}")
