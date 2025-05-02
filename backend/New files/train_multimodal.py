from multi_modal_model import MultiModalDeepfakeModel
from dataset_loader import get_data_loaders, get_transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import numpy as np
import json
import time
from tqdm import tqdm
import cv2
from pathlib import Path
import pandas as pd
import wandb
import argparse
from datetime import datetime
import shutil


def suppress_warnings():
    """Suppress all warnings and unnecessary logs."""
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    """Get the current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def visualize_attention_maps(frames, attention_maps, save_dir, sample_idx, epoch):
    """Visualize attention maps overlaid on input frames."""
    os.makedirs(save_dir, exist_ok=True)
    
    frames = frames.cpu().numpy()
    attention_maps = attention_maps.cpu().numpy()
    
    num_frames = min(5, frames.shape[0])
    
    for t in range(num_frames):
        frame = frames[t].transpose(1, 2, 0)
        frame = (frame * 255).astype(np.uint8)
        
        # Get attention map for the fake class (index 1)
        attention = attention_maps[t, 1]
        
        # Normalize attention map to 0-255
        attention = (attention * 255).astype(np.uint8)
        
        # Apply colormap for visualization
        heatmap = cv2.applyColorMap(attention, cv2.COLORMAP_JET)
        
        # Overlay attention map on frame
        alpha = 0.4
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        # Save visualization
        save_path = os.path.join(save_dir, f'sample_{sample_idx}_frame_{t}_epoch_{epoch}.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def plot_metrics(train_values, val_values, metric_name, epoch, save_dir="plots"):
    """Plot training and validation metrics on the same graph and save to file."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_values) + 1), train_values, label=f"Train {metric_name}", color="blue", marker="o")
    plt.plot(range(1, len(val_values) + 1), val_values, label=f"Validation {metric_name}", color="red", marker="o")
    plt.title(f"{metric_name.capitalize()} for Epoch {epoch}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(save_dir, f"{metric_name}_epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def plot_confusion_matrix(y_true, y_pred, epoch, save_dir="plots"):
    """Plot confusion matrix and save to file."""
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0.5, 1.5], ["Real", "Fake"])
    plt.yticks([0.5, 1.5], ["Real", "Fake"])
    
    # Save plot
    cm_path = os.path.join(save_dir, f"confusion_matrix_epoch_{epoch}.png")
    plt.savefig(cm_path)
    plt.close()
    
    return cm_path


def calculate_metrics(y_true, y_pred, y_probs, epoch, return_dict=False):
    """Calculate metrics like Precision, Recall, F1 Score, Confusion Matrix, and AUC."""
    # Convert lists to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    confusion = confusion_matrix(y_true, y_pred)
    
    # Handle case where we might have only one class in the batch
    try:
        auc_score = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc_score = 0.0

    print(f"Epoch {epoch} Metrics:")
    print(f"- Accuracy : {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall   : {recall:.4f}")
    print(f"- F1 Score : {f1:.4f}")
    print(f"- AUC Score: {auc_score:.4f}")
    print(f"- Confusion Matrix:\n{confusion}")
    
    if return_dict:
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'confusion_matrix': confusion.tolist()
        }
    else:
        return precision, recall, f1, auc_score, accuracy


def save_visualizations(inputs, outputs, results, epoch, sample_idx, viz_dir):
    """Save visualizations of model predictions and attention maps."""
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        # Get input video frames
        video_frames = inputs['video_frames']
        batch_size, num_frames = video_frames.shape[:2]
        
        # Limit to a small number of frames for visualization
        max_viz_frames = min(5, num_frames)
        
        # Extract a sample video
        sample_frames = video_frames[sample_idx, :max_viz_frames].cpu()
        
        # Save original frames
        frames_dir = os.path.join(viz_dir, f'sample_{sample_idx}_epoch_{epoch}')
        os.makedirs(frames_dir, exist_ok=True)
        
        for t in range(max_viz_frames):
            frame = sample_frames[t].permute(1, 2, 0).numpy()
            frame = (frame * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(frames_dir, f'frame_{t}.jpg'),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
        
        # Save model explanation if available
        if 'explanation' in results and results['explanation'] is not None:
            explanation = results['explanation']
            
            # Save highlighted regions if available
            if 'highlighted_regions' in explanation and explanation['highlighted_regions']:
                regions = explanation['highlighted_regions']
                
                for region in regions:
                    if len(region) >= 3:
                        batch_idx, frame_idx, score = region
                        if batch_idx == sample_idx and frame_idx < max_viz_frames:
                            frame = sample_frames[frame_idx].permute(1, 2, 0).numpy()
                            frame = (frame * 255).astype(np.uint8)
                            
                            # Add highlight overlay for suspicious regions
                            highlight = np.zeros_like(frame)
                            highlight[:, :, 0] = 255  # Red channel
                            
                            # Apply highlight with transparency
                            alpha = min(0.7, score * 5)  # Scale by score
                            highlighted_frame = cv2.addWeighted(
                                frame, 1 - alpha, highlight, alpha, 0
                            )
                            
                            cv2.imwrite(
                                os.path.join(frames_dir, f'frame_{frame_idx}_highlighted.jpg'),
                                cv2.cvtColor(highlighted_frame, cv2.COLOR_RGB2BGR)
                            )
            
            # Save explanation to text file
            with open(os.path.join(frames_dir, 'explanation.txt'), 'w') as f:
                f.write(f"Prediction: {'Fake' if outputs.argmax(1)[sample_idx].item() == 1 else 'Real'}\n")
                f.write(f"Confidence: {torch.softmax(outputs, dim=1)[sample_idx][outputs.argmax(1)[sample_idx]].item():.4f}\n\n")
                
                if 'issues_found' in explanation:
                    f.write("Issues found:\n")
                    for issue in explanation['issues_found']:
                        f.write(f"- {issue}\n")
                
                if 'detection_scores' in explanation:
                    f.write("\nDetection scores:\n")
                    for key, value in explanation['detection_scores'].items():
                        f.write(f"- {key}: {value:.4f}\n")
                
                if 'confidence' in explanation:
                    f.write(f"\nOverall confidence: {explanation['confidence']:.4f}\n")
        
        # Generate and save attention maps if model has this capability
        if hasattr(model, 'get_attention_maps'):
            try:
                with torch.no_grad():
                    attention_maps = model.get_attention_maps(inputs)
                    if attention_maps is not None:
                        visualize_attention_maps(
                            sample_frames, 
                            attention_maps[sample_idx], 
                            frames_dir, 
                            sample_idx, 
                            epoch
                        )
            except Exception as e:
                print(f"Error generating attention maps: {e}")
                
    except Exception as e:
        print(f"Error saving visualizations: {e}")


def move_batch_to_device(batch, device):
    """Safely move batch items to device, handling non-tensor items properly."""
    device_batch = {}
    for key, value in batch.items():
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


class DeepfakeTrainer:
    """Class to manage the training, validation, and testing of the multimodal deepfake detection model."""
    
    def __init__(self, config):
        """Initialize the trainer with the given configuration."""
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.amp_enabled = config.amp_enabled and self.device.type == 'cuda'
        self.distributed = config.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1
        self.local_rank = config.local_rank
        self.is_main_process = not self.distributed or self.local_rank == 0
        
        # Set up directories
        self.setup_directories()
        
        # Initialize wandb if enabled
        if config.use_wandb and self.is_main_process:
            self.setup_wandb()
        else:
            self.config.use_wandb = False
        
        # Save configuration
        self.save_config()
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Initialize data loaders
        self.setup_data()
        
        # Initialize model, optimizer, and scheduler
        self.setup_model()
        
        # Initialize metrics tracking
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'train_f1_scores': [],
            'val_f1_scores': [],
            'train_auc_scores': [],
            'val_auc_scores': []
        }
        
        # Initialize best model tracking
        self.best_val_accuracy = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0
        
        # Initialize scaler for mixed precision
        self.scaler = GradScaler() if self.amp_enabled else None
        
        # Create a specific run folder in the checkpoint directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_checkpoint_dir, exist_ok=True)
        
    def setup_directories(self):
        """Set up directories for saving models, logs, and visualizations."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.config.output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create checkpoint directory (using the specified path)
        self.model_dir = self.config.checkpoint_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create subdirectories
        self.log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.viz_dir = os.path.join(self.run_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        
        print(f"Run directory created at: {self.run_dir}")
        print(f"Checkpoint directory created at: {self.model_dir}")
    
    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        try:
            project_name = self.config.wandb_project or "deepfake-detection"
            run_name = self.config.wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=project_name,
                name=run_name,
                config=vars(self.config)
            )
            
            print(f"Weights & Biases initialized: {project_name}/{run_name}")
        except Exception as e:
            print(f"Error initializing Weights & Biases: {e}")
            self.config.use_wandb = False
    
    def save_config(self):
        """Save configuration to JSON file."""
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            # Convert config to dict and handle non-serializable values
            config_dict = vars(self.config).copy()
            for key, value in config_dict.items():
                if not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    config_dict[key] = str(value)
            
            json.dump(config_dict, f, indent=4)
        
        print(f"Configuration saved to: {config_path}")
    
    def setup_data(self):
        """Initialize data loaders for training, validation, and testing."""
        print("Setting up data loaders...")
        
        if self.distributed:
            dist.init_process_group(backend='nccl', init_method='env://')
            torch.cuda.set_device(self.local_rank)
        
        # Get training and validation transforms
        train_video_transform, train_audio_transform = get_transforms('train')
        val_video_transform, val_audio_transform = get_transforms('val')
        
        # Load data loaders
        self.train_loader, self.val_loader, self.test_loader, self.class_weights = get_data_loaders(
            json_path=self.config.json_path,
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            test_split=self.config.test_split,
            shuffle=True,
            num_workers=self.config.num_workers,
            max_samples=self.config.max_samples,
            detect_faces=self.config.detect_faces,
            compute_spectrograms=self.config.compute_spectrograms,
            temporal_features=self.config.temporal_features
        )
        
        print(f"Data loaders created: {len(self.train_loader)} train batches, "
              f"{len(self.val_loader)} validation batches, {len(self.test_loader)} test batches")
    
    def setup_model(self):
        """Initialize model, loss function, optimizer, and scheduler."""
        print(f"Initializing model on {self.device}...")
        
        # Move class weights to device
        self.class_weights = self.class_weights.to(self.device) if self.class_weights is not None else None
        
        # Initialize model
        self.model = MultiModalDeepfakeModel(
            num_classes=self.config.num_classes,
            video_feature_dim=self.config.video_feature_dim,
            audio_feature_dim=self.config.audio_feature_dim,
            transformer_dim=self.config.transformer_dim,
            num_transformer_layers=self.config.num_transformer_layers,
            enable_face_mesh=self.config.enable_face_mesh,
            enable_explainability=self.config.enable_explainability,
            fusion_type=self.config.fusion_type,
            backbone_visual=self.config.backbone_visual,
            backbone_audio=self.config.backbone_audio,
            use_spectrogram=self.config.use_spectrogram,
            detect_deepfake_type=self.config.detect_deepfake_type,
            num_deepfake_types=self.config.num_deepfake_types,
            debug=self.config.debug
        )
        
        # Load pretrained weights if specified
        if self.config.pretrained_path is not None and os.path.exists(self.config.pretrained_path):
            print(f"Loading pretrained weights from: {self.config.pretrained_path}")
            state_dict = torch.load(self.config.pretrained_path, map_location=self.device)
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Handle partial loading with mismatched keys
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained weights")
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize in distributed mode if specified
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Initialize loss function with class weights for imbalanced data
        if self.class_weights is not None and self.config.use_weighted_loss:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            print(f"Using weighted CrossEntropyLoss with weights: {self.class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")
        
        # Initialize optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=self.config.weight_decay)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        # Initialize learning rate scheduler
        if self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.scheduler_step_size, gamma=self.config.scheduler_gamma
            )
        elif self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=self.config.scheduler_gamma, patience=self.config.scheduler_patience, verbose=True
            )
        else:
            self.scheduler = None
        
        # Initialize learning rate warmup
        self.warmup_scheduler = None
        if self.config.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.config.warmup_epochs * len(self.train_loader)
            )
        
        print("Model, optimizer, and scheduler initialized")
    
    def save_intermediate_checkpoint(self, epoch, batch_idx):
        """Save intermediate checkpoint during training."""
        if not self.config.save_intermediate:
            return
            
        if batch_idx % self.config.save_intermediate_interval != 0:
            return
            
        intermediate_dir = os.path.join(self.run_checkpoint_dir, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            intermediate_dir, 
            f"checkpoint_epoch_{epoch+1}_batch_{batch_idx}.pth"
        )
        
        model_state_dict = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch + 1,
            'batch': batch_idx,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Intermediate checkpoint saved: {checkpoint_path}")
    
    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        epoch_loss = 0
        y_true, y_pred, y_probs = [], [], []
        
        train_progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]", 
                           disable=not self.is_main_process)
        
        for batch_idx, batch in enumerate(train_progress):
            try:
                # Move batch to device
                batch = move_batch_to_device(batch, self.device)
                
                # Get labels
                labels = batch['label']
                
                # Forward pass with mixed precision
                self.optimizer.zero_grad()
                
                if self.amp_enabled:
                    with autocast():
                        outputs, results = self.model(batch)
                        loss = self.criterion(outputs, labels)
                        
                        # Add regularization for deepfake type if enabled
                        if self.config.detect_deepfake_type and 'deepfake_type' in results and results['deepfake_type'] is not None:
                            if 'deepfake_type' in batch and batch['deepfake_type'] is not None:
                                deepfake_type_loss = nn.CrossEntropyLoss()(results['deepfake_type'], batch['deepfake_type'])
                                loss += self.config.deepfake_type_weight * deepfake_type_loss
                    
                    # Backward pass with scaler
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs, results = self.model(batch)
                    loss = self.criterion(outputs, labels)
                    
                    # Add regularization for deepfake type if enabled
                    if self.config.detect_deepfake_type and 'deepfake_type' in results and results['deepfake_type'] is not None:
                        if 'deepfake_type' in batch and batch['deepfake_type'] is not None:
                            deepfake_type_loss = nn.CrossEntropyLoss()(results['deepfake_type'], batch['deepfake_type'])
                            loss += self.config.deepfake_type_weight * deepfake_type_loss
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    
                    self.optimizer.step()
                
                # Update learning rate with warmup if enabled
                if self.warmup_scheduler is not None and epoch < self.config.warmup_epochs:
                    self.warmup_scheduler.step()
                
                # Update progress bar
                train_progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{get_lr(self.optimizer):.6f}")
                
                # Accumulate loss
                epoch_loss += loss.item()
                
                # Accumulate predictions and labels for metrics calculation
                y_true.extend(labels.cpu().numpy())
                predictions = outputs.argmax(1).cpu().numpy()
                y_pred.extend(predictions)
                y_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
                
                # Log batch results to WandB
                if self.config.use_wandb and self.is_main_process and batch_idx % self.config.log_interval == 0:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'learning_rate': get_lr(self.optimizer),
                        'batch': batch_idx + epoch * len(self.train_loader)
                    })
                    
                # Visualize sample predictions periodically
                if (batch_idx + 1) % self.config.visualization_interval == 0 and self.is_main_process:
                    try:
                        sample_idx = np.random.randint(min(4, outputs.size(0)))
                        save_visualizations(
                            batch, outputs, results, 
                            epoch + 1, sample_idx, 
                            os.path.join(self.viz_dir, f"train_epoch_{epoch+1}")
                        )
                    except Exception as vis_error:
                        print(f"Error visualizing predictions: {vis_error}")
                
                # Save intermediate checkpoint if enabled
                if self.is_main_process:
                    self.save_intermediate_checkpoint(epoch, batch_idx)
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate average loss and metrics
        avg_loss = epoch_loss / len(self.train_loader)
        precision, recall, f1, auc_score, accuracy = calculate_metrics(y_true, y_pred, y_probs, epoch+1)
        
        # Store metrics
        self.metrics['train_losses'].append(avg_loss)
        self.metrics['train_accuracies'].append(accuracy)
        self.metrics['train_f1_scores'].append(f1)
        self.metrics['train_auc_scores'].append(auc_score)
        
        # Return metrics
        return avg_loss, accuracy, precision, recall, f1, auc_score
    
    def validate_epoch(self, epoch):
        """Validate the model for one epoch."""
        self.model.eval()
        epoch_loss = 0
        y_true, y_pred, y_probs = [], [], []
        
        val_progress = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Val]", 
                          disable=not self.is_main_process)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_progress):
                try:
                    # Move batch to device
                    batch = move_batch_to_device(batch, self.device)
                    
                    # Get labels
                    labels = batch['label']
                    
                    # Forward pass
                    if self.amp_enabled:
                        with autocast():
                            outputs, results = self.model(batch)
                            loss = self.criterion(outputs, labels)
                    else:
                        outputs, results = self.model(batch)
                        loss = self.criterion(outputs, labels)
                    
                    # Update progress bar
                    val_progress.set_postfix(loss=f"{loss.item():.4f}")
                    
                    # Accumulate loss
                    epoch_loss += loss.item()
                    
                    # Accumulate predictions and labels for metrics calculation
                    y_true.extend(labels.cpu().numpy())
                    predictions = outputs.argmax(1).cpu().numpy()
                    y_pred.extend(predictions)
                    y_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
                    
                    # Visualize sample predictions periodically
                    if (batch_idx + 1) % self.config.visualization_interval == 0 and self.is_main_process:
                        try:
                            sample_idx = np.random.randint(min(4, outputs.size(0)))
                            save_visualizations(
                                batch, outputs, results, 
                                epoch + 1, sample_idx, 
                                os.path.join(self.viz_dir, f"val_epoch_{epoch+1}")
                            )
                        except Exception as vis_error:
                            print(f"Error visualizing predictions: {vis_error}")
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Calculate average loss and metrics
        avg_loss = epoch_loss / len(self.val_loader)
        precision, recall, f1, auc_score, accuracy = calculate_metrics(y_true, y_pred, y_probs, epoch+1)
        
        # Store metrics
        self.metrics['val_losses'].append(avg_loss)
        self.metrics['val_accuracies'].append(accuracy)
        self.metrics['val_f1_scores'].append(f1)
        self.metrics['val_auc_scores'].append(auc_score)
        
        # Plot confusion matrix
        if self.is_main_process:
            cm_path = plot_confusion_matrix(y_true, y_pred, epoch+1, self.plot_dir)
            
            # Log confusion matrix to WandB
            if self.config.use_wandb:
                wandb.log({f"confusion_matrix_epoch_{epoch+1}": wandb.Image(cm_path)})
        
        # Return metrics
        return avg_loss, accuracy, precision, recall, f1, auc_score
    
    def test_model(self):
        """Test the trained model on the test set."""
        print("\nEvaluating model on test set...")
        self.model.eval()
        test_loss = 0
        y_true, y_pred, y_probs = [], [], []
        
        test_progress = tqdm(self.test_loader, desc="Testing")
        
        results_data = {
            'file_path': [],
            'true_label': [],
            'pred_label': [],
            'confidence': [],
            'issues_found': [],
            'confidence_score': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_progress):
                try:
                    # Move batch to device
                    batch = move_batch_to_device(batch, self.device)
                    
                    # Get labels and file paths
                    labels = batch['label']
                    file_paths = batch.get('file_path', ['unknown'] * len(labels))
                    
                    # Forward pass
                    if self.amp_enabled:
                        with autocast():
                            outputs, results = self.model(batch)
                            loss = self.criterion(outputs, labels)
                    else:
                        outputs, results = self.model(batch)
                        loss = self.criterion(outputs, labels)
                    
                    # Update progress bar
                    test_progress.set_postfix(loss=f"{loss.item():.4f}")
                    
                    # Accumulate loss
                    test_loss += loss.item()
                    
                    # Accumulate predictions and labels for metrics calculation
                    batch_y_true = labels.cpu().numpy()
                    y_true.extend(batch_y_true)
                    
                    predictions = outputs.argmax(1).cpu().numpy()
                    y_pred.extend(predictions)
                    
                    confidences = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                    y_probs.extend(confidences)
                    
                    # Collect detailed results for analysis
                    for i in range(len(batch_y_true)):
                        issues = []
                        confidence_score = 0.5  # Default confidence
                        
                        # Extract explanation if available
                        if 'explanation' in results and results['explanation'] is not None:
                            explanation = results['explanation']
                            if 'issues_found' in explanation and isinstance(explanation['issues_found'], list):
                                issues = explanation['issues_found'][:3]  # Take first 3 issues
                            if 'confidence' in explanation:
                                confidence_score = explanation['confidence']
                        
                        results_data['file_path'].append(file_paths[i] if i < len(file_paths) else 'unknown')
                        results_data['true_label'].append(int(batch_y_true[i]))
                        results_data['pred_label'].append(int(predictions[i]))
                        results_data['confidence'].append(float(confidences[i]))
                        results_data['issues_found'].append('; '.join(issues))
                        results_data['confidence_score'].append(float(confidence_score))
                    
                    # Visualize sample predictions periodically
                    if (batch_idx + 1) % self.config.visualization_interval == 0 and self.is_main_process:
                        try:
                            sample_idx = np.random.randint(min(4, outputs.size(0)))
                            save_visualizations(
                                batch, outputs, results, 
                                0, sample_idx, 
                                os.path.join(self.viz_dir, "test_results")
                            )
                        except Exception as vis_error:
                            print(f"Error visualizing predictions: {vis_error}")
                    
                except Exception as e:
                    print(f"Error in test batch {batch_idx}: {e}")
                    continue
        
        # Calculate average loss and metrics
        avg_loss = test_loss / len(self.test_loader)
        metrics_dict = calculate_metrics(y_true, y_pred, y_probs, 0, return_dict=True)
        
        print("\nTest Results:")
        print(f"- Loss     : {avg_loss:.4f}")
        print(f"- Accuracy : {metrics_dict['accuracy']:.4f}")
        print(f"- Precision: {metrics_dict['precision']:.4f}")
        print(f"- Recall   : {metrics_dict['recall']:.4f}")
        print(f"- F1 Score : {metrics_dict['f1']:.4f}")
        print(f"- AUC Score: {metrics_dict['auc']:.4f}")
        
        # Plot confusion matrix for test set
        if self.is_main_process:
            cm_path = plot_confusion_matrix(y_true, y_pred, 0, self.plot_dir)
            
            # Log confusion matrix to WandB
            if self.config.use_wandb:
                wandb.log({
                    "test_confusion_matrix": wandb.Image(cm_path),
                    "test_accuracy": metrics_dict['accuracy'],
                    "test_precision": metrics_dict['precision'],
                    "test_recall": metrics_dict['recall'],
                    "test_f1": metrics_dict['f1'],
                    "test_auc": metrics_dict['auc']
                })
        
        # Save detailed results to CSV
        if self.is_main_process:
            results_df = pd.DataFrame(results_data)
            results_path = os.path.join(self.log_dir, "test_results.csv")
            results_df.to_csv(results_path, index=False)
            print(f"Test results saved to: {results_path}")
        
        return avg_loss, metrics_dict
    
    def train_and_validate(self):
        """Train and validate the model for the specified number of epochs."""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = self.validate_epoch(epoch)
            
            # Update learning rate scheduler if using plateau scheduler
            if self.scheduler is not None:
                if self.config.scheduler == 'plateau':
                    # ReduceLROnPlateau step with validation metrics
                    self.scheduler.step(val_f1)  # or val_acc
                else:
                    self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            if self.is_main_process:
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} completed in {epoch_time:.2f}s")
                print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_acc, val_f1)
                
                # Plot metrics
                for metric_name, train_values, val_values in [
                    ('loss', self.metrics['train_losses'], self.metrics['val_losses']),
                    ('accuracy', self.metrics['train_accuracies'], self.metrics['val_accuracies']),
                    ('f1', self.metrics['train_f1_scores'], self.metrics['val_f1_scores']),
                    ('auc', self.metrics['train_auc_scores'], self.metrics['val_auc_scores'])
                ]:
                    plot_path = plot_metrics(train_values, val_values, metric_name, epoch+1, self.plot_dir)
                    
                    # Log metrics plot to WandB
                    if self.config.use_wandb:
                        wandb.log({f"{metric_name}_plot": wandb.Image(plot_path)})
                
                # Log epoch metrics to WandB
                if self.config.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc,
                        'train_precision': train_precision,
                        'val_precision': val_precision,
                        'train_recall': train_recall,
                        'val_recall': val_recall,
                        'train_f1': train_f1,
                        'val_f1': val_f1,
                        'train_auc': train_auc,
                        'val_auc': val_auc,
                        'learning_rate': get_lr(self.optimizer),
                        'epoch_time': epoch_time
                    })
            
            # Check for early stopping
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.early_stop_counter = 0
            elif val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                
            if self.early_stop_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                break
        
        # Load best model for testing
        self.load_best_model()
        
        # Test the model
        test_loss, test_metrics = self.test_model()
        
        # Save final metrics
        self.save_final_metrics(test_loss, test_metrics)
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch+1}")
        return test_metrics
    
    def save_checkpoint(self, epoch, val_acc, val_f1):
        """Save model checkpoint."""
        # Save latest model in the run-specific folder
        epoch_checkpoint_path = os.path.join(self.run_checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        
        model_state_dict = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        # Gather more metrics for the checkpoint
        train_metrics = {
            'loss': self.metrics['train_losses'][-1] if self.metrics['train_losses'] else None,
            'accuracy': self.metrics['train_accuracies'][-1] if self.metrics['train_accuracies'] else None,
            'f1': self.metrics['train_f1_scores'][-1] if self.metrics['train_f1_scores'] else None,
            'auc': self.metrics['train_auc_scores'][-1] if self.metrics['train_auc_scores'] else None
        }
        
        val_metrics = {
            'loss': self.metrics['val_losses'][-1] if self.metrics['val_losses'] else None,
            'accuracy': val_acc,
            'f1': val_f1,
            'auc': self.metrics['val_auc_scores'][-1] if self.metrics['val_auc_scores'] else None
        }
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': vars(self.config),
            'timestamp': self.timestamp
        }
        
        # Save the checkpoint for this epoch
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"Checkpoint saved: {epoch_checkpoint_path}")
        
        # Save a copy in the main checkpoint directory as latest checkpoint
        latest_path = os.path.join(self.model_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)
        
        # Save best model
        is_best = False
        if val_acc > self.best_val_accuracy or (val_acc == self.best_val_accuracy and val_f1 > self.best_val_f1):
            is_best = True
            self.best_val_accuracy = val_acc
            self.best_val_f1 = val_f1
            self.best_epoch = epoch
            
            # Save in both the run-specific directory and the main checkpoint directory
            best_path_run = os.path.join(self.run_checkpoint_dir, "best_model_v1.pth")
            torch.save(checkpoint, best_path_run)
            
            # Main checkpoint directory best model
            best_path = os.path.join(self.model_dir, "best_model_v1.pth")
            torch.save(checkpoint, best_path)
            
            # Also save a timestamped version of the best model
            best_path_timestamped = os.path.join(
                self.model_dir, 
                f"best_model_epoch_{epoch+1}_acc_{val_acc:.4f}_f1_{val_f1:.4f}.pth"
            )
            torch.save(checkpoint, best_path_timestamped)
            
            print(f"Best model saved (Epoch {epoch+1}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f})")
            print(f"Path: {best_path}")
            print(f"Timestamped path: {best_path_timestamped}")
        
        # Optionally remove old checkpoints to save space
        if self.config.keep_n_checkpoints > 0:
            # Find all checkpoint files in the run directory
            checkpoint_files = []
            for file in os.listdir(self.run_checkpoint_dir):
                if file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
                    checkpoint_files.append(os.path.join(self.run_checkpoint_dir, file))
            
            # Sort by modification time (oldest first)
            checkpoint_files.sort(key=os.path.getmtime)
            
            # Keep only the most recent N checkpoints
            while len(checkpoint_files) > self.config.keep_n_checkpoints:
                try:
                    os.remove(checkpoint_files[0])
                    print(f"Removed old checkpoint: {checkpoint_files[0]}")
                    checkpoint_files.pop(0)
                except Exception as e:
                    print(f"Error removing old checkpoint: {e}")
                    break
        
        return is_best
    
    def load_best_model(self):
        """Load the best model for testing."""
        best_model_path = os.path.join(self.model_dir, "best_model_v1.pth")
        if not os.path.exists(best_model_path):
            print("Best model not found. Using current model.")
            return
        
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        
        try:
            # Load model state dict
            if self.distributed:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_metrics']['accuracy']:.4f}")
            
            # Optionally load optimizer and scheduler states as well
            if 'optimizer_state_dict' in checkpoint and self.config.resume_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded")
                
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None and self.config.resume_scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler state loaded")
        except Exception as e:
            print(f"Error loading best model: {e}")
    
    def save_final_metrics(self, test_loss, test_metrics):
        """Save final metrics to JSON file."""
        final_metrics = {
            'train_metrics': {
                'loss': self.metrics['train_losses'][-1],
                'accuracy': self.metrics['train_accuracies'][-1],
                'f1': self.metrics['train_f1_scores'][-1],
                'auc': self.metrics['train_auc_scores'][-1]
            },
            'val_metrics': {
                'loss': self.metrics['val_losses'][-1],
                'accuracy': self.metrics['val_accuracies'][-1],
                'f1': self.metrics['val_f1_scores'][-1],
                'auc': self.metrics['val_auc_scores'][-1]
            },
            'best_val_metrics': {
                'accuracy': self.best_val_accuracy,
                'f1': self.best_val_f1,
                'epoch': self.best_epoch + 1
            },
            'test_metrics': test_metrics
        }
        
        # Save in log directory
        metrics_path = os.path.join(self.log_dir, "final_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        # Also save in checkpoint directory
        metrics_path_checkpoint = os.path.join(self.run_checkpoint_dir, "final_metrics.json")
        with open(metrics_path_checkpoint, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        print(f"Final metrics saved to: {metrics_path}")
        print(f"Final metrics also saved to: {metrics_path_checkpoint}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a multimodal deepfake detection model.')
    
    # Data parameters
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON file with dataset metadata.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing video and audio files.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='D:/Bunny/Deepfake/backend/Models/saved_models',
                        help='Path to save model checkpoints.')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use from dataset.')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (2 for binary classification).')
    parser.add_argument('--video_feature_dim', type=int, default=1024, help='Dimension of video features.')
    parser.add_argument('--audio_feature_dim', type=int, default=1024, help='Dimension of audio features.')
    parser.add_argument('--transformer_dim', type=int, default=768, help='Dimension of transformer features.')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer layers.')
    parser.add_argument('--enable_face_mesh', action='store_true', help='Enable face mesh analysis.')
    parser.add_argument('--enable_explainability', action='store_true', help='Enable model explainability.')
    parser.add_argument('--fusion_type', type=str, default='attention', choices=['attention', 'concat'], help='Type of fusion.')
    parser.add_argument('--backbone_visual', type=str, default='efficientnet', choices=['efficientnet', 'swin'], help='Visual backbone model.')
    parser.add_argument('--backbone_audio', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert'], help='Audio backbone model.')
    parser.add_argument('--use_spectrogram', action='store_true', help='Use spectrogram features.')
    parser.add_argument('--detect_deepfake_type', action='store_true', help='Detect deepfake type.')
    parser.add_argument('--num_deepfake_types', type=int, default=7, help='Number of deepfake types.')
    parser.add_argument('--deepfake_type_weight', type=float, default=0.2, help='Weight of deepfake type loss.')
    parser.add_argument('--detect_faces', action='store_true', help='Detect and extract facial features.')
    parser.add_argument('--compute_spectrograms', action='store_true', help='Compute audio spectrograms.')
    parser.add_argument('--temporal_features', action='store_true', help='Compute temporal consistency features.')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay.')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Fraction of data to use for validation.')
    parser.add_argument('--test_split', type=float, default=0.1, help='Fraction of data to use for testing.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer.')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'plateau', 'none'], help='Learning rate scheduler.')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='Step size for StepLR scheduler.')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma for StepLR and ReduceLROnPlateau schedulers.')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for ReduceLROnPlateau scheduler.')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs.')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use weighted loss for imbalanced data.')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value. 0 disables clipping.')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model weights.')
    
    # Checkpoint parameters
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the best model checkpoint.')
    parser.add_argument('--resume_optimizer', action='store_true', help='Resume optimizer state when resuming training.')
    parser.add_argument('--resume_scheduler', action='store_true', help='Resume scheduler state when resuming training.')
    parser.add_argument('--save_every_epoch', action='store_true', help='Save a checkpoint after every epoch.')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate checkpoints during training.')
    parser.add_argument('--save_intermediate_interval', type=int, default=100, help='Interval (in batches) for saving intermediate checkpoints.')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use.')
    parser.add_argument('--amp_enabled', action='store_true', help='Enable automatic mixed precision.')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training.')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading.')
    
    # Logging and visualization parameters
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for experiment tracking.')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name.')
    parser.add_argument('--log_interval', type=int, default=10, help='Number of batches between logging.')
    parser.add_argument('--visualization_interval', type=int, default=50, help='Number of batches between visualizations.')
    parser.add_argument('--keep_n_checkpoints', type=int, default=3, help='Number of checkpoints to keep.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    return parser.parse_args()


def main():
    """Main function to run the training and testing."""
    # Suppress warnings
    suppress_warnings()
    
    # Parse arguments
    args = parse_arguments()
    
    # Make sure the checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Resume training from checkpoint if requested
    if args.resume_training:
        checkpoint_path = os.path.join(args.checkpoint_dir, "best_model_v1.pth")
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint for resuming training: {checkpoint_path}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Training from scratch.")
    
    # Initialize trainer
    trainer = DeepfakeTrainer(args)
    
    # Train and validate the model
    test_metrics = trainer.train_and_validate()
    
    # Print final results
    print("\nFinal Results:")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test AUC Score: {test_metrics['auc']:.4f}")
    
    # Create a zip archive of the best model for easier distribution
    try:
        best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            import zipfile
            zip_path = os.path.join(args.checkpoint_dir, f"best_model_{trainer.timestamp}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(best_model_path, os.path.basename(best_model_path))
                
                # Also include the config file
                config_path = os.path.join(trainer.run_dir, "config.json")
                if os.path.exists(config_path):
                    zipf.write(config_path, os.path.basename(config_path))
                    
            print(f"Zipped best model saved to: {zip_path}")
    except Exception as e:
        print(f"Error creating model zip archive: {e}")
    
    # Close WandB
    if args.use_wandb and trainer.is_main_process:
        wandb.finish()


if __name__ == '__main__':
    main()