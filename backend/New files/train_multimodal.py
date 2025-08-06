from multi_modal_model import MultiModalDeepfakeModel
from dataset_loader import get_data_loaders, get_transforms, get_transforms_enhanced
import torch
from skin_analyzer import SkinColorAnalyzer
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
from datetime import datetime, timedelta
import shutil
import torch.multiprocessing as mp
import traceback
import glob
import atexit
import gc
import signal
import sys

# 🧼 PROCESS SAFETY: Set multiprocessing method early and force it
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# 🧼 GPU Memory Management and Process Safety
def cleanup_gpu_memory():
    """Clean up GPU memory and handle CUDA context properly."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    except Exception as e:
        print(f"GPU cleanup warning: {e}")

def cleanup_processes():
    """Clean up processes and GPU memory."""
    try:
        cleanup_gpu_memory()
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Process cleanup warning: {e}")

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print(f"🧼 Received signal {signum}, cleaning up...")
    cleanup_processes()
    sys.exit(0)

# Register cleanup functions
atexit.register(cleanup_processes)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


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
        if 'model' in locals() and hasattr(model, 'get_attention_maps'):
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
                
        # NEW: Visualize facial landmarks if available
        if 'facial_landmarks' in inputs and inputs['facial_landmarks'] is not None:
            try:
                landmarks = inputs['facial_landmarks'][sample_idx].cpu().numpy()
                
                for t in range(min(max_viz_frames, landmarks.shape[0])):
                    # Load the frame
                    frame_path = os.path.join(frames_dir, f'frame_{t}.jpg')
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        
                        # Draw landmarks
                        lm = landmarks[t]
                        for i in range(0, len(lm), 2):
                            if i+1 < len(lm) and lm[i] > 0 and lm[i+1] > 0:
                                x, y = int(lm[i]), int(lm[i+1])
                                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                        
                        # Save the frame with landmarks
                        cv2.imwrite(
                            os.path.join(frames_dir, f'frame_{t}_landmarks.jpg'),
                            frame
                        )
            except Exception as e:
                print(f"Error visualizing facial landmarks: {e}")
                
        # NEW: Visualize head pose if available
        if 'head_pose' in inputs and inputs['head_pose'] is not None:
            try:
                head_pose = inputs['head_pose'][sample_idx].cpu().numpy()
                
                # Create a plot of head pose over time
                plt.figure(figsize=(10, 5))
                time_axis = np.arange(len(head_pose))
                plt.plot(time_axis, head_pose[:, 0], label='Pitch', color='red')
                plt.plot(time_axis, head_pose[:, 1], label='Yaw', color='green')
                plt.plot(time_axis, head_pose[:, 2], label='Roll', color='blue')
                plt.title(f'Head Pose Over Time - Sample {sample_idx}')
                plt.xlabel('Frame')
                plt.ylabel('Angle (normalized)')
                plt.legend()
                plt.grid(True)
                
                # Save the plot
                plt.savefig(os.path.join(frames_dir, 'head_pose.png'))
                plt.close()
            except Exception as e:
                print(f"Error visualizing head pose: {e}")
                
        # NEW: Visualize eye blink pattern if available
        if 'eye_blink_features' in inputs and inputs['eye_blink_features'] is not None:
            try:
                blink_pattern = inputs['eye_blink_features'][sample_idx].cpu().numpy()
                
                # Create a plot of blink pattern over time
                plt.figure(figsize=(10, 3))
                time_axis = np.arange(len(blink_pattern))
                plt.plot(time_axis, blink_pattern, label='Blink Score', color='purple')
                plt.title(f'Eye Blink Pattern - Sample {sample_idx}')
                plt.xlabel('Frame')
                plt.ylabel('Blink Score')
                plt.ylim([0, 1])
                plt.grid(True)
                
                # Save the plot
                plt.savefig(os.path.join(frames_dir, 'eye_blink_pattern.png'))
                plt.close()
            except Exception as e:
                print(f"Error visualizing eye blink pattern: {e}")
                
        # NEW: Visualize pulse signal if available
        if 'pulse_signal' in inputs and inputs['pulse_signal'] is not None:
            try:
                pulse = inputs['pulse_signal'][sample_idx].cpu().numpy()
                
                # Create a plot of pulse signal over time
                plt.figure(figsize=(10, 3))
                time_axis = np.arange(len(pulse))
                plt.plot(time_axis, pulse, label='Pulse Signal', color='red')
                plt.title(f'Pulse Signal - Sample {sample_idx}')
                plt.xlabel('Frame')
                plt.ylabel('Signal Amplitude')
                plt.grid(True)
                
                # Save the plot
                plt.savefig(os.path.join(frames_dir, 'pulse_signal.png'))
                plt.close()
            except Exception as e:
                print(f"Error visualizing pulse signal: {e}")
                
        # NEW: Visualize frequency domain features if available
        if 'frequency_features' in inputs and inputs['frequency_features'] is not None:
            try:
                freq_features = inputs['frequency_features'][sample_idx].cpu().numpy()
                
                # Create heatmap of frequency features
                plt.figure(figsize=(6, 6))
                sns.heatmap(freq_features[0], cmap='viridis')
                plt.title(f'Frequency Domain Features - Sample {sample_idx}')
                plt.axis('off')
                
                # Save the plot
                plt.savefig(os.path.join(frames_dir, 'frequency_features.png'))
                plt.close()
            except Exception as e:
                print(f"Error visualizing frequency features: {e}")
                
    except Exception as e:
        print(f"Error saving visualizations: {e}")


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return allocated, reserved
    return 0, 0

def check_memory_and_reduce_batch_size(batch, max_retries=3, trainer=None):
    """Check memory usage and reduce batch size if needed."""
    current_batch_size = len(batch['label']) if 'label' in batch else 1
    
    for retry in range(max_retries):
        try:
            allocated, reserved = get_gpu_memory_usage()
            if allocated > 35.0:  # If using more than 35GB, reduce batch size
                new_batch_size = max(1, current_batch_size // 2)
                print(f"[MEMORY] Reducing batch size from {current_batch_size} to {new_batch_size} (allocated: {allocated:.2f}GB)")
                
                # Track memory warnings
                if trainer is not None:
                    trainer.memory_warnings += 1
                
                # Create smaller batch - ENSURE ALL COMPONENTS ARE REDUCED CONSISTENTLY
                reduced_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        if value.size(0) > new_batch_size:
                            reduced_batch[key] = value[:new_batch_size]
                        else:
                            reduced_batch[key] = value
                    elif isinstance(value, list):
                        if len(value) > new_batch_size:
                            reduced_batch[key] = value[:new_batch_size]
                        else:
                            reduced_batch[key] = value
                    else:
                        reduced_batch[key] = value
                
                # Verify batch consistency after reduction
                actual_batch_size = len(reduced_batch['label']) if 'label' in reduced_batch else new_batch_size
                if actual_batch_size != new_batch_size:
                    print(f"[WARNING] Batch size inconsistency after reduction: expected {new_batch_size}, got {actual_batch_size}")
                
                batch = reduced_batch
                current_batch_size = new_batch_size
                clear_gpu_cache()
            else:
                break
        except Exception as e:
            print(f"[MEMORY] Error checking memory usage: {e}")
            break
    
    return batch

def move_batch_to_device(batch, device, trainer=None):
    """Safely move batch items to device, handling non-tensor items properly."""
    # Check and optimize memory before moving to device
    batch = check_memory_and_reduce_batch_size(batch, trainer=trainer)
    
    device_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            device_batch[key] = value.to(device, non_blocking=True)
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
        self.distributed = config.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1
        self.local_rank = config.local_rank
        self.is_main_process = not self.distributed or self.local_rank == 0
        
        # Set device properly for distributed training
        if self.distributed:
            # In distributed mode, use the local rank as device ID
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.amp_enabled = config.amp_enabled and self.device.type == 'cuda'
        
        # Set up directories
        self.setup_directories()
        
        # In distributed mode, synchronize directory paths across all processes
        if self.distributed:
            # Wait for main process to create directories
            if dist.is_initialized():
                dist.barrier()
            
            # Non-main processes need to get the directory paths from main process
            if not self.is_main_process:
                # Find the most recent run directory created by main process
                import glob
                run_pattern = os.path.join(self.config.output_dir, "run_*")
                run_dirs = glob.glob(run_pattern)
                if run_dirs:
                    # Get the most recent run directory
                    self.run_dir = max(run_dirs, key=os.path.getctime)
                    self.log_dir = os.path.join(self.run_dir, "logs")
                    self.viz_dir = os.path.join(self.run_dir, "visualizations")
                    self.plot_dir = os.path.join(self.run_dir, "plots")
                    print(f"[Rank {self.local_rank}] Using shared run directory: {self.run_dir}")
                else:
                    print(f"[Rank {self.local_rank}] Warning: No run directory found, using fallback")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.run_dir = os.path.join(self.config.output_dir, f"run_{timestamp}")
                    self.log_dir = os.path.join(self.run_dir, "logs")
                    self.viz_dir = os.path.join(self.run_dir, "visualizations")
                    self.plot_dir = os.path.join(self.run_dir, "plots")
            
            # Another barrier to ensure all processes have the correct paths
            if dist.is_initialized():
                dist.barrier()
        
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
        
        # Initialize numerical stability tracking
        self.nan_count = 0
        self.total_batches = 0
        
        # Initialize memory management tracking
        self.memory_warnings = 0
        self.oom_errors = 0
        
        # Initialize best model tracking
        self.best_val_accuracy = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0
        
        # Initialize scaler for mixed precision
        self.scaler = GradScaler() if self.amp_enabled else None
        
        # Create a specific run folder in the checkpoint directory
        # In distributed training, only main process creates the directory
        if self.distributed:
            # Use the same timestamp across all processes by broadcasting from rank 0
            if self.local_rank == 0:
                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                self.timestamp = None
            
            # Synchronize timestamp across all processes (simplified approach)
            # In practice, you'd use dist.broadcast, but we'll use a simpler approach
            import time
            if self.local_rank == 0:
                # Main process creates timestamp file
                timestamp_file = os.path.join(self.config.checkpoint_dir, "current_run_timestamp.txt")
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                with open(timestamp_file, 'w') as f:
                    f.write(self.timestamp)
            
            # Wait a bit and then all processes read the same timestamp
            time.sleep(1)
            timestamp_file = os.path.join(self.config.checkpoint_dir, "current_run_timestamp.txt")
            if os.path.exists(timestamp_file):
                with open(timestamp_file, 'r') as f:
                    self.timestamp = f.read().strip()
            else:
                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"run_{self.timestamp}")
        
        # Only main process creates the directory in distributed training
        if not self.distributed or self.is_main_process:
            os.makedirs(self.run_checkpoint_dir, exist_ok=True)
        
        # Initialize starting epoch
        self.start_epoch = 0
        
        # Check for resume checkpoint
        if hasattr(config, 'resume_checkpoint') and config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training state."""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint file not found at {checkpoint_path}")
            return
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Load checkpoint on appropriate device
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model weights
            if self.distributed:
                self.model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Update metrics and best values if available
            if 'accuracy' in checkpoint:
                self.best_val_accuracy = checkpoint['accuracy']
            
            if 'f1_score' in checkpoint:
                self.best_val_f1 = checkpoint['f1_score']
            
            # Set the starting epoch
            self.start_epoch = checkpoint.get('epoch', 0)
            
            # Load metrics history if available
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            
            print(f"Checkpoint loaded successfully. Resuming from epoch {self.start_epoch}")
            print(f"Previous best metrics - Accuracy: {self.best_val_accuracy:.4f}, F1: {self.best_val_f1:.4f}")
        
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
        
    def setup_directories(self):
        """Set up directories for saving models, logs, and visualizations."""
        # Only main process creates directories in distributed training
        if self.distributed and not self.is_main_process:
            return
            
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
            
            # Define detailed tags for better tracking
            tags = [
                f"backbone_visual:{self.config.backbone_visual}",
                f"backbone_audio:{self.config.backbone_audio}",
                f"fusion:{self.config.fusion_type}"
            ]
            
            # Add enhanced feature tags
            if hasattr(self.config, 'detect_faces') and self.config.detect_faces:
                tags.append("facial_analysis")
            if hasattr(self.config, 'enhanced_preprocessing') and self.config.enhanced_preprocessing:
                tags.append("enhanced_preprocessing")
            if hasattr(self.config, 'enhanced_augmentation') and self.config.enhanced_augmentation:
                tags.append("enhanced_augmentation")
            
            wandb.init(
                project=project_name,
                name=run_name,
                config=vars(self.config),
                tags=tags
            )
            
            print(f"Weights & Biases initialized: {project_name}/{run_name}")
        except Exception as e:
            print(f"Error initializing Weights & Biases: {e}")
            self.config.use_wandb = False
    
    def save_config(self):
        """Save configuration to JSON file."""
        # Only main process saves config in distributed training
        if self.distributed and not self.is_main_process:
            return
            
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
        
        # Initialize distributed training BEFORE setting up data loaders
        if self.distributed:
            # Initialize the process group with timeout
            if not dist.is_initialized():
                # Set timeout for NCCL operations to prevent hanging (use new env var names)
                os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes
                os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'  # Updated env var name
                os.environ['NCCL_DEBUG'] = 'INFO'  # Enable NCCL debugging
                
                # Initialize process group with explicit device_id to avoid GPU mapping warnings
                dist.init_process_group(
                    backend='nccl', 
                    init_method='env://',
                    timeout=timedelta(seconds=1800)  # 30 minute timeout
                )
            
            # Set the device for this process BEFORE any CUDA operations
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            
            print(f"[Rank {self.local_rank}] Using CUDA device: {self.device}")
            print(f"[Rank {self.local_rank}] Process group initialized with {dist.get_world_size()} processes")
            
            # Add barrier to ensure all processes are synchronized
            if dist.is_initialized():
                dist.barrier()
                print(f"[Rank {self.local_rank}] Passed initialization barrier")
        
        # 🧼 OPTIMIZED: Check num_workers setting based on benchmarks
        if self.config.num_workers > 0:
            print(f"⚠️  EXPERIMENTAL: Using num_workers={self.config.num_workers}")
            print("📊 Benchmarks show num_workers=0 is fastest for this multimodal dataset")
            print("🧼 If training is slow, try --num_workers=0")
        else:
            print(f"✅ OPTIMAL: Using num_workers=0 (benchmarked as fastest for multimodal data)")
            print("🚀 Single-threaded loading avoids I/O contention with complex preprocessing")
        
        # Get data loaders with appropriate options
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
            temporal_features=self.config.temporal_features,
            enhanced_preprocessing=getattr(self.config, 'enhanced_preprocessing', True),
            enhanced_augmentation=getattr(self.config, 'enhanced_augmentation', False),
            multiprocessing_context="spawn"  # 🧼 SAFETY: Prevent GPU context corruption in multiprocessing
        )
        
        print(f"✅ Data loaders created: {len(self.train_loader)} train batches, "
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
            debug=self.config.debug,
            enable_skin_color_analysis=getattr(self.config, 'enable_skin_color_analysis', False),
            enable_advanced_physiological=getattr(self.config, 'enable_advanced_physiological', False)
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
            # Ensure the model is on the correct device before wrapping with DDP
            print(f"[Rank {self.local_rank}] Moving model to device: {self.device}")
            
            # Explicitly set the device before DDP wrapping to avoid GPU mapping warnings
            torch.cuda.set_device(self.local_rank)
            self.model = self.model.to(self.device)
            
            # Wrap model with DDP, explicitly specifying device_ids to avoid warnings
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank], 
                output_device=self.local_rank,
                find_unused_parameters=True,  # Re-enable to handle dynamic parameter usage
                static_graph=False  # Disable static graph to allow dynamic parameter usage
            )
            
            # Note: Static graph disabled due to dynamic parameter usage in multimodal model
            
            print(f"[Rank {self.local_rank}] Model wrapped with DDP on device {self.local_rank}")
            print(f"[Rank {self.local_rank}] Dynamic parameter detection enabled (find_unused_parameters=True)")
        # Enable DataParallel for multi-GPU if available and not using distributed
        elif torch.cuda.device_count() > 1:
            print(f"[INFO] Using DataParallel on {torch.cuda.device_count()} GPUs.")
            self.model = torch.nn.DataParallel(self.model)

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
        
        # Debug: Print GPU usage information
        self.print_gpu_usage_info()
    
    def print_gpu_usage_info(self):
        """Print detailed information about GPU usage."""
        print("\n" + "="*60)
        print("🔥 GPU USAGE INFORMATION")
        print("="*60)
        
        if torch.cuda.is_available():
            total_gpus = torch.cuda.device_count()
            print(f"📊 Total GPUs Available: {total_gpus}")
            
            # Show which GPUs are visible
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
                print(f"🎯 CUDA_VISIBLE_DEVICES: {visible_devices}")
                gpu_list = visible_devices.split(',')
                print(f"📝 GPUs in use: {len(gpu_list)} GPUs (indices: {gpu_list})")
            else:
                print(f"📝 All {total_gpus} GPUs are visible (no CUDA_VISIBLE_DEVICES set)")
            
            # Show current device
            current_device = torch.cuda.current_device()
            print(f"🎮 Current CUDA Device: {current_device}")
            
            # Show GPU memory for each visible device
            print(f"\n💾 GPU Memory Status:")
            for i in range(total_gpus):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                    gpu_name = torch.cuda.get_device_properties(i).name
                    print(f"   GPU {i} ({gpu_name}): {memory_allocated:.2f}GB/{memory_total:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                except Exception as e:
                    print(f"   GPU {i}: Error getting memory info - {e}")
            
            # Show model parallel status
            print(f"\n🚀 Model Parallel Status:")
            if self.distributed:
                print(f"   ✅ Using DistributedDataParallel (DDP)")
                print(f"   📍 Local Rank: {self.local_rank}")
                print(f"   🌟 Main Process: {self.is_main_process}")
            elif hasattr(self.model, 'module'):
                # DataParallel wraps model in .module
                print(f"   ✅ Using DataParallel")
                if hasattr(self.model, 'device_ids'):
                    print(f"   📍 Device IDs: {self.model.device_ids}")
                else:
                    print(f"   📍 Using all available GPUs")
            else:
                print(f"   ❌ Using Single GPU only")
                print(f"   💡 Consider using --distributed for better multi-GPU performance")
            
        else:
            print("❌ CUDA not available - using CPU")
        
        print("="*60)
    
    def save_intermediate_checkpoint(self, epoch, batch_idx):
        """Save intermediate checkpoint during training."""
        # Only main process saves checkpoints in distributed training
        if not self.is_main_process:
            return
            
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
        
        # Handle distributed vs non-distributed model state dict
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        elif hasattr(self.model, 'module'):  # DataParallel
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
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
        
        # Apply model stabilization measures
        if hasattr(self.model, 'stabilize_model'):
            self.model.stabilize_model()
        
        # Gradually unfreeze visual layers
        if hasattr(self.model, 'unfreeze_visual_layers'):
            self.model.unfreeze_visual_layers(epoch)
        
        epoch_loss = 0
        y_true, y_pred, y_probs = [], [], []
        
        train_progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]", 
                           disable=not self.is_main_process)
        
        # Add debugging info for distributed training
        if self.is_main_process:
            print(f"[RANK {self.local_rank}] Starting training epoch {epoch+1} with {len(self.train_loader)} batches")
        
        for batch_idx, batch in enumerate(train_progress):
            # Debug: Print batch loading progress
            if self.is_main_process and batch_idx == 0:
                print(f"[RANK {self.local_rank}] Processing first batch...")
            elif batch_idx % 10 == 0 and self.is_main_process:
                print(f"[RANK {self.local_rank}] Processing batch {batch_idx}/{len(self.train_loader)}")
            try:
                # Add timeout protection for first batch
                if batch_idx == 0:
                    print(f"[RANK {self.local_rank}] Loading first batch, this may take time...")
                    start_time = time.time()
                
                # Clear GPU cache periodically to prevent memory buildup
                if batch_idx % 10 == 0:
                    clear_gpu_cache()
                
                # Check memory usage before processing
                allocated, reserved = get_gpu_memory_usage()
                if allocated > 30.0:  # If memory usage is high, force cleanup
                    print(f"[MEMORY] High memory usage detected: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    clear_gpu_cache()
                
                # Move batch to device with timeout protection
                print(f"[RANK {self.local_rank}] Moving batch {batch_idx} to device...")
                batch = move_batch_to_device(batch, self.device, trainer=self)
                
                if batch_idx == 0:
                    load_time = time.time() - start_time
                    print(f"[RANK {self.local_rank}] First batch loaded in {load_time:.2f} seconds")
                # --- SkinColorAnalyzer integration ---
                if getattr(self.config, 'enable_skin_color_analysis', False) and 'video_frames' in batch:
                    if not hasattr(self, '_skin_color_analyzer'):
                        self._skin_color_analyzer = SkinColorAnalyzer().to(self.device)
                    with torch.no_grad():
                        frames = batch['video_frames']
                        if frames.dim() == 5 and frames.shape[2] == 3:
                            frames_small = torch.nn.functional.interpolate(
                                frames.view(-1, 3, frames.shape[3], frames.shape[4]),
                                size=(56, 56), mode='bilinear', align_corners=False
                            ).view(frames.shape[0], frames.shape[1], 3, 56, 56)
                        else:
                            frames_small = frames
                        skin_features = self._skin_color_analyzer(frames_small)
                        batch['skin_color_features'] = skin_features
                
                # Get labels
                labels = batch['label']
                
                # For first few batches, reduce effective batch size to prevent hanging
                if batch_idx < 3 and labels.shape[0] > 40:
                    print(f"[RANK {self.local_rank}] Reducing batch size for initial batches: {labels.shape[0]} -> 40")
                    # Truncate all batch elements to smaller size
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor) and value.shape[0] == labels.shape[0]:
                            batch[key] = value[:40]
                    labels = batch['label']
                
                # Debug: Print initial batch information
                if batch_idx < 5:  # Only for first few batches to avoid spam
                    print(f"[DEBUG] Batch {batch_idx} - Initial labels shape: {labels.shape}")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            print(f"[DEBUG] Batch {batch_idx} - {key} shape: {value.shape}")
                        elif isinstance(value, list):
                            print(f"[DEBUG] Batch {batch_idx} - {key} length: {len(value)}")
                
                # Validate batch consistency before forward pass
                expected_batch_size = labels.shape[0]
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] != expected_batch_size:
                        print(f"[WARNING] Batch size mismatch in {key}: expected {expected_batch_size}, got {value.shape[0]}")
                        # Fix the mismatch by truncating or padding
                        if value.shape[0] > expected_batch_size:
                            batch[key] = value[:expected_batch_size]
                        elif value.shape[0] < expected_batch_size:
                            # Skip this batch if we can't fix it easily
                            print(f"[ERROR] Cannot fix batch size mismatch for {key}, skipping batch")
                            continue

                # Forward pass with mixed precision
                self.optimizer.zero_grad()

                # Count successful vs failed samples for debugging
                success_count = 0
                total_samples = len(labels)

                def _check_and_fix_batch_size(outputs, labels):
                    """Check and fix batch size mismatches between outputs and labels."""
                    out_bs = outputs.shape[0] if hasattr(outputs, 'shape') and len(outputs.shape) > 0 else 1
                    lbl_bs = labels.shape[0] if hasattr(labels, 'shape') and len(labels.shape) > 0 else 1
                    
                    if out_bs != lbl_bs:
                        print(f"[ERROR] Output batch size {out_bs} does not match label batch size {lbl_bs} at batch {batch_idx}")
                        print(f"[DEBUG] outputs.shape: {getattr(outputs, 'shape', None)} labels.shape: {getattr(labels, 'shape', None)}")
                        
                        # Try to fix the mismatch
                        if out_bs > lbl_bs:
                            # Truncate outputs to match labels
                            print(f"[FIX] Truncating outputs from {out_bs} to {lbl_bs}")
                            outputs = outputs[:lbl_bs]
                            return outputs, labels, True
                        elif lbl_bs > out_bs:
                            # Truncate labels to match outputs 
                            print(f"[FIX] Truncating labels from {lbl_bs} to {out_bs}")
                            labels = labels[:out_bs]
                            return outputs, labels, True
                    return outputs, labels, True

                if self.amp_enabled:
                    with autocast():
                        # Debug: Check input shapes before model forward pass
                        if batch_idx < 5:
                            print(f"[DEBUG] Before model forward - labels shape: {labels.shape}")
                            if 'video_frames' in batch:
                                print(f"[DEBUG] Before model forward - video_frames shape: {batch['video_frames'].shape}")
                            if 'audio' in batch:
                                print(f"[DEBUG] Before model forward - audio shape: {batch['audio'].shape}")
                        
                        outputs, results = self.model(batch)
                        
                        # Debug: Check output shapes after model forward pass
                        if batch_idx < 5:
                            print(f"[DEBUG] After model forward - outputs shape: {outputs.shape}")
                            print(f"[DEBUG] After model forward - labels shape: {labels.shape}")
                        
                        # Check for NaN in outputs
                        if torch.isnan(outputs).any():
                            print(f"[ERROR] NaN detected in model outputs at batch {batch_idx}")
                            print(f"[DEBUG] Output shape: {outputs.shape}")
                            print(f"[DEBUG] Output stats - min: {outputs.min()}, max: {outputs.max()}, mean: {outputs.mean()}")
                            # Reset optimizer state and skip this batch
                            self.optimizer.zero_grad()
                            self.nan_count += 1
                            continue
                        # Add numerical stability to outputs
                        outputs = torch.clamp(outputs, min=-50, max=50)  # Prevent extreme values
                        # Check and fix batch size
                        outputs, labels, batch_valid = _check_and_fix_batch_size(outputs, labels)
                        if not batch_valid:
                            print(f"[ERROR] Cannot fix batch size mismatch, skipping batch {batch_idx}")
                            self.optimizer.zero_grad()
                            continue
                        loss = self.criterion(outputs, labels)
                        # Check for NaN in loss
                        if torch.isnan(loss):
                            print(f"[ERROR] NaN loss detected at batch {batch_idx}")
                            print(f"[DEBUG] Labels: {labels}")
                            print(f"[DEBUG] Outputs: {outputs}")
                            continue
                        # Add regularization for deepfake type if enabled
                        if self.config.detect_deepfake_type and 'deepfake_type' in results and results['deepfake_type'] is not None:
                            if 'deepfake_type' in batch and batch['deepfake_type'] is not None:
                                # Ensure deepfake_type target is a tensor
                                deepfake_type_target = batch['deepfake_type']
                                if isinstance(deepfake_type_target, list):
                                    deepfake_type_target = torch.tensor(deepfake_type_target, device=outputs.device, dtype=torch.long)
                                elif not isinstance(deepfake_type_target, torch.Tensor):
                                    deepfake_type_target = torch.tensor([deepfake_type_target], device=outputs.device, dtype=torch.long)
                                
                                # Ensure batch size consistency
                                if deepfake_type_target.shape[0] != results['deepfake_type'].shape[0]:
                                    min_batch = min(deepfake_type_target.shape[0], results['deepfake_type'].shape[0])
                                    deepfake_type_target = deepfake_type_target[:min_batch]
                                    deepfake_type_pred = results['deepfake_type'][:min_batch]
                                else:
                                    deepfake_type_pred = results['deepfake_type']
                                
                                try:
                                    deepfake_type_loss = nn.CrossEntropyLoss()(deepfake_type_pred, deepfake_type_target)
                                    if not torch.isnan(deepfake_type_loss):
                                        loss += self.config.deepfake_type_weight * deepfake_type_loss
                                except Exception as e:
                                    if self.debug:
                                        print(f"[WARNING] Error computing deepfake type loss: {e}")
                                    # Skip deepfake type loss if there's an error
                    
                    # Backward pass with scaler (AMP enabled)
                    try:
                        self.scaler.scale(loss).backward()
                        
                        # Check for NaN in gradients using model method
                        if hasattr(self.model, 'check_for_nan_gradients') and self.model.check_for_nan_gradients():
                            print(f"[ERROR] Skipping backward pass due to NaN gradients at batch {batch_idx}")
                            self.optimizer.zero_grad()
                            # Apply model stabilization
                            if hasattr(self.model, 'stabilize_model'):
                                self.model.stabilize_model()
                            # Reset scaler state
                            self.scaler.update()
                            # Skip this batch but continue training
                            continue
                        
                        # Apply model's gradient clipping with AMP
                        unscale_called = False
                        if hasattr(self.model, 'clip_gradients'):
                            self.scaler.unscale_(self.optimizer)
                            unscale_called = True
                            self.model.clip_gradients(max_norm=self.config.gradient_clip)
                        
                        # Additional gradient clipping if configured
                        if self.config.gradient_clip > 0:
                            if not hasattr(self.model, 'clip_gradients'):  # Only if not already done
                                self.scaler.unscale_(self.optimizer)
                                unscale_called = True
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                        
                        # Ensure unscale is called at least once before step
                        if not unscale_called:
                            self.scaler.unscale_(self.optimizer)
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                    except Exception as scaler_error:
                        print(f"[ERROR] AMP scaler error at batch {batch_idx}: {scaler_error}")
                        # Reset optimizer and scaler state
                        self.optimizer.zero_grad()
                        # Recreate scaler to fix state issues
                        self.scaler = GradScaler(enabled=self.amp_enabled, init_scale=2**8)
                        print(f"[INFO] Recreated AMP scaler with fresh state")
                        continue
                else:
                    outputs, results = self.model(batch)
                    # Check for NaN in outputs
                    if torch.isnan(outputs).any():
                        print(f"[ERROR] NaN detected in model outputs at batch {batch_idx}")
                        print(f"[DEBUG] Output shape: {outputs.shape}")
                        print(f"[DEBUG] Output stats - min: {outputs.min()}, max: {outputs.max()}, mean: {outputs.mean()}")
                        # Reset optimizer state and skip this batch
                        self.optimizer.zero_grad()
                        self.nan_count += 1
                        continue
                    # Add numerical stability to outputs
                    outputs = torch.clamp(outputs, min=-50, max=50)  # Prevent extreme values
                    # Check and fix batch size
                    outputs, labels, batch_valid = _check_and_fix_batch_size(outputs, labels)
                    if not batch_valid:
                        print(f"[ERROR] Cannot fix batch size mismatch, skipping batch {batch_idx}")
                        self.optimizer.zero_grad()
                        continue
                    loss = self.criterion(outputs, labels)
                    # Check for NaN in loss
                    if torch.isnan(loss):
                        print(f"[ERROR] NaN loss detected at batch {batch_idx}")
                        print(f"[DEBUG] Labels: {labels}")
                        print(f"[DEBUG] Outputs: {outputs}")
                        continue
                    # Add regularization for deepfake type if enabled
                    if self.config.detect_deepfake_type and 'deepfake_type' in results and results['deepfake_type'] is not None:
                        if 'deepfake_type' in batch and batch['deepfake_type'] is not None:
                            # Ensure deepfake_type target is a tensor
                            deepfake_type_target = batch['deepfake_type']
                            if isinstance(deepfake_type_target, list):
                                deepfake_type_target = torch.tensor(deepfake_type_target, device=outputs.device, dtype=torch.long)
                            elif not isinstance(deepfake_type_target, torch.Tensor):
                                deepfake_type_target = torch.tensor([deepfake_type_target], device=outputs.device, dtype=torch.long)
                            
                            # Ensure batch size consistency
                            if deepfake_type_target.shape[0] != results['deepfake_type'].shape[0]:
                                min_batch = min(deepfake_type_target.shape[0], results['deepfake_type'].shape[0])
                                deepfake_type_target = deepfake_type_target[:min_batch]
                                deepfake_type_pred = results['deepfake_type'][:min_batch]
                            else:
                                deepfake_type_pred = results['deepfake_type']
                            
                            try:
                                deepfake_type_loss = nn.CrossEntropyLoss()(deepfake_type_pred, deepfake_type_target)
                                if not torch.isnan(deepfake_type_loss):
                                    loss += self.config.deepfake_type_weight * deepfake_type_loss
                            except Exception as e:
                                if getattr(self.config, 'debug', False):
                                    print(f"[WARNING] Error computing deepfake type loss: {e}")
                                # Skip deepfake type loss if there's an error
                    # Backward pass
                    loss.backward()
                    # Check for NaN in gradients using model method
                    if hasattr(self.model, 'check_for_nan_gradients') and self.model.check_for_nan_gradients():
                        print(f"[ERROR] Skipping backward pass due to NaN gradients at batch {batch_idx}")
                        self.optimizer.zero_grad()
                        continue
                    # Apply model's gradient clipping
                    if hasattr(self.model, 'clip_gradients'):
                        self.model.clip_gradients(max_norm=1.0)
                    # Additional gradient clipping if configured
                    elif self.config.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                
                # Update learning rate with warmup if enabled
                if self.warmup_scheduler is not None and epoch < self.config.warmup_epochs:
                    self.warmup_scheduler.step()
                
                # Track total batches and check for excessive NaN occurrence
                self.total_batches += 1
                if self.total_batches > 0 and self.nan_count / self.total_batches > 0.1:  # If more than 10% NaN
                    print(f"[WARNING] High NaN rate detected ({self.nan_count}/{self.total_batches}). Reducing learning rate.")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    self.nan_count = 0  # Reset counter after adjustment
                    self.total_batches = 0
                
                # Update progress bar
                loss_display = loss.item() if not torch.isnan(loss) else "nan"
                train_progress.set_postfix(loss=f"{loss_display:.4f}" if isinstance(loss_display, float) else loss_display, 
                                         lr=f"{get_lr(self.optimizer):.6f}")
                
                # Accumulate loss only if it's not NaN
                if not torch.isnan(loss):
                    epoch_loss += loss.item()
                else:
                    print(f"[WARNING] Skipping NaN loss in accumulation at batch {batch_idx}")
                
                # Accumulate predictions and labels for metrics calculation
                y_true.extend(labels.cpu().numpy())
                predictions = outputs.argmax(1).cpu().numpy()
                y_pred.extend(predictions)
                y_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
                
                # Log batch results to WandB
                if self.config.use_wandb and self.is_main_process and batch_idx % self.config.log_interval == 0:
                    # Log component weights if available
                    component_weights = {}
                    if 'component_weights' in results and results['component_weights'] is not None:
                        for i, weight in enumerate(results['component_weights']):
                            component_weights[f'component_weight_{i}'] = weight.item()
                    
                    wandb.log({
                        'batch_loss': loss.item(),
                        'learning_rate': get_lr(self.optimizer),
                        'batch': batch_idx + epoch * len(self.train_loader),
                        **component_weights
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
                
            except torch.cuda.OutOfMemoryError as oom_error:
                print(f"[CUDA OOM] Out of memory error in training batch {batch_idx}: {oom_error}")
                print(f"[CUDA OOM] Attempting to recover by clearing cache and reducing batch size...")
                
                # Track OOM errors
                self.oom_errors += 1
                
                # Clear GPU cache
                clear_gpu_cache()
                
                # Skip this batch and continue
                self.optimizer.zero_grad()
                continue
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                
                # Clear GPU cache on any error to prevent memory leaks
                clear_gpu_cache()
                self.optimizer.zero_grad()
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
        
        # Track component contributions to analyze feature importance
        all_component_contributions = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_progress):
                try:
                    # Clear GPU cache periodically during validation
                    if batch_idx % 20 == 0:
                        clear_gpu_cache()
                    
                    # Move batch to device
                    batch = move_batch_to_device(batch, self.device, trainer=self)
                    
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
                    
                    # Track component contributions for feature importance analysis
                    if 'component_contributions' in results and results['component_contributions'] is not None:
                        for key, value in results['component_contributions'].items():
                            if key not in all_component_contributions:
                                all_component_contributions[key] = []
                            
                            # Extract scalar value if it's a tensor
                            if isinstance(value, torch.Tensor):
                                if value.numel() == 1:
                                    all_component_contributions[key].append(value.item())
                                else:
                                    # Average for multi-element tensors
                                    all_component_contributions[key].append(value.mean().item())
                            elif isinstance(value, (int, float)):
                                all_component_contributions[key].append(value)
                    
                except torch.cuda.OutOfMemoryError as oom_error:
                    print(f"[CUDA OOM] Out of memory error in validation batch {batch_idx}: {oom_error}")
                    print(f"[CUDA OOM] Clearing cache and continuing...")
                    
                    # Track OOM errors
                    self.oom_errors += 1
                    
                    clear_gpu_cache()
                    continue
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    clear_gpu_cache()
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
                
            # Log component importance for feature analysis to WandB
            if self.config.use_wandb and all_component_contributions:
                component_importance = {}
                for key, values in all_component_contributions.items():
                    if values:
                        avg_value = np.mean(values)
                        component_importance[f"component_{key}"] = avg_value
                
                wandb.log(component_importance)
                
                # Create feature importance chart
                if component_importance:
                    plt.figure(figsize=(12, 6))
                    keys = sorted(component_importance.keys())
                    values = [component_importance[k] for k in keys]
                    
                    # Sort by value
                    sorted_indices = np.argsort(values)
                    sorted_keys = [keys[i] for i in sorted_indices]
                    sorted_values = [values[i] for i in sorted_indices]
                    
                    plt.barh(sorted_keys, sorted_values)
                    plt.title("Feature Importance Scores")
                    plt.xlabel("Average Contribution")
                    plt.tight_layout()
                    
                    feature_importance_path = os.path.join(self.plot_dir, f"feature_importance_epoch_{epoch+1}.png")
                    plt.savefig(feature_importance_path)
                    plt.close()
                    
                    wandb.log({f"feature_importance_epoch_{epoch+1}": wandb.Image(feature_importance_path)})
        
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
        
        # Track detailed component results for analysis
        detailed_component_results = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_progress):
                try:
                    # Move batch to device
                    batch = move_batch_to_device(batch, self.device, trainer=self)
                    
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
                        
                        # Basic result data
                        results_data['file_path'].append(file_paths[i] if i < len(file_paths) else 'unknown')
                        results_data['true_label'].append(int(batch_y_true[i]))
                        results_data['pred_label'].append(int(predictions[i]))
                        results_data['confidence'].append(float(confidences[i]))
                        results_data['issues_found'].append('; '.join(issues))
                        results_data['confidence_score'].append(float(confidence_score))
                        
                        # Extract and store component contributions for detailed analysis
                        if 'component_contributions' in results and results['component_contributions'] is not None:
                            for key, value in results['component_contributions'].items():
                                if key not in detailed_component_results:
                                    detailed_component_results[key] = []
                                
                                # Extract value for this sample
                                if isinstance(value, torch.Tensor):
                                    if value.numel() == 1:
                                        detailed_component_results[key].append(value.item())
                                    elif i < value.size(0):
                                        # First value for this batch item
                                        if value[i].numel() == 1:
                                            detailed_component_results[key].append(value[i].item())
                                        else:
                                            detailed_component_results[key].append(value[i].mean().item())
                                    else:
                                        detailed_component_results[key].append(0.0)
                                elif isinstance(value, (int, float)):
                                    detailed_component_results[key].append(value)
                                elif isinstance(value, dict) and 'naturalness' in value:
                                    if isinstance(value['naturalness'], torch.Tensor):
                                        detailed_component_results[key].append(value['naturalness'].item())
                                    else:
                                        detailed_component_results[key].append(value['naturalness'])
                                else:
                                    detailed_component_results[key].append(0.0)
                    
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
                    import traceback
                    traceback.print_exc()
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
            # Basic results
            results_df = pd.DataFrame(results_data)
            results_path = os.path.join(self.log_dir, "test_results.csv")
            results_df.to_csv(results_path, index=False)
            print(f"Test results saved to: {results_path}")
            
            # Add component results to detailed analysis DataFrame
            for key, values in detailed_component_results.items():
                if len(values) == len(results_data['file_path']):
                    results_df[f"component_{key}"] = values
            
            # Save enhanced results with component details
            enhanced_results_path = os.path.join(self.log_dir, "test_results_detailed.csv")
            results_df.to_csv(enhanced_results_path, index=False)
            print(f"Detailed test results with component analysis saved to: {enhanced_results_path}")
            
            # Create feature importance visualization
            if detailed_component_results:
                # Calculate average importance for each component
                component_importance = {}
                for key, values in detailed_component_results.items():
                    if values:
                        component_importance[key] = np.mean(values)
                
                # Sort components by importance
                sorted_components = sorted(component_importance.items(), key=lambda x: x[1], reverse=True)
                
                # Create feature importance chart
                plt.figure(figsize=(12, 8))
                component_names = [item[0] for item in sorted_components]
                component_values = [item[1] for item in sorted_components]
                
                plt.barh(component_names, component_values)
                plt.title("Component Importance in Deepfake Detection")
                plt.xlabel("Average Contribution")
                plt.tight_layout()
                
                feature_importance_path = os.path.join(self.plot_dir, "feature_importance_test.png")
                plt.savefig(feature_importance_path)
                plt.close()
                
                if self.config.use_wandb:
                    wandb.log({"feature_importance_test": wandb.Image(feature_importance_path)})
        
        return avg_loss, metrics_dict
    
    def train_and_validate(self):
        """Train and validate the model for the specified number of epochs."""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        
        # Start from the resumed epoch if checkpoint was loaded
        start_epoch = getattr(self, 'start_epoch', 0)
        print(f"Starting from epoch {start_epoch+1}")
        
        for epoch in range(start_epoch, self.config.num_epochs):
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
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.best_val_accuracy = val_acc
                    self.best_epoch = epoch + 1
                    self.early_stop_counter = 0
                    
                    # Save best model
                    self.save_best_model(epoch, val_acc, val_f1)
                else:
                    self.early_stop_counter += 1
                    print(f"Early stopping counter: {self.early_stop_counter}/{self.config.early_stopping_patience}")
                
                if self.early_stop_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best validation F1: {self.best_val_f1:.4f}, Accuracy: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})")
                    break
        
        # Print training summary
        if self.is_main_process:
            print("\nTraining completed!")
            print(f"Best validation F1: {self.best_val_f1:.4f}, Accuracy: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})")
            
            # Load best model for testing
            self.load_best_model()
            
            # Test the model
            test_loss, test_metrics = self.test_model()
            
            # Save final results
            final_results = {
                'best_epoch': self.best_epoch,
                'best_val_accuracy': float(self.best_val_accuracy),
                'best_val_f1': float(self.best_val_f1),
                'test_loss': float(test_loss),
                'test_accuracy': float(test_metrics['accuracy']),
                'test_precision': float(test_metrics['precision']),
                'test_recall': float(test_metrics['recall']),
                'test_f1': float(test_metrics['f1']),
                'test_auc': float(test_metrics['auc']),
                'training_time': time.time() - self.training_start_time if hasattr(self, 'training_start_time') else None,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'config': vars(self.config)
            }
            
            # Save as JSON
            final_results_path = os.path.join(self.log_dir, "final_results.json")
            with open(final_results_path, 'w') as f:
                # Handle non-serializable values
                serializable_results = {}
                for key, value in final_results.items():
                    if key == 'config':
                        # Convert config to serializable dict
                        config_dict = {}
                        for k, v in value.items():
                            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                                config_dict[k] = v
                            else:
                                config_dict[k] = str(v)
                        serializable_results[key] = config_dict
                    elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                        serializable_results[key] = value
                    else:
                        serializable_results[key] = str(value)
                
                json.dump(serializable_results, f, indent=4)
            
            print(f"Final results saved to: {final_results_path}")
    
    def save_checkpoint(self, epoch, accuracy, f1_score):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.run_checkpoint_dir, "regular")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch_{epoch+1}_acc_{accuracy:.4f}_f1_{f1_score:.4f}.pth"
        )
        
        model_state_dict = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_best_model(self, epoch, accuracy, f1_score):
        """Save best model checkpoint."""
        best_model_path = os.path.join(self.run_checkpoint_dir, "best_model.pth")
        
        model_state_dict = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved: {best_model_path}")
        
        # Copy to fixed best model location for easy reference
        shutil.copy(best_model_path, os.path.join(self.model_dir, "best_model.pth"))
    
    def load_best_model(self):
        """Load the best model for testing."""
        best_model_path = os.path.join(self.run_checkpoint_dir, "best_model.pth")
        
        if not os.path.exists(best_model_path):
            print(f"Best model not found at {best_model_path}. Using current model.")
            return
        
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        
        # Load model weights
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        print(f"Best model loaded (Epoch {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.4f}, F1: {checkpoint['f1_score']:.4f})")
    
    def run(self):
        """
        Run the full training pipeline with comprehensive error handling.
        🧼 PROCESS SAFETY: Includes proper cleanup to prevent zombie processes.
        """
        try:
            # Record start time
            self.training_start_time = time.time()
            
            # Print initial memory status
            if torch.cuda.is_available():
                allocated, reserved = get_gpu_memory_usage()
                print(f"[MEMORY] Initial GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            print("🧼 Starting training with GPU memory management enabled")
            
            # Train and validate
            self.train_and_validate()
            
            # Print final memory statistics
            if self.is_main_process:
                print(f"\n[MEMORY] Training completed with {self.oom_errors} out-of-memory errors and {self.memory_warnings} memory warnings")
                if torch.cuda.is_available():
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"[MEMORY] Final GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            print("✅ Training pipeline completed successfully")
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {e}")
            traceback.print_exc()
            # Cleanup on error
            cleanup_gpu_memory()
            raise
            
        finally:
            try:
                # Finalize experiment on WandB
                if self.config.use_wandb and self.is_main_process:
                    wandb.finish()
                
                # Clean up distributed training resources
                if self.distributed and dist.is_initialized():
                    dist.destroy_process_group()
                    
                # Final GPU cleanup
                cleanup_gpu_memory()
                print("🧼 Training cleanup completed")
                
            except Exception as cleanup_error:
                print(f"Warning during cleanup: {cleanup_error}")
        
        # Final GPU cache cleanup
        clear_gpu_cache()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multimodal Deepfake Detection Training")
    
    # Data parameters
    parser.add_argument('--json_path', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for runs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    
    # Model parameters
    parser.add_argument('--backbone_visual', type=str, default='efficientnet', choices=['efficientnet', 'swin'], help='Visual backbone architecture')
    parser.add_argument('--backbone_audio', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert'], help='Audio backbone architecture')
    parser.add_argument('--fusion_type', type=str, default='attention', choices=['attention', 'concat'], help='Fusion type for multimodal features')
    parser.add_argument('--video_feature_dim', type=int, default=1024, help='Dimension of video features')
    parser.add_argument('--audio_feature_dim', type=int, default=1024, help='Dimension of audio features')
    parser.add_argument('--transformer_dim', type=int, default=768, help='Dimension of transformer encoder')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer encoder layers')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes (real/fake)')
    parser.add_argument('--num_deepfake_types', type=int, default=7, help='Number of deepfake types for fine-grained classification')
    parser.add_argument('--enable_face_mesh', action='store_true', help='Enable face mesh analysis')
    parser.add_argument('--enable_explainability', action='store_true', help='Enable model explainability components')
    parser.add_argument('--use_spectrogram', action='store_true', help='Use audio spectrogram features')
    parser.add_argument('--detect_deepfake_type', action='store_true', help='Enable deepfake type detection')
    parser.add_argument('--deepfake_type_weight', type=float, default=0.3, help='Weight for deepfake type classification loss')
    parser.add_argument('--detect_faces', action='store_true', help='Enable face detection')
    parser.add_argument('--compute_spectrograms', action='store_true', help='Compute audio spectrograms')
    parser.add_argument('--temporal_features', action='store_true', help='Compute temporal consistency features')
    parser.add_argument('--enhanced_preprocessing', action='store_true', help='Enable enhanced preprocessing features (physiological, etc.)')
    parser.add_argument('--enhanced_augmentation', action='store_true', help='Enable enhanced data augmentation')
    parser.add_argument('--enable_skin_color_analysis', action='store_true', default=True, help='Enable skin color analysis (memory intensive, enabled by default)')
    parser.add_argument('--enable_advanced_physiological', action='store_true', help='Enable advanced physiological analysis (heartbeat, blood flow, breathing)')
    parser.add_argument('--physiological_fps', type=int, default=30, help='Frame rate for physiological signal analysis')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to use')
    parser.add_argument('--num_workers', type=int, default=0, help='🧼 SAFETY: Number of data loader workers (default=0, optimal for complex multimodal datasets)')  # Reverted based on benchmark results
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split ratio')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use class-weighted loss function')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'plateau', 'none'], help='Learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--gradient_clip', type=float, default=0.5, help='Gradient clipping value')  # Reduced for stability
    
    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    # Logging and visualization parameters
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='deepfake-detection', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging batch results')
    parser.add_argument('--visualization_interval', type=int, default=50, help='Interval for visualizing predictions')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate checkpoints')
    parser.add_argument('--save_intermediate_interval', type=int, default=500, help='Interval for saving intermediate checkpoints')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--amp_enabled', action='store_true', default=True, help='Enable automatic mixed precision (default: True)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model weights')
    
    # Performance optimization parameters - 🧼 SAFETY OVERRIDES
    parser.add_argument('--pin_memory', action='store_true', default=False, help='🧼 SAFETY: Pin memory for faster data loading (default=False to prevent memory leaks)')
    parser.add_argument('--persistent_workers', action='store_true', default=False, help='🧼 SAFETY: Keep workers alive between epochs (default=False to prevent hanging processes)')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of samples loaded in advance by each worker (reduced for safety)')
    parser.add_argument('--reduce_frames', type=int, default=8, help='Reduce number of frames per video for faster processing (default: 8, original: 16)')
    parser.add_argument('--disable_skin_analysis', action='store_true', help='Disable memory-intensive skin color analysis for speed')
    parser.add_argument('--disable_advanced_physio', action='store_true', help='Disable advanced physiological analysis for speed')
    parser.add_argument('--fast_mode', action='store_true', help='Enable fast mode with reduced feature extraction')
    
    return parser.parse_args()


def main():
    """
    Main entry point with comprehensive error handling and cleanup.
    🧼 PROCESS SAFETY: Includes proper cleanup to prevent zombie processes.
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Set local_rank from environment variable if running with torchrun
        if args.distributed and 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
        
        # Suppress warnings
        suppress_warnings()
        
        print("🧼 Starting training with process safety measures enabled")
        
        # Create trainer and run
        trainer = DeepfakeTrainer(args)
        trainer.run()
        
        print("✅ Training completed successfully")
        
    except KeyboardInterrupt:
        print("🧼 Training interrupted by user, cleaning up...")
        cleanup_processes()
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        traceback.print_exc()
        cleanup_processes()
        raise
        
    finally:
        # Final cleanup
        cleanup_processes()
        print("🧼 Process cleanup completed")


if __name__ == "__main__":
    main()
