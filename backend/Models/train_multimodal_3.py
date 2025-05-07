
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


def run_inference(model_path, video_path, output_dir="./inference_results", use_edge_optimized=False):
    """
    Run inference on a video file using a trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        video_path: Path to the video file for inference
        output_dir: Directory to save inference results
        use_edge_optimized: Whether to use the edge-optimized version of the model
    """
    import os
    import torch
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from multi_modal_model import MultiModalDeepfakeModel
    import time
    from torchvision import transforms
    import torchaudio
    from PIL import Image
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration from checkpoint
    config = checkpoint.get('config', {})
    
    # Create model with the same configuration
    model = MultiModalDeepfakeModel(
        num_classes=config.get('num_classes', 2),
        video_feature_dim=config.get('video_feature_dim', 1024),
        audio_feature_dim=config.get('audio_feature_dim', 1024),
        transformer_dim=config.get('transformer_dim', 768),
        num_transformer_layers=config.get('num_transformer_layers', 4),
        enable_face_mesh=config.get('enable_face_mesh', True),
        enable_explainability=config.get('enable_explainability', True),
        fusion_type=config.get('fusion_type', 'attention'),
        backbone_visual=config.get('backbone_visual', 'efficientnet'),
        backbone_audio=config.get('backbone_audio', 'wav2vec2'),
        use_spectrogram=config.get('use_spectrogram', True),
        detect_deepfake_type=config.get('detect_deepfake_type', False),
        num_deepfake_types=config.get('num_deepfake_types', 7),
        debug=False
    )
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Enable edge optimization if requested
    if use_edge_optimized:
        edge_model = model.enable_edge_optimization(
            quantize=True, 
            use_onnx=False, 
            adaptive_resolution=True
        )
        if edge_model is not None:
            print("Using edge-optimized model")
    
    # Video preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess video
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Calculate frames to sample (sample 32 frames evenly)
    max_frames = 32
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    
    # Extract frames
    video_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {idx}")
            continue
        
        # Convert to RGB and apply transforms
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        transformed = transform(pil_img)
        video_frames.append(transformed)
    
    cap.release()
    
    if not video_frames:
        print("Error: No valid frames extracted from video")
        return
    
    # Stack frames into tensor
    video_tensor = torch.stack(video_frames).unsqueeze(0)  # [1, num_frames, C, H, W]
    
    # Load audio
    audio_path = video_path.replace('.mp4', '.wav')
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        print("Extracting audio from video...")
        
        # Use ffmpeg to extract audio
        import subprocess
        cmd = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y'
        subprocess.call(cmd, shell=True)
        
        if not os.path.exists(audio_path):
            print("Error extracting audio. Generating zero audio.")
            audio_tensor = torch.zeros(16000)
        else:
            print(f"Audio extracted to: {audio_path}")
    
    try:
        # Load audio with torchaudio
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.squeeze(0).numpy()
        
        # Process audio (use center 5 seconds if longer)
        target_length = 16000  # 1 second at 16kHz
        if len(audio) > target_length:
            start = (len(audio) - target_length) // 2
            audio = audio[start:start + target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, audio_length]
        
        # Create spectrogram for additional features
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=128,
            hop_length=512,
            n_fft=2048
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = cv2.resize(mel_spec, (128, 128))
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        audio_spectrogram = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    except Exception as e:
        print(f"Error processing audio: {e}")
        audio_tensor = torch.zeros(1, 16000)
        audio_spectrogram = torch.zeros(1, 1, 128, 128)
    
    # Move inputs to device
    inputs = {
        'video_frames': video_tensor.to(device),
        'audio': audio_tensor.to(device),
        'audio_spectrogram': audio_spectrogram.to(device)
    }
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    
    with torch.no_grad():
        logits, outputs = model(inputs)
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    # Get prediction
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(logits, dim=1).item()
    confidence = probs[0, pred_class].item()
    
    prediction = "FAKE" if pred_class == 1 else "REAL"
    print(f"Prediction: {prediction} with confidence {confidence:.4f}")
    
    # Get deepfake type if available
    deepfake_type = "Unknown"
    if 'deepfake_type' in outputs and outputs['deepfake_type'] is not None:
        deepfake_types = [
            "Unknown", "Face Swap", "Face Reenactment", "Lip Sync", 
            "Audio Only", "Entire Synthesis", "Attribute Manipulation"
        ]
        type_idx = torch.argmax(outputs['deepfake_type'], dim=1).item()
        deepfake_type = deepfake_types[type_idx]
        print(f"Deepfake type: {deepfake_type}")
    
    # Get explanation if available
    if 'explanation' in outputs and outputs['explanation'] is not None:
        explanation = outputs['explanation']
        print("\nExplanation:")
        
        if 'issues_found' in explanation:
            issues = explanation['issues_found']
            print("Issues found:")
            for issue in issues:
                print(f"- {issue}")
        
        if 'detection_scores' in explanation:
            print("\nDetection scores:")
            for key, value in explanation['detection_scores'].items():
                print(f"- {key}: {value:.4f}")
    
    # Generate visualizations
    plt.figure(figsize=(16, 10))
    
    # 1. Plot sample frames with heatmap
    plt.subplot(2, 2, 1)
    sample_frame = video_tensor[0, 0].permute(1, 2, 0).cpu().numpy()
    sample_frame = (sample_frame * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    sample_frame = np.clip(sample_frame, 0, 1)
    plt.imshow(sample_frame)
    plt.title(f"Sample Frame: {prediction} ({confidence:.2f})")
    plt.axis('off')
    
    # 2. Plot physiological signal detections if available
    plt.subplot(2, 2, 2)
    physio_metrics = []
    physio_values = []
    if 'physiological_outputs' in outputs and outputs['physiological_outputs'] is not None:
        physio_outputs = outputs['physiological_outputs']
        for key in ['heart_rate', 'hrv_score', 'hr_consistency', 'skin_naturality', 'breathing_consistency']:
            if key in physio_outputs:
                value = physio_outputs[key].item() if hasattr(physio_outputs[key], 'item') else physio_outputs[key]
                physio_metrics.append(key.replace('_', ' ').title())
                physio_values.append(value)
    
    if physio_metrics:
        plt.barh(physio_metrics, physio_values, color='skyblue')
        plt.xlim(0, 1)
        plt.title("Physiological Signal Analysis")
        plt.xlabel("Score")
    else:
        plt.text(0.5, 0.5, "No physiological data available", ha='center', va='center')
        plt.title("Physiological Signal Analysis")
        plt.axis('off')
    
    # 3. Plot ocular behavior detections if available
    plt.subplot(2, 2, 3)
    ocular_metrics = []
    ocular_values = []
    if 'ocular_outputs' in outputs and outputs['ocular_outputs'] is not None:
        ocular_outputs = outputs['ocular_outputs']
        for key in ['blink_naturalness', 'dilation_consistency', 'saccade_score', 'micro_expression_score']:
            if key in ocular_outputs:
                value = ocular_outputs[key].item() if hasattr(ocular_outputs[key], 'item') else ocular_outputs[key]
                ocular_metrics.append(key.replace('_', ' ').title())
                ocular_values.append(value)
    
    if ocular_metrics:
        plt.barh(ocular_metrics, ocular_values, color='lightgreen')
        plt.xlim(0, 1)
        plt.title("Ocular Behavior Analysis")
        plt.xlabel("Score")
    else:
        plt.text(0.5, 0.5, "No ocular data available", ha='center', va='center')
        plt.title("Ocular Behavior Analysis")
        plt.axis('off')
    
    # 4. Plot lip-audio sync detections if available
    plt.subplot(2, 2, 4)
    sync_metrics = []
    sync_values = []
    if 'lip_sync_outputs' in outputs and outputs['lip_sync_outputs'] is not None:
        sync_outputs = outputs['lip_sync_outputs']
        for key in ['sync_score', 'speech_consistency', 'phoneme_match', 'movement_naturalness']:
            if key in sync_outputs:
                value = sync_outputs[key].item() if hasattr(sync_outputs[key], 'item') else sync_outputs[key]
                sync_metrics.append(key.replace('_', ' ').title())
                sync_values.append(value)
    
    if sync_metrics:
        plt.barh(sync_metrics, sync_values, color='salmon')
        plt.xlim(0, 1)
        plt.title("Lip-Audio Sync Analysis")
        plt.xlabel("Score")
    else:
        plt.text(0.5, 0.5, "No lip-audio sync data available", ha='center', va='center')
        plt.title("Lip-Audio Sync Analysis")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_results.png"))
    
    # Save detailed results to a text file
    with open(os.path.join(output_dir, "detection_report.txt"), 'w') as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Prediction: {prediction}\n")
        f.write(f"Confidence: {confidence:.4f}\n")
        f.write(f"Deepfake Type: {deepfake_type}\n\n")
        
        if 'explanation' in outputs and outputs['explanation'] is not None:
            explanation = outputs['explanation']
            
            if 'issues_found' in explanation:
                f.write("Issues Found:\n")
                for issue in explanation['issues_found']:
                    f.write(f"- {issue}\n")
                f.write("\n")
            
            if 'detection_scores' in explanation:
                f.write("Detection Scores:\n")
                for key, value in sorted(explanation['detection_scores'].items()):
                    f.write(f"- {key}: {value:.4f}\n")
                f.write("\n")
        
        # Write physiological signal details
        if 'physiological_outputs' in outputs and outputs['physiological_outputs'] is not None:
            f.write("Physiological Signal Analysis:\n")
            for key, value in outputs['physiological_outputs'].items():
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.cpu().numpy()
                f.write(f"- {key}: {value}\n")
            f.write("\n")
        
        # Write ocular behavior details
        if 'ocular_outputs' in outputs and outputs['ocular_outputs'] is not None:
            f.write("Ocular Behavior Analysis:\n")
            for key, value in outputs['ocular_outputs'].items():
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.cpu().numpy()
                f.write(f"- {key}: {value}\n")
            f.write("\n")
        
        # Write lip-audio sync details
        if 'lip_sync_outputs' in outputs and outputs['lip_sync_outputs'] is not None:
            f.write("Lip-Audio Sync Analysis:\n")
            for key, value in outputs['lip_sync_outputs'].items():
                if isinstance(value, torch.Tensor) and key != 'attention_weights':
                    value = value.item() if value.numel() == 1 else value.cpu().numpy()
                    f.write(f"- {key}: {value}\n")
            f.write("\n")
        
        f.write(f"Inference Time: {inference_time:.2f} seconds\n")
    
    print(f"Results saved to {output_dir}")
    return prediction, confidence, outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal Deepfake Detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    train_parser.add_argument("--json_path", type=str, required=True, help="Path to JSON metadata file")
    train_parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for training artifacts")
    train_parser.add_argument("--checkpoint_dir", type=str, default="./saved_models", help="Directory to save model checkpoints")
    train_parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    train_parser.add_argument("--learning_rate", type=float, default=0.0001, help="Initial learning rate")
    train_parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer")
    train_parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio")
    train_parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    train_parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    train_parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    train_parser.add_argument("--model_type", type=str, default="efficientnet", choices=["efficientnet", "swin"], help="Visual backbone type")
    train_parser.add_argument("--audio_type", type=str, default="wav2vec2", choices=["wav2vec2", "hubert"], help="Audio backbone type")
    train_parser.add_argument("--detect_faces", action="store_true", help="Detect and extract face regions")
    train_parser.add_argument("--compute_spectrograms", action="store_true", help="Compute audio spectrograms")
    train_parser.add_argument("--enable_face_mesh", action="store_true", help="Enable face mesh analysis")
    train_parser.add_argument("--enable_explainability", action="store_true", help="Enable model explainability")
    train_parser.add_argument("--detect_deepfake_type", action="store_true", help="Detect specific deepfake type")
    train_parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for tracking")
    train_parser.add_argument("--wandb_project", type=str, default="deepfake-detection", help="Weights & Biases project name")
    train_parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (cuda, cpu)")
    train_parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    train_parser.add_argument("--amp_enabled", action="store_true", help="Use automatic mixed precision")
    train_parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference on a video")
    inference_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    inference_parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    inference_parser.add_argument("--output_dir", type=str, default="./inference_results", help="Directory to save inference results")
    inference_parser.add_argument("--use_edge_optimized", action="store_true", help="Use edge-optimized model")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on a dataset")
    eval_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    eval_parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    eval_parser.add_argument("--json_path", type=str, required=True, help="Path to JSON metadata file")
    eval_parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    eval_parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    eval_parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    eval_parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    eval_parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation (cuda, cpu)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX format")
    export_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    export_parser.add_argument("--output_path", type=str, default="./deepfake_detector.onnx", help="Path to save ONNX model")
    export_parser.add_argument("--quantize", action="store_true", help="Quantize model before export")
    export_parser.add_argument("--batch_size", type=int, default=1, help="Batch size for export")
    
    args = parser.parse_args()
    
    if args.command == "train":
        # Convert arguments to AttributeDict for compatibility with existing code
        config = argparse.Namespace(
            data_dir=args.data_dir,
            json_path=args.json_path,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_split=args.validation_split,
            test_split=args.test_split,
            seed=args.seed,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            backbone_visual=args.model_type,
            backbone_audio=args.audio_type,
            detect_faces=args.detect_faces,
            compute_spectrograms=args.compute_spectrograms,
            enable_face_mesh=args.enable_face_mesh,
            enable_explainability=args.enable_explainability,
            detect_deepfake_type=args.detect_deepfake_type,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            device=args.device,
            distributed=args.distributed,
            amp_enabled=args.amp_enabled,
            local_rank=args.local_rank,
            resume=args.resume,
            # Add other default parameters
            num_classes=2,
            video_feature_dim=1024,
            audio_feature_dim=1024,
            transformer_dim=768,
            num_transformer_layers=4,
            fusion_type="attention",
            use_spectrogram=True,
            num_deepfake_types=7,
            temporal_features=True,
            debug=False
        )
        
        # Initialize trainer
        trainer = DeepfakeTrainer(config)
        
        # Train the model
        trainer.train()
        
    elif args.command == "inference":
        # Run inference
        run_inference(
            model_path=args.model_path,
            video_path=args.video_path,
            output_dir=args.output_dir,
            use_edge_optimized=args.use_edge_optimized
        )
        
    elif args.command == "evaluate":
        # Load model
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Extract model configuration from checkpoint
        config = checkpoint.get('config', {})
        
        # Create model with the same configuration
        model = MultiModalDeepfakeModel(
            num_classes=config.get('num_classes', 2),
            video_feature_dim=config.get('video_feature_dim', 1024),
            audio_feature_dim=config.get('audio_feature_dim', 1024),
            transformer_dim=config.get('transformer_dim', 768),
            num_transformer_layers=config.get('num_transformer_layers', 4),
            enable_face_mesh=config.get('enable_face_mesh', True),
            enable_explainability=config.get('enable_explainability', True),
            fusion_type=config.get('fusion_type', 'attention'),
            backbone_visual=config.get('backbone_visual', 'efficientnet'),
            backbone_audio=config.get('backbone_audio', 'wav2vec2'),
            use_spectrogram=config.get('use_spectrogram', True),
            detect_deepfake_type=config.get('detect_deepfake_type', False),
            num_deepfake_types=config.get('num_deepfake_types', 7),
            debug=False
        )
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        
        # Get data loaders
        _, _, test_loader, _ = get_data_loaders(
            json_path=args.json_path,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            validation_split=0.0,  # Only use test set
            test_split=1.0,
            shuffle=False,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            detect_faces=True,
            compute_spectrograms=True,
            temporal_features=True
        )
        
        # Evaluate the model
        print(f"Evaluating model on {len(test_loader)} batches...")
        
        # Initialize metrics
        all_labels = []
        all_preds = []
        all_probs = []
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Move inputs to device
                inputs = move_batch_to_device(batch, device)
                
                # Get model outputs
                outputs, additional_outputs = model(inputs)
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                # Store labels and predictions
                all_labels.extend(inputs['label'].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of being fake
                
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds, all_probs, epoch=0, return_dict=True)
        
        # Plot confusion matrix
        cm_path = plot_confusion_matrix(all_labels, all_preds, epoch=0, save_dir=args.output_dir)
        
        # Save metrics to JSON
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Evaluation complete. Results saved to {args.output_dir}")
        
    elif args.command == "export":
        # Load model
        device = torch.device("cpu")  # Always use CPU for export
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Extract model configuration from checkpoint
        config = checkpoint.get('config', {})
        
        # Create model with the same configuration
        model = MultiModalDeepfakeModel(
            num_classes=config.get('num_classes', 2),
            video_feature_dim=config.get('video_feature_dim', 1024),
            audio_feature_dim=config.get('audio_feature_dim', 1024),
            transformer_dim=config.get('transformer_dim', 768),
            num_transformer_layers=config.get('num_transformer_layers', 4),
            enable_face_mesh=config.get('enable_face_mesh', True),
            enable_explainability=config.get('enable_explainability', True),
            fusion_type=config.get('fusion_type', 'attention'),
            backbone_visual=config.get('backbone_visual', 'efficientnet'),
            backbone_audio=config.get('backbone_audio', 'wav2vec2'),
            use_spectrogram=config.get('use_spectrogram', True),
            detect_deepfake_type=config.get('detect_deepfake_type', False),
            num_deepfake_types=config.get('num_deepfake_types', 7),
            debug=False
        )
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.eval()
        
        # Quantize if requested
        if args.quantize:
            try:
                import torch.quantization
                
                # Create a copy of the model for quantization
                quantized_model = copy.deepcopy(model)
                
                # Prepare model for quantization
                quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(quantized_model, inplace=True)
                
                # Convert to quantized model
                torch.quantization.convert(quantized_model, inplace=True)
                
                print("Model successfully quantized")
                model = quantized_model
            except Exception as e:
                print(f"Error quantizing model: {e}")
        
        # Create dummy input for tracing
        dummy_input = {
            'video_frames': torch.randn(args.batch_size, 32, 3, 224, 224),
            'audio': torch.randn(args.batch_size, 16000),
            'audio_spectrogram': torch.randn(args.batch_size, 1, 128, 128)
        }
        
        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                (dummy_input,),
                args.output_path,
                input_names=['video_frames', 'audio', 'audio_spectrogram'],
                output_names=['logits', 'outputs'],
                dynamic_axes={
                    'video_frames': {0: 'batch_size', 1: 'seq_length'},
                    'audio': {0: 'batch_size', 1: 'audio_length'},
                    'audio_spectrogram': {0: 'batch_size'},
                    'logits': {0: 'batch_size'},
                    'outputs': {0: 'batch_size'}
                },
                opset_version=12
            )
            
            print(f"Model successfully exported to {args.output_path}")
        except Exception as e:
            print(f"Error exporting model to ONNX: {e}")
    
    else:
        parser.print_help()
