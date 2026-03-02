"""
DEPLOYMENT-READY: Test ANY video without needing metadata.
This bypasses dataset validation by adding video to temporary metadata.
Automatically extracts audio if needed.
DEBUG MODE: Use --debug flag to save detailed artifacts for analysis.
"""
import os
import sys
import torch
import json
import cv2
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train_multimodal import DeepfakeTrainer

def predict_any_video(video_path, checkpoint_path=r"F:\deepfake\backup\Models\server_checkpoints\run_20251111_105258\best_model.pth", debug=False):
    """
    Predict if ANY video is fake or real - no metadata needed!
    This is what you'd use for DEPLOYMENT.
    Automatically extracts audio if .wav file doesn't exist.
    
    Args:
        video_path: Path to video file
        checkpoint_path: Path to BEST model checkpoint (Epoch 13: 79.2% val acc, trained with fixed class weights [1.4528, 0.8095])
        debug: If True, saves detailed debug artifacts (face crops, tensors, spectrograms, logits)
    
    NOTE: Using BEST MODEL from epoch 13 (79.2% validation accuracy)
          Trained with FIXED class weights [1.4528, 0.8095] to handle 3.22:1 class imbalance
    """
    
    video_path = os.path.abspath(video_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Setup debug directory if debug mode enabled
    debug_dir = None
    if debug:
        video_name = Path(video_path).stem
        debug_dir = Path(r"F:\deepfake\backup\Models\debug") / video_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "crops").mkdir(exist_ok=True)
        print(f"\n🔍 DEBUG MODE ENABLED - Artifacts will be saved to: {debug_dir}")
    
    print(f"\n{'='*70}")
    print(f"🎬 DEEPFAKE DETECTION")
    print(f"{'='*70}")
    print(f"Video: {os.path.basename(video_path)}")
    
    # Check video exists and get info
    if not os.path.exists(video_path):
        print("❌ Video not found!")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video!")
        return None
    
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / fps if fps > 0 else 0
    cap.release()
    
    print(f"Duration: {duration:.1f}s, Frames: {frames}, FPS: {fps:.1f}")
    
    # Check if audio file exists, if not extract it
    audio_path = video_path.replace('.mp4', '.wav')
    if not os.path.exists(audio_path):
        print("Extracting audio...", end=" ", flush=True)
        try:
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                audio_path,
                '-y'  # Overwrite if exists
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓")
            else:
                print("⚠️  Warning: Could not extract audio, continuing anyway...")
        except FileNotFoundError:
            print("⚠️  Warning: ffmpeg not found. Install ffmpeg or manually extract audio.")
            print(f"   Run: ffmpeg -i \"{video_path}\" -vn -acodec pcm_s16le -ar 16000 -ac 1 \"{audio_path}\" -y")
    else:
        print("Audio file found ✓")
    
    # Load model config
    print("Loading model...", end=" ", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config_dict = checkpoint.get('config', {})
    
    if not config_dict:
        config_path = r"F:\deepfake\backup\Models\server_outputs\run_20251104_094138\config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    
    # TRICK: Add this video to temporary metadata so dataset loader accepts it
    temp_json = 'temp_deploy.json'
    temp_metadata_path = os.path.join(os.path.dirname(video_path), 'temp_metadata.json')
    
    # Create temporary metadata with just this video
    with open(temp_metadata_path, 'w') as f:
        json.dump([{
            'file': os.path.basename(video_path),
            'n_fakes': 0,  # Placeholder - we don't know yet!
            'duration': duration,
            'split': 'test',
            'video_frames': frames,
            'audio_channels': 1,
            'audio_frames': 0,
            'fake_periods': [],
            'modify_audio': False,
            'modify_video': False,
            'original': None,
            'timestamps': [],
            'transcript': ''
        }], f)
    
    try:
        # Create args
        class Args:
            pass
        args = Args()
        for key, value in config_dict.items():
            setattr(args, key, value)
        
        # Point to temporary metadata
        args.json_path = temp_metadata_path
        args.data_dir = os.path.dirname(video_path)
        args.num_epochs = 0
        args.use_wandb = False
        args.max_samples = 1
        
        # Create trainer (this loads the video)
        trainer = DeepfakeTrainer(args)
        
        if len(trainer.test_loader.dataset) == 0:
            print("\n❌ Failed! Possible reasons:")
            print("   - No face detected")
            print("   - Video corrupted")
            print("   - Unsupported format")
            return None
        
        # Load model weights
        trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        trainer.model.to(trainer.device)
        trainer.model.eval()
        print("✓")
        
        # Predict
        print("Analyzing video...", end=" ", flush=True)
        sample = trainer.test_loader.dataset[0]
        inputs = {k: v.unsqueeze(0).to(trainer.device) for k, v in sample.items() if isinstance(v, torch.Tensor)}
        
        # DEBUG: Save input tensors and visualizations
        if debug:
            print(f"\n🔍 Saving debug artifacts to {debug_dir}...")
            
            # Save raw input tensors
            torch.save(inputs, debug_dir / "input_tensors.pt")
            print(f"   ✓ Saved input tensors")
            
            # Save video frames as images
            if 'video' in inputs:
                video_tensor = inputs['video'][0].cpu()  # Shape: [C, T, H, W] or [T, C, H, W]
                num_frames = video_tensor.shape[1] if video_tensor.shape[0] == 3 else video_tensor.shape[0]
                
                # Create frame montage
                fig, axes = plt.subplots(2, 8, figsize=(20, 6))
                axes = axes.flatten()
                
                for i in range(min(16, num_frames)):
                    if video_tensor.shape[0] == 3:  # [C, T, H, W]
                        frame = video_tensor[:, i, :, :].permute(1, 2, 0).numpy()
                    else:  # [T, C, H, W]
                        frame = video_tensor[i, :, :, :].permute(1, 2, 0).numpy()
                    
                    # Denormalize
                    frame = (frame * 0.5) + 0.5  # Assuming normalization with mean=0.5, std=0.5
                    frame = np.clip(frame, 0, 1)
                    
                    axes[i].imshow(frame)
                    axes[i].axis('off')
                    axes[i].set_title(f'Frame {i}')
                
                plt.tight_layout()
                plt.savefig(debug_dir / "video_frames.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   ✓ Saved video frames montage ({num_frames} frames)")
            
            # Save face crops if available
            if 'face_crops' in sample:
                face_crops = sample['face_crops']
                if isinstance(face_crops, torch.Tensor):
                    num_crops = face_crops.shape[0] if len(face_crops.shape) == 4 else 1
                    
                    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
                    axes = axes.flatten()
                    
                    for i in range(min(16, num_crops)):
                        if len(face_crops.shape) == 4:
                            crop = face_crops[i].permute(1, 2, 0).cpu().numpy()
                        else:
                            crop = face_crops.permute(1, 2, 0).cpu().numpy()
                        
                        crop = (crop * 0.5) + 0.5
                        crop = np.clip(crop, 0, 1)
                        
                        axes[i].imshow(crop)
                        axes[i].axis('off')
                        axes[i].set_title(f'Crop {i}')
                        
                        # Save individual crop
                        cv2.imwrite(str(debug_dir / "crops" / f"crop_{i:02d}.jpg"), 
                                   (crop * 255).astype(np.uint8)[:, :, ::-1])
                    
                    plt.tight_layout()
                    plt.savefig(debug_dir / "face_crops_montage.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"   ✓ Saved {num_crops} face crops")
            
            # Save audio spectrogram
            if 'audio' in inputs:
                audio_tensor = inputs['audio'][0].cpu().numpy()
                
                fig, ax = plt.subplots(figsize=(12, 4))
                
                # Handle different audio tensor shapes
                if len(audio_tensor.shape) == 3:  # [C, H, W] - spectrogram
                    spec = audio_tensor[0]
                    im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Audio Spectrogram')
                    plt.colorbar(im, ax=ax)
                elif len(audio_tensor.shape) == 2:  # [H, W] - spectrogram
                    im = ax.imshow(audio_tensor, aspect='auto', origin='lower', cmap='viridis')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Audio Spectrogram')
                    plt.colorbar(im, ax=ax)
                elif len(audio_tensor.shape) == 1:  # [T] - raw waveform
                    ax.plot(audio_tensor)
                    ax.set_xlabel('Sample')
                    ax.set_ylabel('Amplitude')
                    ax.set_title(f'Audio Waveform ({len(audio_tensor)} samples)')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Unexpected audio shape: {audio_tensor.shape}', 
                           ha='center', va='center', transform=ax.transAxes)
                
                plt.tight_layout()
                plt.savefig(debug_dir / "audio_spectrogram.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   ✓ Saved audio visualization (shape: {audio_tensor.shape})")
            
            # Save input statistics
            with open(debug_dir / "input_stats.txt", 'w') as f:
                f.write("=== INPUT TENSOR STATISTICS ===\n\n")
                for key, tensor in inputs.items():
                    f.write(f"{key}:\n")
                    f.write(f"  Shape: {tensor.shape}\n")
                    f.write(f"  Dtype: {tensor.dtype}\n")
                    f.write(f"  Device: {tensor.device}\n")
                    
                    # Check if tensor is not empty
                    if tensor.numel() > 0:
                        f.write(f"  Min: {tensor.min().item():.4f}\n")
                        f.write(f"  Max: {tensor.max().item():.4f}\n")
                        
                        # Only compute mean/std for float tensors
                        if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
                            f.write(f"  Mean: {tensor.mean().item():.4f}\n")
                            f.write(f"  Std: {tensor.std().item():.4f}\n")
                        else:
                            f.write(f"  Mean: (N/A - integer type)\n")
                            f.write(f"  Std: (N/A - integer type)\n")
                    else:
                        f.write(f"  (EMPTY TENSOR - no statistics)\n")
                    f.write("\n")
            print(f"   ✓ Saved input statistics")
        
        with torch.no_grad():
            output, features = trainer.model(inputs)
            probs = torch.softmax(output, dim=-1)
            logits = output[0].cpu().numpy()
            
            r_prob = float(probs[0, 0])
            f_prob = float(probs[0, 1])
            pred = "FAKE" if f_prob > r_prob else "REAL"
            conf = max(r_prob, f_prob) * 100
        
        # DEBUG: Save model outputs
        if debug:
            with open(debug_dir / "model_output.txt", 'w') as f:
                f.write("=== MODEL OUTPUT ===\n\n")
                f.write(f"Logits: {logits}\n")
                f.write(f"Probabilities: [REAL: {r_prob:.6f}, FAKE: {f_prob:.6f}]\n")
                f.write(f"Prediction: {pred}\n")
                f.write(f"Confidence: {conf:.2f}%\n\n")
                
                if features:
                    f.write("=== INTERMEDIATE FEATURES ===\n\n")
                    for key, feat in features.items():
                        if isinstance(feat, torch.Tensor):
                            f.write(f"{key}:\n")
                            f.write(f"  Shape: {feat.shape}\n")
                            f.write(f"  Mean: {feat.mean().item():.4f}\n")
                            f.write(f"  Std: {feat.std().item():.4f}\n\n")
            
            print(f"   ✓ Saved model outputs and logits")
            print(f"\n📁 All debug artifacts saved to: {debug_dir}")
        
        print("✓")
        
        # Show result
        print(f"\n{'='*70}")
        if pred == "FAKE":
            print(f"🚨 RESULT: FAKE ({conf:.1f}% confidence)")
        else:
            print(f"✅ RESULT: REAL ({conf:.1f}% confidence)")
        print(f"{'='*70}")
        print(f"REAL: {r_prob*100:.1f}%  |  FAKE: {f_prob*100:.1f}%")
        print(f"{'='*70}")
        
        # Interpretation
        if conf >= 80:
            print("✓ HIGH CONFIDENCE - Trust this result")
        elif conf >= 65:
            print("⚠️  MODERATE CONFIDENCE - Reasonable but verify")
        else:
            print("⚠️  LOW CONFIDENCE - Model uncertain (close to 50/50)")
        print(f"{'='*70}\n")
        
        return {
            'prediction': pred,
            'confidence': conf,
            'real_prob': r_prob * 100,
            'fake_prob': f_prob * 100
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup temp files
        if os.path.exists(temp_metadata_path):
            os.remove(temp_metadata_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python predict_deployment.py <video_file> [--debug]")
        print("\nExample:")
        print('  python predict_deployment.py "C:/suspect_video.mp4"')
        print('  python predict_deployment.py "F:/deepfake/backup/LAV-DF/REALO.mp4" --debug')
        print("\nOptions:")
        print("  --debug    Save detailed debug artifacts (face crops, tensors, spectrograms)")
        sys.exit(1)
    
    debug_mode = '--debug' in sys.argv
    predict_any_video(sys.argv[1], debug=debug_mode)