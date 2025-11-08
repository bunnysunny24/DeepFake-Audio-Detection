"""
DEPLOYMENT-READY: Test ANY video without needing metadata.
This bypasses dataset validation by adding video to temporary metadata.
Automatically extracts audio if needed.
"""
import os
import sys
import torch
import json
import cv2
import subprocess
from train_multimodal import DeepfakeTrainer

def predict_any_video(video_path, checkpoint_path="server_checkpoints/best_model.pth"):
    """
    Predict if ANY video is fake or real - no metadata needed!
    This is what you'd use for DEPLOYMENT.
    Automatically extracts audio if .wav file doesn't exist.
    """
    
    video_path = os.path.abspath(video_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    
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
        args.test_json = temp_metadata_path  # Also set test_json so trainer finds it
        args.data_dir = os.path.dirname(video_path)
        args.num_epochs = 0
        args.use_wandb = False
        args.max_samples = 1
        
        # Create trainer (this loads the video)
        trainer = DeepfakeTrainer(args)
        
        # CRITICAL FIX: Override class weights to be balanced (1.0, 1.0)
        # The trainer recalculates them as sqrt_balanced which biases toward REAL
        trainer.class_weights = torch.tensor([1.0, 1.0], device=trainer.device)
        print(f"✓ (Class weights fixed: {trainer.class_weights.cpu().tolist()})")
        
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
        
        with torch.no_grad():
            output, _ = trainer.model(inputs)
            probs = torch.softmax(output, dim=-1)
            
            r_prob = float(probs[0, 0])
            f_prob = float(probs[0, 1])
            pred = "FAKE" if f_prob > r_prob else "REAL"
            conf = max(r_prob, f_prob) * 100
        
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
        print("\nUsage: python predict_deployment.py <video_file>")
        print("\nExample:")
        print('  python predict_deployment.py "C:/suspect_video.mp4"')
        print('  python predict_deployment.py "F:/deepfake/backup/LAV-DF/REALO.mp4"')
        sys.exit(1)
    
    predict_any_video(sys.argv[1])
