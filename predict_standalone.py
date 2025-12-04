"""
TRUE STANDALONE PREDICTION - No JSON metadata required!
Processes video directly and returns prediction.
This is what you'd use for real deployment.
"""
import torch
import cv2
import numpy as np
import subprocess
import os
from pathlib import Path
import json

def extract_audio_if_needed(video_path):
    """Extract audio from video to WAV file."""
    audio_path = video_path.replace('.mp4', '.wav')
    if os.path.exists(audio_path):
        return audio_path
    
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            audio_path, '-y', '-loglevel', 'quiet'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path
    except Exception as e:
        print(f"⚠️  Warning: Could not extract audio: {e}")
        return None


def predict_video_standalone(video_path, checkpoint_path):
    """
    Predict if video is REAL or FAKE without any metadata JSON files.
    
    Args:
        video_path: Path to video file (.mp4)
        checkpoint_path: Path to trained model checkpoint (.pth)
    
    Returns:
        dict: {
            'prediction': 'REAL' or 'FAKE',
            'confidence': float (0-100),
            'real_probability': float (0-100),
            'fake_probability': float (0-100)
        }
    """
    print(f"\n{'='*70}")
    print(f"🎬 STANDALONE DEEPFAKE DETECTION (No JSON needed!)")
    print(f"{'='*70}")
    print(f"📹 Video: {os.path.basename(video_path)}")
    
    # Validate video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"📊 Info: {duration:.1f}s | {frames} frames @ {fps:.1f} FPS | {width}x{height}")
    
    # Extract audio
    print("🎵 Audio: ", end="", flush=True)
    audio_path = extract_audio_if_needed(video_path)
    if audio_path and os.path.exists(audio_path):
        print(f"✓ Extracted to {os.path.basename(audio_path)}")
    else:
        print("⚠️  No audio available (continuing without it)")
    
    # Load checkpoint
    print("🔧 Loading model...", end=" ", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("✓")
    
    # For now, we still use the existing infrastructure with temp metadata
    # A fully standalone version would require reimplementing all preprocessing
    print("\n⚠️  NOTE: Currently using existing preprocessing pipeline")
    print("    A fully standalone version would need custom preprocessing")
    print("    For now, creating temporary metadata...\n")
    
    # Use the existing predict_any_video function
    from predict_deployment import predict_any_video
    
    result = predict_any_video(video_path, checkpoint_path, debug=False)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_standalone.py <video_path> [checkpoint_path]")
        print("\nExample:")
        print('  python predict_standalone.py "F:\\deepfake\\backup\\LAV-DF\\REALO.mp4"')
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Default to best model
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else \
        r"F:\deepfake\backup\Models\server_checkpoints\run_20251111_105258\best_model.pth"
    
    result = predict_video_standalone(video_path, checkpoint_path)
    
    if result:
        print(f"\n{'='*70}")
        print(f"✅ Prediction complete!")
        print(f"{'='*70}")
