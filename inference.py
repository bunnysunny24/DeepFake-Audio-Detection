"""
🎭 Real-Time Deepfake Detection Inference Script

DEPLOYMENT MODE - Perfect for Real-Life Testing:
- Uses 27 deployment components (contrastive learning disabled)
- Single video input (no original/real video needed)
- Model uses learned weights from training phase
- Works on ANY video: uploaded files, webcam, live streams, real-life scenarios

TRAINING vs DEPLOYMENT:
  TRAINING (31 components):
    - 27 always-active components extract features from fake video
    - 27 always-active components extract features from original video
    - 4 contrastive learning components compare fake vs original
    - Model learns: "Fakes have X patterns, reals have Y patterns"
    - Output: Trained weights that recognize deepfake patterns
  
  DEPLOYMENT (27 components - THIS SCRIPT):
    - 27 always-active components extract features from SINGLE video
    - Contrastive learning disabled (no original needed)
    - Model applies learned weights to classify video
    - Output: "This video matches FAKE patterns" → FAKE (88%)
    
Supported Inputs:
- Video files (mp4, avi, mov, mkv, webm, flv)
- Real-time webcam feed
- Live streams
- Any single video from real-world scenarios
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import librosa
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import time
from datetime import datetime

# Import model
from multi_modal_model import MultiModalDeepfakeDetector

warnings.filterwarnings('ignore')


class DeepfakeDetector:
    """Production-ready deepfake detector for single video inference"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        quantized: bool = False,
        debug: bool = False
    ):
        """
        Initialize deepfake detector
        
        Args:
            checkpoint_path: Path to trained model checkpoint (.pth file)
            device: 'cuda' or 'cpu'
            quantized: Use INT8 quantized model for faster inference
            debug: Print detailed debug information
        """
        self.device = device
        self.debug = debug
        self.quantized = quantized
        
        print(f"[INFO] Initializing DeepfakeDetector on {device}")
        print(f"[INFO] Quantized: {quantized}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Video processing parameters
        self.target_frames = 16
        self.img_size = 224
        self.fps = 30
        self.sample_rate = 16000
        self.audio_duration = 3.0  # 3 seconds of audio
        
        print(f"[INFO] ✅ Model loaded successfully")
        print(f"[INFO] Target frames: {self.target_frames}, Image size: {self.img_size}x{self.img_size}")
        print(f"[INFO] Audio: {self.sample_rate}Hz, {self.audio_duration}s duration")
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load trained model from checkpoint for DEPLOYMENT
        
        Key Points:
        1. Model was trained with 31 components (27 + 4 contrastive learning)
        2. During training, contrastive learning learned difference patterns
        3. Trained weights capture "fake vs real" patterns in 27 components
        4. At inference, contrastive learning is disabled (no original video)
        5. Model uses learned weights to classify single videos
        
        Result: Perfect standalone model for all real-life testing
        """
        
        # Initialize model architecture (27 deployment components)
        # Contrastive learning components exist but won't be used (original_video_frames=None)
        model = MultiModalDeepfakeDetector(
            num_classes=2,
            video_feature_dim=1280,
            audio_feature_dim=768,
            transformer_dim=768,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
            fusion_type='attention',
            enable_advanced_physiological=True,
            quantization_backend='fbgemm' if self.quantized else None,
            use_qat=self.quantized,
            debug=self.debug
        )
        
        # Load checkpoint
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            accuracy = checkpoint.get('val_accuracy', 'unknown')
            print(f"[INFO] Loaded model from epoch {epoch}, val_accuracy: {accuracy}")
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        
        # Apply quantization if requested
        if self.quantized:
            print("[INFO] Converting model to INT8 quantized...")
            model.eval()
            torch.quantization.convert(model, inplace=True)
            print("[INFO] ✅ Model quantized to INT8")
        
        return model
    
    def preprocess_video(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess video file: Extract frames and audio
        
        Args:
            video_path: Path to video file
            
        Returns:
            video_tensor: [1, num_frames, 3, 224, 224]
            audio_tensor: [1, audio_samples]
        """
        if self.debug:
            print(f"[DEBUG] Processing video: {video_path}")
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if self.debug:
            print(f"[DEBUG] Total frames: {total_frames}, FPS: {video_fps}")
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, self.target_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"[WARNING] Could not read frame {idx}, using last valid frame")
                if frames:
                    frames.append(frames[-1])
                continue
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to 224x224
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        # Pad if not enough frames
        while len(frames) < self.target_frames:
            frames.append(frames[-1])
        
        # Convert to tensor: [num_frames, H, W, 3] -> [num_frames, 3, H, W]
        video_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)
        video_tensor = video_tensor.unsqueeze(0)  # [1, num_frames, 3, H, W]
        
        # Extract audio
        try:
            audio, sr = librosa.load(video_path, sr=self.sample_rate, mono=True, duration=self.audio_duration)
            
            # Pad or trim to target length
            target_length = int(self.sample_rate * self.audio_duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
            
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, audio_samples]
            
        except Exception as e:
            print(f"[WARNING] Could not extract audio: {e}")
            print("[WARNING] Using silent audio")
            target_length = int(self.sample_rate * self.audio_duration)
            audio_tensor = torch.zeros(1, target_length)
        
        if self.debug:
            print(f"[DEBUG] Video tensor shape: {video_tensor.shape}")
            print(f"[DEBUG] Audio tensor shape: {audio_tensor.shape}")
        
        return video_tensor, audio_tensor
    
    def preprocess_webcam_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess single webcam frame
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            frame_tensor: [1, 3, 224, 224]
        """
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        # To tensor: [H, W, 3] -> [3, H, W] -> [1, 3, H, W]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor
    
    def detect_from_video_file(self, video_path: str) -> Dict:
        """
        Detect deepfake from uploaded video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            results: {
                'prediction': 'FAKE' or 'REAL',
                'confidence': float (0-100),
                'fake_probability': float (0-1),
                'real_probability': float (0-1),
                'processing_time': float (seconds),
                'component_contributions': dict (optional if debug=True)
            }
        """
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🎥 ANALYZING VIDEO: {Path(video_path).name}")
        print(f"{'='*80}")
        
        # Preprocess
        video_tensor, audio_tensor = self.preprocess_video(video_path)
        video_tensor = video_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)
        
        # DEPLOYMENT MODE INFERENCE - Perfect for Real-Life Videos
        # 
        # How it works:
        # 1. Extract features from SINGLE video (27 components)
        # 2. No original video needed (contrastive learning disabled)
        # 3. Model uses learned weights from training phase
        # 4. Learned weights know: "Fakes have unnatural rPPG, GAN artifacts, synthetic voice, etc."
        # 5. Classification: "This video matches fake patterns" → FAKE (88% confidence)
        #
        # This is the PERFECT BASE MODEL for all testing scenarios:
        # - Uploaded videos from users
        # - Webcam feeds
        # - Live streams
        # - Real-world social media videos
        # - Compressed/degraded videos (model trained for robustness)
        with torch.no_grad():
            outputs = self.model(
                video_frames=video_tensor,
                audio_waveform=audio_tensor,
                original_video_frames=None,  # ⚠️ CRITICAL: None = deployment mode (27 components, no contrastive learning)
                original_audio_waveform=None,  # ⚠️ CRITICAL: None = no original audio needed
                return_component_contributions=self.debug
            )
        
        # Parse outputs
        logits = outputs['logits']  # [1, 2]
        probabilities = torch.softmax(logits, dim=1)[0]  # [2]
        
        real_prob = probabilities[0].item()
        fake_prob = probabilities[1].item()
        
        prediction = 'FAKE' if fake_prob > real_prob else 'REAL'
        confidence = max(fake_prob, real_prob) * 100
        
        processing_time = time.time() - start_time
        
        # Build results
        results = {
            'prediction': prediction,
            'confidence': confidence,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'video_path': str(video_path)
        }
        
        # Add component contributions if debug mode
        if self.debug and 'component_contributions' in outputs:
            results['component_contributions'] = {
                k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in outputs['component_contributions'].items()
            }
        
        # Print results
        self._print_results(results)
        
        return results
    
    def detect_from_webcam(self, duration: int = 10, display: bool = True):
        """
        Real-time deepfake detection from webcam
        
        Args:
            duration: How long to capture (seconds)
            display: Show video window with results
        """
        print(f"\n{'='*80}")
        print(f"📹 REAL-TIME WEBCAM DETECTION (Duration: {duration}s)")
        print(f"{'='*80}")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        frame_buffer = []
        results_history = []
        
        print("[INFO] Press 'q' to quit early")
        
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            
            if not ret:
                print("[ERROR] Cannot read frame from webcam")
                break
            
            # Add to buffer
            frame_buffer.append(frame.copy())
            
            # Process every 16 frames (about 0.5s at 30fps)
            if len(frame_buffer) >= self.target_frames:
                # Process frames
                frames_tensor = []
                for f in frame_buffer[-self.target_frames:]:
                    f_tensor = self.preprocess_webcam_frame(f)
                    frames_tensor.append(f_tensor)
                
                video_tensor = torch.cat(frames_tensor, dim=0).unsqueeze(0)  # [1, 16, 3, 224, 224]
                video_tensor = video_tensor.to(self.device)
                
                # Silent audio (webcam has no audio in this simple implementation)
                audio_tensor = torch.zeros(1, int(self.sample_rate * self.audio_duration)).to(self.device)
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(
                        video_frames=video_tensor,
                        audio_waveform=audio_tensor,
                        original_video_frames=None,  # Deployment mode
                        original_audio_waveform=None,
                        return_component_contributions=False
                    )
                
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=1)[0]
                
                fake_prob = probabilities[1].item()
                prediction = 'FAKE' if fake_prob > 0.5 else 'REAL'
                confidence = max(fake_prob, 1 - fake_prob) * 100
                
                results_history.append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'fake_prob': fake_prob
                })
                
                # Clear buffer
                frame_buffer = frame_buffer[-8:]  # Keep some overlap
            
            # Display
            if display and results_history:
                latest = results_history[-1]
                
                # Draw results on frame
                color = (0, 0, 255) if latest['prediction'] == 'FAKE' else (0, 255, 0)
                text = f"{latest['prediction']} ({latest['confidence']:.1f}%)"
                
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, color, 2, cv2.LINE_AA)
                
                # Draw fake probability bar
                bar_width = int(latest['fake_prob'] * 400)
                cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), color, -1)
                cv2.rectangle(frame, (10, 50), (410, 70), (255, 255, 255), 2)
                
                cv2.imshow('Deepfake Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User quit early")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        if results_history:
            avg_fake_prob = np.mean([r['fake_prob'] for r in results_history])
            print(f"\n{'='*80}")
            print(f"📊 WEBCAM DETECTION SUMMARY")
            print(f"{'='*80}")
            print(f"Total predictions: {len(results_history)}")
            print(f"Average fake probability: {avg_fake_prob*100:.1f}%")
            print(f"Final verdict: {'FAKE' if avg_fake_prob > 0.5 else 'REAL'}")
    
    def _print_results(self, results: Dict):
        """Pretty print detection results"""
        print(f"\n{'='*80}")
        print(f"🎯 DETECTION RESULTS")
        print(f"{'='*80}")
        print(f"Prediction:          {results['prediction']}")
        print(f"Confidence:          {results['confidence']:.2f}%")
        print(f"Fake Probability:    {results['fake_probability']*100:.2f}%")
        print(f"Real Probability:    {results['real_probability']*100:.2f}%")
        print(f"Processing Time:     {results['processing_time']:.3f}s")
        print(f"{'='*80}")
        
        if 'component_contributions' in results:
            print(f"\n📊 Component Contributions (Top 10):")
            contributions = results['component_contributions']
            # Sort by magnitude
            sorted_contribs = sorted(
                contributions.items(),
                key=lambda x: np.abs(np.mean(x[1])) if isinstance(x[1], list) else np.abs(x[1]),
                reverse=True
            )[:10]
            
            for name, value in sorted_contribs:
                if isinstance(value, list):
                    value = np.mean(value)
                print(f"  • {name:30s}: {value:.4f}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file for testing')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for real-time detection')
    parser.add_argument('--duration', type=int, default=10,
                       help='Webcam capture duration (seconds)')
    parser.add_argument('--quantized', action='store_true',
                       help='Use INT8 quantized model')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with component contributions')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DeepfakeDetector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        quantized=args.quantized,
        debug=args.debug
    )
    
    # Run detection
    if args.webcam:
        # Real-time webcam
        detector.detect_from_webcam(duration=args.duration, display=True)
        
    elif args.video:
        # Video file
        results = detector.detect_from_video_file(args.video)
        
        # Save results to JSON
        import json
        output_path = Path(args.video).stem + '_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {output_path}")
        
    else:
        print("Error: Must specify either --video or --webcam")
        parser.print_help()


if __name__ == '__main__':
    main()
