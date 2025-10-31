import torch
import os
import sys
import cv2
import numpy as np
from multi_modal_model import MultiModalDeepfakeModel

def extract_video_frames(video_path, num_frames=32, resize=(224, 224)):
    """Extract frames from a video file."""
    print(f"📹 Extracting frames from {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(torch.tensor(frame, device='cuda').permute(2, 0, 1).float() / 255.)  # [C, H, W]
    
    cap.release()
    
    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video.")
    
    return torch.stack(frames)  # [num_frames, C, H, W]

def extract_audio_tensor(video_path, audio_length=8000):
    """Extract audio from video file using ffmpeg and librosa."""
    print(f"🔊 Extracting audio from {video_path}")
    import subprocess
    import tempfile
    import librosa
    import shlex

    # Create a temporary wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name

    # Build the FFmpeg command safely
    cmd = f'ffmpeg -y -i "{video_path}" -ar 16000 -ac 1 -vn "{temp_wav_path}"'

    try:
        # Use shlex.split for safe argument parsing
        result = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        # Load the extracted audio
        audio, sample_rate = librosa.load(temp_wav_path, sr=16000, mono=True)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode(errors="ignore")[:500]
        raise RuntimeError(f"Failed to extract audio with ffmpeg.\nCommand: {cmd}\nError: {error_msg}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during audio extraction: {e}")
    finally:
        # Cleanup temp file safely
        try:
            os.remove(temp_wav_path)
        except Exception:
            pass

    # Center crop or pad the audio to a fixed length
    if len(audio) > audio_length:
        start = (len(audio) - audio_length) // 2
        audio = audio[start:start + audio_length]
    else:
        audio = np.pad(audio, (0, audio_length - len(audio)), mode='constant')

    return torch.tensor(audio, dtype=torch.float32, device='cuda')  # [audio_length]

def create_matched_model(checkpoint_dict):
    """Create a model that exactly matches the dimensions in the checkpoint."""
    
    # Find the transformer dimension in the checkpoint (this is critical for all parameters to match)
    transformer_dim = 768  # Default dimension
    
    # Check transformer layers for dimension information
    for key, value in checkpoint_dict.items():
        if 'transformer.layers' in key and 'self_attn.out_proj.weight' in key:
            transformer_dim = value.shape[0]  # First dimension of output projection is transformer dimension
            print(f"  ✓ Found transformer dimension from layers: {transformer_dim}")
            break
    
    # Find the classifier dimensions in the checkpoint
    classifier_input_dim = transformer_dim  # Default to transformer dimension
    for key, value in checkpoint_dict.items():
        if 'classifier.0.weight' in key:
            classifier_input_dim = value.shape[1]
            print(f"  ✓ Found classifier input dimension: {classifier_input_dim}")
            break
    
    # Create a model with the matched dimensions
    print(f"  ✓ Creating model with transformer dimension: {transformer_dim}")
    model = MultiModalDeepfakeModel(
        num_classes=2,
        video_feature_dim=1024,
        audio_feature_dim=1024, 
        transformer_dim=transformer_dim,  # Use the exact transformer dimension from checkpoint
        num_transformer_layers=4,
        enable_face_mesh=True,
        enable_explainability=False,  # Simplified to match training
        fusion_type='attention',
        backbone_visual='efficientnet',
        backbone_audio='wav2vec2',
        use_spectrogram=True,
        detect_deepfake_type=False,
        enable_skin_color_analysis=False,
        enable_advanced_physiological=False,
    )
    
    return model, classifier_input_dim

def load_model(model_path, device):
    """Load model from checkpoint, matching the exact configuration used during training."""
    print(f"🔄 Loading model from {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    print("\n📊 Model Configuration Analysis:")
    
    # Extract configuration from checkpoint if available
    config = {}
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        print(f"  ✓ Found configuration in checkpoint")
        print(f"  ✓ Transformer dim: {config.get('transformer_dim', 'Not specified')}")
        print(f"  ✓ Advanced features: {not config.get('disable_advanced_physio', False)}")
        print(f"  ✓ Skin analysis: {not config.get('disable_skin_analysis', False)}")
        
    # Extract the state dict for dimension matching
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Create a model that matches the exact dimensions in the checkpoint
        model, classifier_dim = create_matched_model(checkpoint['model_state_dict'])
        print(f"  ✓ Created model with matched classifier dimension: {classifier_dim}")
    else:
        # Fallback to default model
        model = MultiModalDeepfakeModel(
            num_classes=2,
            video_feature_dim=1024,
            audio_feature_dim=1024, 
            transformer_dim=768,
            num_transformer_layers=4,
            enable_face_mesh=True,
            enable_explainability=False,
            fusion_type='attention',
            backbone_visual='efficientnet',
            backbone_audio='wav2vec2',
            use_spectrogram=True,
            detect_deepfake_type=False,
            enable_skin_color_analysis=False,
            enable_advanced_physiological=False,
        )
    
    # Debug model structure
    print("\n🔍 Model Structure Analysis:")
    classifier_shapes = {}
    fusion_dims = []
    
    # Load the weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Analyze model structure
        for name, param in model.named_parameters():
            if 'classifier' in name and 'weight' in name:
                classifier_shapes[name] = param.shape
                print(f"  ✓ Model classifier: {name} = {param.shape}")
            if 'fusion' in name and 'weight' in name:
                fusion_dims.append((name, param.shape))
                print(f"  ✓ Fusion layer: {name} = {param.shape}")
        
        # Analyze checkpoint structure
        checkpoint_classifiers = {}
        for key, value in checkpoint['model_state_dict'].items():
            if 'classifier' in key and 'weight' in key:
                checkpoint_classifiers[key] = value.shape
                print(f"  ✓ Checkpoint classifier: {key} = {value.shape}")
        
        # Try to load the state dict
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("\n✅ Model loaded successfully with non-strict loading!")
        except Exception as e:
            print(f"\n❌ Error during non-strict loading: {e}")
            print("\n🔧 Attempting to load with parameter shape matching...")
            
            # Count parameters for reporting
            total_params = len(model.state_dict())
            matched_params = 0
            
            # If that fails, try a more flexible approach
            model_dict = model.state_dict()
            pretrained_dict = {}
            
            # Load only parameters that match in shape
            for k, v in checkpoint['model_state_dict'].items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        pretrained_dict[k] = v
                        matched_params += 1
                    else:
                        print(f"  ⚠️ Shape mismatch: {k} - Model: {model_dict[k].shape}, Checkpoint: {v.shape}")
            
            # Update model with matched parameters
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            
            print(f"\n✅ Loaded {matched_params}/{total_params} parameters successfully ({matched_params/total_params:.1%}).")
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    return model

def main(model_path, video_path):
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Process video
    print(f"🎬 Processing video: {video_path}")
    video_frames = extract_video_frames(video_path)
    audio_tensor = extract_audio_tensor(video_path)
    
    # Prepare inputs
    inputs = {
        "video_frames": video_frames.unsqueeze(0).to(device),  # [1, num_frames, C, H, W]
        "audio": audio_tensor.unsqueeze(0).to(device),         # [1, audio_length]
    }
    
    # Run prediction
    print("\n🧠 Running prediction analysis...")
    with torch.no_grad():
        try:
            output, results = model(inputs)
            
            # Handle the output
            if output.shape[-1] > 1:
                pred = torch.softmax(output, dim=-1)
                fake_prob = float(pred[0, 1])
                real_prob = float(pred[0, 0])
            else:
                fake_prob = float(torch.sigmoid(output).item())
                real_prob = 1.0 - fake_prob
            
            # Check for possible class imbalance during training that needs correction
            # The model might have been trained with extreme class balancing that affects prediction
            print(f"\n⚠️ Model prediction analysis:")
            print(f"  Raw output scores: {output.tolist()}")
            print(f"  Raw probabilities: Real={real_prob:.4f}, Fake={fake_prob:.4f}")
            
            # Based on the training script, the model used:
            # - manual_extreme class weights 
            # - oversample_minority flag
            # - focal loss with alpha=0.75, gamma=2.0
            # - use_weighted_loss flag
            # Extract logits for analysis
            real_logit = output[0, 0].item()
            fake_logit = output[0, 1].item()
            logit_diff = real_logit - fake_logit
            
            print(f"  Logit difference (real-fake): {logit_diff:.4f}")
            print(f"  Raw probabilities: Real={real_prob:.4f}, Fake={fake_prob:.4f}")
            
            # Use simple probability-based decision
            # The model outputs are already calibrated from training
            result = "REAL" if real_prob > fake_prob else "FAKE"
            
            # Confidence is the probability of the predicted class
            confidence = real_prob if result == "REAL" else fake_prob
            confidence_pct = confidence * 100
            
            # Determine confidence level
            if confidence > 0.90:
                confidence_level = "Very High"
            elif confidence > 0.75:
                confidence_level = "High"
            elif confidence > 0.60:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            decision_rule = f"real_prob ({real_prob:.4f}) {'>' if result == 'REAL' else '<'} fake_prob ({fake_prob:.4f})"
            
            print("\n" + "="*60)
            print(f"📊 PREDICTION: {result} (Confidence: {confidence_pct:.2f}% - {confidence_level})")
            print("="*60)
            
            # Show detailed probabilities with visual bar
            print(f"\n📈 Model Analysis:")
            
            # Create visual bar representation
            bar_length = 40
            
            # Show probabilities
            print(f"📊 Output Probabilities:")
            real_bar = "█" * int(real_prob * bar_length)
            fake_bar = "█" * int(fake_prob * bar_length)
            print(f"  REAL: {real_prob:.4f} ({real_prob*100:.2f}%) {real_bar}")
            print(f"  FAKE: {fake_prob:.4f} ({fake_prob*100:.2f}%) {fake_bar}")
            
            # Logit information (for reference)
            print(f"\n📊 Logit Analysis (for reference):")
            print(f"  Real logit: {real_logit:.4f}")
            print(f"  Fake logit: {fake_logit:.4f}")
            print(f"  Logit difference (real-fake): {logit_diff:.4f}")
            
            # Interpretation of result
            print("\n🔍 Interpretation:")
            
            
            if confidence > 0.75:
                if result == "FAKE":
                    print(f"  🚨 HIGH CONFIDENCE: This is highly likely to be a deepfake video.")
                    print(f"     • Model confidence: {confidence_pct:.2f}%")
                    print(f"     • Fake probability ({fake_prob:.4f}) is significantly higher than real ({real_prob:.4f})")
                else:
                    print(f"  ✅ HIGH CONFIDENCE: This is highly likely to be an authentic video.")
                    print(f"     • Model confidence: {confidence_pct:.2f}%")
                    print(f"     • Real probability ({real_prob:.4f}) is significantly higher than fake ({fake_prob:.4f})")
            elif confidence > 0.60:
                if result == "FAKE":
                    print(f"  ⚠️ MEDIUM CONFIDENCE: This video shows characteristics of being manipulated.")
                    print(f"     • Model confidence: {confidence_pct:.2f}%")
                    print(f"     • Fake probability ({fake_prob:.4f}) is higher than real ({real_prob:.4f})")
                else:
                    print(f"  ⚠️ MEDIUM CONFIDENCE: This video appears mostly authentic.")
                    print(f"     • Model confidence: {confidence_pct:.2f}%")
                    print(f"     • Real probability ({real_prob:.4f}) is higher than fake ({fake_prob:.4f})")
            else:
                print(f"  ⚠️ LOW CONFIDENCE: The model is uncertain about this video.")
                print(f"     • Model confidence: {confidence_pct:.2f}%")
                print(f"     • Probabilities are very close: Real={real_prob:.4f}, Fake={fake_prob:.4f}")
                print("     • Consider additional verification methods")
            
            print("\n🔍 Detection Summary:")
            print(f"  • Prediction: {result}")
            print(f"  • Confidence: {confidence_pct:.2f}% ({confidence_level})")
            print(f"  • Decision based on: {decision_rule}")
            
            # Show a simple conclusion
            if result == "FAKE":
                print(f"\n🚨 This video appears to be manipulated (deepfake)")
            else:
                print(f"\n✅ This video appears to be authentic (real)")
            
            # Show key features if available
            if isinstance(results, dict) and results:
                print("\n🔎 Additional Features:")
                feature_count = 0
                for key, value in results.items():
                    if feature_count >= 5:  # Limit to 5 features
                        break
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            print(f"  • {key}: {float(value):.4f}")
                            feature_count += 1
                    elif isinstance(value, float):
                        print(f"  • {key}: {value:.4f}")
                        feature_count += 1
                    
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict_deepfake_fixed.py <model.pth> <video.mp4>")
        exit(1)
    main(sys.argv[1], sys.argv[2])