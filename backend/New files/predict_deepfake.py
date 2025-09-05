import torch
import os
import cv2
import numpy as np
from multi_modal_model import MultiModalDeepfakeModel
import torch.nn.functional as F

class EnsemblePredictor:
    """
    🚀 EXTREME ACCURACY PREDICTOR: 95%+ TARGET
    Uses multiple models + Test-Time Augmentation (TTA)
    """
    
    def __init__(self, model_paths, device='cuda'):
        self.device = device
        self.models = []
        
        print(f"🔧 Loading {len(model_paths)} models for ensemble prediction...")
        for i, model_path in enumerate(model_paths):
            print(f"  📥 Loading model {i+1}: {os.path.basename(model_path)}")
            model = self.load_model(model_path, device)
            self.models.append(model)
        
        print(f"✅ Ensemble loaded: {len(self.models)} models ready")
    
    def load_model(self, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            config = checkpoint.get('config', {})
            model = MultiModalDeepfakeModel(
                num_classes=config.get('num_classes', 2),
                video_feature_dim=config.get('video_feature_dim', 1024),
                audio_feature_dim=config.get('audio_feature_dim', 8000),
                transformer_dim=config.get('transformer_dim', 768),
                num_transformer_layers=config.get('num_transformer_layers', 4),
                enable_face_mesh=config.get('enable_face_mesh', True),
                enable_explainability=config.get('enable_explainability', True),
                fusion_type=config.get('fusion_type', 'attention'),
                backbone_visual=config.get('backbone_visual', 'efficientnet'),
                backbone_audio=config.get('backbone_audio', 'wav2vec2'),
            )
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model = MultiModalDeepfakeModel()
            model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()
        return model

    def extract_video_frames_tta(self, video_path, num_frames=32, base_size=(384, 384)):
        """Extract frames with Test-Time Augmentation"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        # TTA: Different sampling strategies
        tta_variants = []
        
        # Variant 1: Center crop
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, base_size)
            frames.append(torch.tensor(frame, device=self.device).permute(2, 0, 1).float() / 255.)
        if len(frames) > 0:
            tta_variants.append(torch.stack(frames))
        
        # Variant 2: Random start (slight offset)
        offset_indices = np.clip(frame_indices + np.random.randint(-2, 3, len(frame_indices)), 0, total_frames-1)
        frames = []
        for idx in offset_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, base_size)
            frames.append(torch.tensor(frame, device=self.device).permute(2, 0, 1).float() / 255.)
        if len(frames) > 0:
            tta_variants.append(torch.stack(frames))
        
        # Variant 3: Horizontal flip
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)  # Horizontal flip
            frame = cv2.resize(frame, base_size)
            frames.append(torch.tensor(frame, device=self.device).permute(2, 0, 1).float() / 255.)
        if len(frames) > 0:
            tta_variants.append(torch.stack(frames))
        
        cap.release()
        return tta_variants

    def extract_audio_tensor(self, video_path, audio_length=8000):
        """Extract audio with slight variations for TTA"""
        import librosa
        
        try:
            audio, sample_rate = librosa.load(video_path, sr=16000, mono=True)
        except Exception as e:
            print(f"⚠️ Audio extraction failed, using dummy audio: {e}")
            audio = np.zeros(audio_length, dtype=np.float32)

        # TTA: Different audio segments
        tta_audio = []
        
        # Variant 1: Center crop
        if len(audio) > audio_length:
            start = (len(audio) - audio_length) // 2
            audio_center = audio[start:start + audio_length]
        else:
            audio_center = np.pad(audio, (0, audio_length - len(audio)), mode='constant')
        tta_audio.append(torch.tensor(audio_center, dtype=torch.float32, device=self.device))
        
        # Variant 2: Start crop
        if len(audio) > audio_length:
            audio_start = audio[:audio_length]
        else:
            audio_start = np.pad(audio, (0, audio_length - len(audio)), mode='constant')
        tta_audio.append(torch.tensor(audio_start, dtype=torch.float32, device=self.device))
        
        # Variant 3: End crop
        if len(audio) > audio_length:
            audio_end = audio[-audio_length:]
        else:
            audio_end = np.pad(audio, (0, audio_length - len(audio)), mode='constant')
        tta_audio.append(torch.tensor(audio_end, dtype=torch.float32, device=self.device))
        
        return tta_audio

    def predict_with_tta(self, video_path, confidence_threshold=0.8):
        """
        🎯 EXTREME ACCURACY PREDICTION with TTA + Ensemble
        Target: 95%+ accuracy
        """
        print(f"🚀 Predicting with TTA + Ensemble: {video_path}")
        
        # Extract multiple variants
        video_variants = self.extract_video_frames_tta(video_path)
        audio_variants = self.extract_audio_tensor(video_path)
        
        all_predictions = []
        all_confidences = []
        
        # Run prediction for each model and each TTA variant
        for model_idx, model in enumerate(self.models):
            print(f"  🔄 Model {model_idx+1}/{len(self.models)}")
            
            for video_var in video_variants[:min(3, len(video_variants))]:  # Limit variants
                for audio_var in audio_variants[:min(3, len(audio_variants))]:
                    
                    # Create inputs
                    inputs = {
                        "video_frames": video_var.unsqueeze(0),
                        "audio": audio_var.unsqueeze(0),
                        "audio_spectrogram": None,
                        "original_video_frames": video_var.unsqueeze(0),
                        "original_audio": audio_var.unsqueeze(0),
                        "fake_periods": [None],
                        "timestamps": [None],
                        "fake_mask": [None],
                        "face_embeddings": None,
                        "temporal_consistency": torch.zeros((1,), device=self.device),
                        "metadata_features": torch.zeros((1, 10), device=self.device),
                        "ela_features": None,
                        "audio_visual_sync": torch.zeros((1, 5), device=self.device),
                        "facial_landmarks": None,
                        "file_path": [video_path],
                    }
                    
                    with torch.no_grad():
                        try:
                            output, results = model(inputs)
                            probs = torch.softmax(output, dim=-1)
                            
                            all_predictions.append(probs.cpu().numpy()[0])
                            all_confidences.append(torch.max(probs).item())
                            
                        except Exception as e:
                            print(f"    ⚠️ Error in prediction variant: {e}")
                            continue
        
        if len(all_predictions) == 0:
            return "Error", 0.0, "No successful predictions"
        
        # Ensemble averaging
        predictions_array = np.array(all_predictions)
        avg_predictions = np.mean(predictions_array, axis=0)
        ensemble_confidence = np.max(avg_predictions)
        
        # Weighted ensemble (give more weight to high-confidence predictions)
        weights = np.array(all_confidences)
        weights = weights / np.sum(weights)
        weighted_predictions = np.average(predictions_array, axis=0, weights=weights)
        weighted_confidence = np.max(weighted_predictions)
        
        # Final decision with confidence thresholding
        final_predictions = weighted_predictions
        final_confidence = weighted_confidence
        
        fake_prob = final_predictions[1] if len(final_predictions) > 1 else final_predictions[0]
        
        # High confidence threshold for 95%+ accuracy
        if final_confidence < confidence_threshold:
            decision = "Uncertain"
            confidence_value = final_confidence
        elif fake_prob >= 0.5:
            decision = "Fake"
            confidence_value = fake_prob
        else:
            decision = "Real"
            confidence_value = 1 - fake_prob
        
        analysis = {
            'ensemble_size': len(all_predictions),
            'avg_confidence': np.mean(all_confidences),
            'final_confidence': final_confidence,
            'fake_probability': fake_prob,
            'prediction_std': np.std(predictions_array[:, 1] if predictions_array.shape[1] > 1 else predictions_array[:, 0])
        }
        
        return decision, confidence_value, analysis

def main():
    import sys
    if len(sys.argv) < 2:
        print("🚀 EXTREME ACCURACY DEEPFAKE DETECTOR (95%+ Target)")
        print("Usage: python extreme_predict.py <video.mp4>")
        print("\nThis script uses ensemble + TTA for maximum accuracy")
        print("Expected models:")
        print("  1. Best single model: checkpoint_epoch_11_acc_0.7956_f1_0.8651.pth")
        print("  2. Ensemble model 1: (EfficientNet-B4)")
        print("  3. Ensemble model 2: (Swin Transformer)")
        print("  4. Ensemble model 3: (ConvNeXt)")
        return
    
    video_path = sys.argv[1]
    
    # Model paths (update these based on your trained models)
    model_paths = [
        "F:/deepfake/backup/Models/stratified_checkpoints/run_20250904_165625/regular/checkpoint_epoch_11_acc_0.7956_f1_0.8651.pth",
        # Add more model paths as they become available
        # "F:/deepfake/backup/Models/extreme_checkpoints/ensemble_model1/best_model.pth",
        # "F:/deepfake/backup/Models/extreme_checkpoints/ensemble_model2/best_model.pth", 
        # "F:/deepfake/backup/Models/extreme_checkpoints/ensemble_model3/best_model.pth",
    ]
    
    # Filter existing models
    existing_models = [path for path in model_paths if os.path.exists(path)]
    
    if len(existing_models) == 0:
        print("❌ No model files found!")
        return
    
    print(f"🎯 Using {len(existing_models)} models for prediction")
    
    # Create predictor
    predictor = EnsemblePredictor(existing_models)
    
    # Run prediction
    decision, confidence, analysis = predictor.predict_with_tta(video_path, confidence_threshold=0.8)
    
    print(f"\n🎯 EXTREME ACCURACY PREDICTION RESULTS:")
    print(f"📊 Decision: {decision}")
    print(f"🔥 Confidence: {confidence:.4f}")
    print(f"📈 Analysis: {analysis}")
    
    if analysis['prediction_std'] < 0.1:
        print(f"✅ High agreement between models (std: {analysis['prediction_std']:.3f})")
    else:
        print(f"⚠️ Models disagree (std: {analysis['prediction_std']:.3f}) - consider manual review")

if __name__ == "__main__":
    main()
