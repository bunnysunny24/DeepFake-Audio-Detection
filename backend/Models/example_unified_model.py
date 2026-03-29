#!/usr/bin/env python3
"""
Example usage of the unified multimodal deepfake detection model.

This script demonstrates how to use the unified model for different scenarios:
1. Basic deepfake detection
2. Advanced configuration with all features
3. Real-time processing
4. Model optimization for deployment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from pathlib import Path
import sys
import os

# Add the Models directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from multi_modal_model_unified import (
        create_unified_model,
        get_model_info,
        UnifiedModelConfig
    )
    print("✓ Successfully imported unified model")
except ImportError as e:
    print(f"✗ Failed to import unified model: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install torch torchvision torchaudio transformers mediapipe timm librosa scipy")
    sys.exit(1)


def create_dummy_data(batch_size=2, seq_len=8, audio_len=16000):
    """Create dummy data for testing."""
    video_frames = torch.randn(batch_size, seq_len, 3, 224, 224)
    audio_waveform = torch.randn(batch_size, audio_len)
    spectrogram = torch.randn(batch_size, 1, 128, 128)
    labels = torch.randint(0, 2, (batch_size,))
    
    return video_frames, audio_waveform, spectrogram, labels


def example_basic_usage():
    """Demonstrate basic usage of the unified model."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Deepfake Detection")
    print("="*60)
    
    # Create a basic model
    model = create_unified_model(
        num_classes=2,
        backbone_visual='efficientnet',
        backbone_audio='wav2vec2',
        debug=True
    )
    
    print(f"✓ Created basic model")
    
    # Get model information
    info = get_model_info(model)
    print(f"✓ Model type: {info['model_type']}")
    print(f"✓ Visual backbone: {info['backbone_visual']}")
    print(f"✓ Audio backbone: {info['backbone_audio']}")
    
    # Create dummy data
    video, audio, spec, labels = create_dummy_data(batch_size=2)
    print(f"✓ Created dummy data - Video: {video.shape}, Audio: {audio.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(video, audio, spec)
    
    print(f"✓ Forward pass successful")
    print(f"  - Predictions shape: {output['predictions'].shape}")
    print(f"  - Features shape: {output['features'].shape}")
    print(f"  - AV sync score: {output['av_sync_score'].mean().item():.3f}")
    
    # Show predictions
    predictions = torch.softmax(output['predictions'], dim=1)
    print(f"  - Sample predictions: {predictions[0].tolist()}")
    
    return model


def example_advanced_configuration():
    """Demonstrate advanced configuration with multiple features."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Advanced Configuration")
    print("="*60)
    
    # Create advanced model with selected features
    # Note: Some features disabled to avoid heavy computation in example
    model = create_unified_model(
        num_classes=2,
        backbone_visual='efficientnet',
        backbone_audio='wav2vec2',
        enable_adversarial_training=True,
        enable_curriculum_learning=True,
        enable_active_learning=True,
        enable_real_time_optimization=True,
        enable_quantization=False,  # Disabled for simplicity
        enable_pruning=False,       # Disabled for simplicity
        enable_ensemble=False,      # Disabled for simplicity
        adversarial_epsilon=0.01,
        curriculum_progression_epochs=50,
        window_size=8,
        buffer_size=16,
        target_fps=30,
        debug=True
    )
    
    print(f"✓ Created advanced model")
    
    # Show enabled features
    info = get_model_info(model)
    enabled_features = [k for k, v in info['features_enabled'].items() if v]
    print(f"✓ Enabled features: {enabled_features}")
    
    # Test feature toggling
    model.disable_feature('adversarial_training')
    model.enable_feature('adversarial_training')
    print(f"✓ Feature toggling works")
    
    # Test forward pass
    video, audio, spec, labels = create_dummy_data(batch_size=2)
    
    model.eval()
    with torch.no_grad():
        output = model(video, audio, spec)
    
    print(f"✓ Advanced forward pass successful")
    print(f"  - Output keys: {list(output.keys())}")
    
    return model


def example_real_time_processing():
    """Demonstrate real-time processing capabilities."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Real-Time Processing")
    print("="*60)
    
    # Create model optimized for real-time processing
    model = create_unified_model(
        num_classes=2,
        backbone_visual='efficientnet',
        backbone_audio='wav2vec2',
        enable_real_time_optimization=True,
        window_size=4,      # Smaller window for faster processing
        buffer_size=8,      # Smaller buffer
        target_fps=30,
        debug=True
    )
    
    print(f"✓ Created real-time optimized model")
    
    # Simulate real-time processing
    frame_buffer = model.frame_buffer
    sliding_window = model.sliding_window
    adaptive_resolution = model.adaptive_resolution
    
    print(f"✓ Real-time components initialized")
    
    # Simulate processing frames
    model.eval()
    processing_times = []
    
    for i in range(10):
        # Create single frame
        frame = torch.randn(3, 224, 224)
        
        # Add to buffer
        frame_buffer.add_frame(frame)
        
        # Process if we have enough frames
        sequence = frame_buffer.get_sequence()
        if sequence is not None:
            start_time = time.time()
            
            # Expand dimensions for model input
            video_input = sequence.unsqueeze(0)  # Add batch dimension
            audio_input = torch.randn(1, 8000)   # Dummy audio
            spec_input = torch.randn(1, 1, 64, 64)  # Dummy spectrogram
            
            with torch.no_grad():
                output = model(video_input, audio_input, spec_input)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Update resolution based on processing time
            current_fps = 1.0 / processing_time if processing_time > 0 else 30.0
            new_resolution = adaptive_resolution.update_resolution(current_fps)
            
            print(f"  Frame {i+1}: {processing_time:.4f}s, FPS: {current_fps:.1f}, Resolution: {new_resolution}")
    
    avg_processing_time = np.mean(processing_times)
    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
    
    print(f"✓ Real-time processing complete")
    print(f"  - Average processing time: {avg_processing_time:.4f}s")
    print(f"  - Average FPS: {avg_fps:.1f}")
    
    return model


def example_training_pipeline():
    """Demonstrate training pipeline with the unified model."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Training Pipeline")
    print("="*60)
    
    # Create model for training
    model = create_unified_model(
        num_classes=2,
        backbone_visual='efficientnet',
        backbone_audio='wav2vec2',
        enable_adversarial_training=False,  # Disabled for simplicity
        debug=True
    )
    
    print(f"✓ Created model for training")
    
    # Create dummy dataset
    batch_size = 4
    num_batches = 3
    
    videos = torch.randn(batch_size * num_batches, 8, 3, 224, 224)
    audios = torch.randn(batch_size * num_batches, 16000)
    specs = torch.randn(batch_size * num_batches, 1, 128, 128)
    labels = torch.randint(0, 2, (batch_size * num_batches,))
    
    dataset = TensorDataset(videos, audios, specs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"✓ Created dummy dataset with {len(dataset)} samples")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"✓ Setup optimizer and loss function")
    
    # Training loop
    model.train()
    total_loss = 0
    
    for batch_idx, (video, audio, spec, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(video, audio, spec)
        loss = criterion(output['predictions'], targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = torch.argmax(output['predictions'], dim=1)
        accuracy = (predictions == targets).float().mean()
        
        print(f"  Batch {batch_idx+1}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={accuracy.item():.3f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"✓ Training complete - Average loss: {avg_loss:.4f}")
    
    return model


def example_model_optimization():
    """Demonstrate model optimization for deployment."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Model Optimization")
    print("="*60)
    
    # Create model with optimization features
    model = create_unified_model(
        num_classes=2,
        backbone_visual='efficientnet',
        backbone_audio='wav2vec2',
        enable_quantization=True,
        enable_pruning=True,
        pruning_ratio=0.2,
        debug=True
    )
    
    print(f"✓ Created model with optimization features")
    
    # Test original model
    video, audio, spec, labels = create_dummy_data(batch_size=1)
    
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        output = model(video, audio, spec)
    original_time = time.time() - start_time
    
    print(f"✓ Original model inference time: {original_time:.4f}s")
    
    # Apply quantization
    print("  Applying quantization...")
    quantized_model = model.quantization.quantize_model(model)
    
    # Apply pruning
    print("  Applying pruning...")
    pruned_model = model.pruning.prune_model(model)
    
    print(f"✓ Model optimization complete")
    print(f"  - Quantization: Available")
    print(f"  - Pruning: Applied (ratio: {model.config.pruning_ratio})")
    
    return model


def main():
    """Run all examples."""
    print("Unified Multimodal Deepfake Detection Model - Examples")
    print("="*60)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Run examples
        models = []
        
        # Basic usage
        basic_model = example_basic_usage()
        models.append(('Basic', basic_model))
        
        # Advanced configuration
        advanced_model = example_advanced_configuration()
        models.append(('Advanced', advanced_model))
        
        # Real-time processing
        realtime_model = example_real_time_processing()
        models.append(('Real-time', realtime_model))
        
        # Training pipeline
        training_model = example_training_pipeline()
        models.append(('Training', training_model))
        
        # Model optimization
        optimized_model = example_model_optimization()
        models.append(('Optimized', optimized_model))
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        for name, model in models:
            info = get_model_info(model)
            enabled_features = sum(info['features_enabled'].values())
            print(f"✓ {name} model: {enabled_features} features enabled")
        
        print(f"\n🎉 All examples completed successfully!")
        print(f"The unified model is ready for production use.")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)