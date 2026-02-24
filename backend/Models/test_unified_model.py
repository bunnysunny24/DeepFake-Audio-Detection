#!/usr/bin/env python3
"""
Test script for the unified multimodal deepfake detection model.

This script validates that the unified model works correctly with both
basic and advanced configurations.
"""

import torch
import sys
import os
import traceback
from pathlib import Path

# Add the Models directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from multi_modal_model_unified import (
        MultiModalDeepfakeModel,
        UnifiedModelConfig,
        create_unified_model,
        get_model_info
    )
    print("✓ Successfully imported unified model components")
except ImportError as e:
    print(f"✗ Failed to import unified model: {e}")
    traceback.print_exc()
    sys.exit(1)


def test_basic_model():
    """Test basic model configuration."""
    print("\n=== Testing Basic Model Configuration ===")
    
    try:
        # Create basic configuration
        config = UnifiedModelConfig(
            num_classes=2,
            backbone_visual='efficientnet',
            backbone_audio='wav2vec2',
            enable_adversarial_training=False,
            enable_self_supervised=False,
            enable_curriculum_learning=False,
            enable_active_learning=False,
            enable_real_time_optimization=False,
            enable_ensemble=False,
            debug=True
        )
        
        # Create model
        model = MultiModalDeepfakeModel(config=config)
        print("✓ Basic model created successfully")
        
        # Test model info
        info = get_model_info(model)
        print(f"✓ Model info retrieved: {info['model_type']}")
        
        # Test forward pass with dummy data
        batch_size, seq_len, channels, height, width = 1, 4, 3, 224, 224
        audio_len = 8000  # 0.5 seconds at 16kHz
        
        dummy_video = torch.randn(batch_size, seq_len, channels, height, width)
        dummy_audio = torch.randn(batch_size, audio_len)
        dummy_spectrogram = torch.randn(batch_size, 1, 64, 64)
        
        print(f"✓ Created dummy data - Video: {dummy_video.shape}, Audio: {dummy_audio.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_video, dummy_audio, dummy_spectrogram)
        
        print(f"✓ Forward pass successful - Output shape: {output['predictions'].shape}")
        
        # Verify output structure
        expected_keys = ['predictions', 'features', 'visual_features', 'audio_features', 
                        'forensic_features', 'av_sync_score']
        for key in expected_keys:
            if key in output:
                print(f"✓ Output contains expected key: {key}")
            else:
                print(f"✗ Missing expected key: {key}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic model test failed: {e}")
        traceback.print_exc()
        return False


def test_advanced_model():
    """Test advanced model configuration with limited features."""
    print("\n=== Testing Advanced Model Configuration ===")
    
    try:
        # Create advanced configuration (limited to avoid heavy dependencies)
        config = UnifiedModelConfig(
            num_classes=2,
            backbone_visual='efficientnet',
            backbone_audio='wav2vec2',
            enable_adversarial_training=True,
            enable_self_supervised=False,  # Skip to avoid heavy computation
            enable_curriculum_learning=True,
            enable_active_learning=True,
            enable_real_time_optimization=True,
            enable_ensemble=False,  # Skip to avoid heavy computation
            adversarial_epsilon=0.01,
            curriculum_progression_epochs=10,
            window_size=8,
            buffer_size=16,
            debug=True
        )
        
        # Create model
        model = MultiModalDeepfakeModel(config=config)
        print("✓ Advanced model created successfully")
        
        # Test model info
        info = get_model_info(model)
        print(f"✓ Model info retrieved: {info['model_type']}")
        
        # Verify advanced features are enabled
        features = info['features_enabled']
        if features['adversarial_training']:
            print("✓ Adversarial training enabled")
        if features['curriculum_learning']:
            print("✓ Curriculum learning enabled")
        if features['active_learning']:
            print("✓ Active learning enabled")
        if features['real_time_optimization']:
            print("✓ Real-time optimization enabled")
        
        # Test forward pass with dummy data
        batch_size, seq_len, channels, height, width = 1, 4, 3, 224, 224
        audio_len = 8000
        
        dummy_video = torch.randn(batch_size, seq_len, channels, height, width)
        dummy_audio = torch.randn(batch_size, audio_len)
        dummy_spectrogram = torch.randn(batch_size, 1, 64, 64)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_video, dummy_audio, dummy_spectrogram)
        
        print(f"✓ Forward pass successful - Output shape: {output['predictions'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced model test failed: {e}")
        traceback.print_exc()
        return False


def test_factory_function():
    """Test the factory function for creating models."""
    print("\n=== Testing Factory Function ===")
    
    try:
        # Test factory function
        model = create_unified_model(
            num_classes=3,
            backbone_visual='efficientnet',
            backbone_audio='wav2vec2',
            enable_adversarial_training=False
        )
        
        print("✓ Factory function created model successfully")
        
        # Test configuration
        config = model.get_config()
        if config.num_classes == 3:
            print("✓ Configuration applied correctly")
        else:
            print(f"✗ Configuration not applied correctly: expected 3 classes, got {config.num_classes}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Factory function test failed: {e}")
        traceback.print_exc()
        return False


def test_feature_toggle():
    """Test feature enable/disable functionality."""
    print("\n=== Testing Feature Toggle ===")
    
    try:
        # Create model with features disabled
        model = create_unified_model(
            enable_adversarial_training=False,
            enable_ensemble=False
        )
        
        # Test enabling features
        model.enable_feature('adversarial_training')
        model.enable_feature('ensemble')
        
        config = model.get_config()
        if config.enable_adversarial_training and config.enable_ensemble:
            print("✓ Features enabled successfully")
        else:
            print("✗ Features not enabled correctly")
            return False
        
        # Test disabling features
        model.disable_feature('adversarial_training')
        model.disable_feature('ensemble')
        
        config = model.get_config()
        if not config.enable_adversarial_training and not config.enable_ensemble:
            print("✓ Features disabled successfully")
        else:
            print("✗ Features not disabled correctly")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Feature toggle test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Starting Unified Model Tests...")
    
    # Check PyTorch availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Run tests
    tests = [
        test_basic_model,
        test_advanced_model,
        test_factory_function,
        test_feature_toggle
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)