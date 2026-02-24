#!/usr/bin/env python3
"""
Simple test script for the unified multimodal deepfake detection model.

This test validates the model structure and components without requiring
internet access to download pretrained models.
"""

import torch
import torch.nn as nn
import sys
import os
import traceback
from pathlib import Path

# Add the Models directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from multi_modal_model_unified import (
        UnifiedModelConfig,
        AttentionFusion,
        TemporalAttention,
        StatsPooling,
        AdversarialNoise,
        GradientMaskingDefense,
        AdversarialTraining,
        SelfSupervisedPretrainer,
        MaskedAutoencoderPretraining,
        CurriculumLearningScheduler,
        ProgressiveTraining,
        ActiveLearningSelector,
        ModelQuantization,
        ModelPruning,
        KnowledgeDistillation,
        SlidingWindowInference,
        FrameBufferManager,
        AdaptiveResolutionScaling,
        MultiHeadEnsemble,
        BayesianUncertaintyEstimation,
        ForensicConsistencyModule,
        AudioVisualSyncDetector,
        get_model_info
    )
    print("✓ Successfully imported unified model components")
except ImportError as e:
    print(f"✗ Failed to import unified model: {e}")
    traceback.print_exc()
    sys.exit(1)


def test_config_system():
    """Test the configuration system."""
    print("\n=== Testing Configuration System ===")
    
    try:
        # Test basic configuration
        config = UnifiedModelConfig(
            num_classes=3,
            backbone_visual='efficientnet',
            backbone_audio='wav2vec2',
            enable_adversarial_training=True,
            enable_self_supervised=False,
            adversarial_epsilon=0.02
        )
        
        print("✓ Basic configuration created")
        
        # Test configuration values
        assert config.num_classes == 3
        assert config.backbone_visual == 'efficientnet'
        assert config.enable_adversarial_training == True
        assert config.enable_self_supervised == False
        assert config.adversarial_epsilon == 0.02
        
        print("✓ Configuration values correct")
        
        # Test default values
        assert config.transformer_dim == 768
        assert config.num_transformer_layers == 4
        assert config.enable_face_mesh == True
        
        print("✓ Default values correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_fusion_modules():
    """Test the fusion modules."""
    print("\n=== Testing Fusion Modules ===")
    
    try:
        # Test AttentionFusion
        attention_fusion = AttentionFusion(
            visual_dim=512,
            audio_dim=768,
            output_dim=256
        )
        
        batch_size = 2
        seq_len = 8
        visual_features = torch.randn(batch_size, seq_len, 512)
        audio_features = torch.randn(batch_size, seq_len, 768)
        
        fused_features = attention_fusion(visual_features, audio_features)
        assert fused_features.shape == (batch_size, seq_len, 256)
        
        print("✓ AttentionFusion working correctly")
        
        # Test TemporalAttention
        temporal_attention = TemporalAttention(dim=256, num_heads=8)
        
        input_features = torch.randn(batch_size, seq_len, 256)
        output_features = temporal_attention(input_features)
        assert output_features.shape == (batch_size, seq_len, 256)
        
        print("✓ TemporalAttention working correctly")
        
        # Test StatsPooling
        stats_pooling = StatsPooling()
        
        input_features = torch.randn(batch_size, seq_len, 256)
        pooled_features = stats_pooling(input_features)
        assert pooled_features.shape == (batch_size, 512)  # 2 * 256 (mean + std)
        
        print("✓ StatsPooling working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Fusion modules test failed: {e}")
        traceback.print_exc()
        return False


def test_adversarial_modules():
    """Test the adversarial training modules."""
    print("\n=== Testing Adversarial Modules ===")
    
    try:
        # Test AdversarialNoise
        adversarial_noise = AdversarialNoise(epsilon=0.01, alpha=0.005, num_steps=5)
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        grad_tensor = torch.randn(batch_size, 3, 224, 224)
        
        # Test FGSM attack
        fgsm_output = adversarial_noise(input_tensor, attack_type='fgsm', grad=grad_tensor)
        assert fgsm_output.shape == input_tensor.shape
        
        # Test PGD attack
        pgd_output = adversarial_noise(input_tensor, attack_type='pgd', grad=grad_tensor)
        assert pgd_output.shape == input_tensor.shape
        
        print("✓ AdversarialNoise working correctly")
        
        # Test GradientMaskingDefense
        gradient_masking = GradientMaskingDefense(masking_ratio=0.1)
        
        masked_grad = gradient_masking(grad_tensor)
        assert masked_grad.shape == grad_tensor.shape
        
        print("✓ GradientMaskingDefense working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Adversarial modules test failed: {e}")
        traceback.print_exc()
        return False


def test_curriculum_learning():
    """Test curriculum learning modules."""
    print("\n=== Testing Curriculum Learning ===")
    
    try:
        # Test CurriculumLearningScheduler
        scheduler = CurriculumLearningScheduler(
            initial_difficulty=0.3,
            final_difficulty=1.0,
            progression_epochs=100,
            strategy='linear'
        )
        
        # Test difficulty progression
        difficulty_0 = scheduler.get_difficulty(0)
        difficulty_50 = scheduler.get_difficulty(50)
        difficulty_100 = scheduler.get_difficulty(100)
        
        assert difficulty_0 == 0.3
        assert 0.3 < difficulty_50 < 1.0
        assert difficulty_100 == 1.0
        
        print("✓ CurriculumLearningScheduler working correctly")
        
        # Test exponential strategy
        exp_scheduler = CurriculumLearningScheduler(
            initial_difficulty=0.1,
            final_difficulty=1.0,
            progression_epochs=100,
            strategy='exponential'
        )
        
        exp_difficulty = exp_scheduler.get_difficulty(50)
        assert 0.1 < exp_difficulty < 1.0
        
        print("✓ Exponential curriculum strategy working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Curriculum learning test failed: {e}")
        traceback.print_exc()
        return False


def test_active_learning():
    """Test active learning modules."""
    print("\n=== Testing Active Learning ===")
    
    try:
        # Test ActiveLearningSelector
        selector = ActiveLearningSelector(strategy='uncertainty', batch_size=32)
        
        # Test uncertainty sampling
        batch_size = 100
        num_classes = 2
        predictions = torch.softmax(torch.randn(batch_size, num_classes), dim=1)
        
        indices = selector.uncertainty_sampling(predictions, n_samples=10)
        assert len(indices) == 10
        assert all(0 <= idx < batch_size for idx in indices)
        
        print("✓ Uncertainty sampling working correctly")
        
        # Test with features (will fall back to random sampling without sklearn)
        features = torch.randn(batch_size, 128)
        indices = selector.diversity_sampling(features, n_samples=10)
        assert len(indices) == 10
        
        print("✓ Diversity sampling working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Active learning test failed: {e}")
        traceback.print_exc()
        return False


def test_optimization_modules():
    """Test model optimization modules."""
    print("\n=== Testing Optimization Modules ===")
    
    try:
        # Test ModelQuantization
        quantization = ModelQuantization(quantization_type='dynamic')
        
        # Create a simple model
        simple_model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        quantized_model = quantization.quantize_model(simple_model)
        assert quantized_model is not None
        
        print("✓ ModelQuantization working correctly")
        
        # Test ModelPruning
        pruning = ModelPruning(pruning_ratio=0.2, structured=False)
        
        test_model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        pruned_model = pruning.prune_model(test_model)
        assert pruned_model is not None
        
        print("✓ ModelPruning working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimization modules test failed: {e}")
        traceback.print_exc()
        return False


def test_real_time_modules():
    """Test real-time processing modules."""
    print("\n=== Testing Real-Time Processing ===")
    
    try:
        # Test SlidingWindowInference
        sliding_window = SlidingWindowInference(window_size=16, overlap=8)
        
        assert sliding_window.window_size == 16
        assert sliding_window.overlap == 8
        assert sliding_window.stride == 8
        
        print("✓ SlidingWindowInference initialized correctly")
        
        # Test FrameBufferManager
        frame_buffer = FrameBufferManager(buffer_size=32, min_frames=8)
        
        # Add some frames
        for i in range(10):
            frame = torch.randn(3, 224, 224)
            frame_buffer.add_frame(frame)
        
        sequence = frame_buffer.get_sequence()
        assert sequence is not None
        assert sequence.shape[0] == 8  # min_frames
        
        print("✓ FrameBufferManager working correctly")
        
        # Test AdaptiveResolutionScaling
        adaptive_res = AdaptiveResolutionScaling(
            target_fps=30,
            min_resolution=224,
            max_resolution=512
        )
        
        # Test resolution updates
        current_res = adaptive_res.update_resolution(25)  # Below target
        assert current_res <= 512
        
        current_res = adaptive_res.update_resolution(35)  # Above target
        assert current_res >= 224
        
        print("✓ AdaptiveResolutionScaling working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Real-time modules test failed: {e}")
        traceback.print_exc()
        return False


def test_forensic_modules():
    """Test forensic analysis modules."""
    print("\n=== Testing Forensic Modules ===")
    
    try:
        # Test ForensicConsistencyModule
        forensic_module = ForensicConsistencyModule(feature_dim=256)
        
        batch_size = 2
        input_images = torch.randn(batch_size, 3, 224, 224)
        
        consistency_score = forensic_module(input_images)
        assert consistency_score.shape == (batch_size, 256)
        
        print("✓ ForensicConsistencyModule working correctly")
        
        # Test AudioVisualSyncDetector
        av_sync_detector = AudioVisualSyncDetector(
            visual_dim=512,
            audio_dim=768,
            hidden_dim=256
        )
        
        visual_features = torch.randn(batch_size, 512)
        audio_features = torch.randn(batch_size, 768)
        
        sync_score = av_sync_detector(visual_features, audio_features)
        assert sync_score.shape == (batch_size, 1)
        assert torch.all(sync_score >= 0) and torch.all(sync_score <= 1)  # Sigmoid output
        
        print("✓ AudioVisualSyncDetector working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Forensic modules test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Starting Unified Model Component Tests...")
    
    # Check PyTorch availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Run tests
    tests = [
        test_config_system,
        test_fusion_modules,
        test_adversarial_modules,
        test_curriculum_learning,
        test_active_learning,
        test_optimization_modules,
        test_real_time_modules,
        test_forensic_modules
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
        print("🎉 All component tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)