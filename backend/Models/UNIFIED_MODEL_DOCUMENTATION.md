# Unified Multimodal Deepfake Detection Model

This document describes the unified multimodal deepfake detection model that combines advanced features from both `multi_modal_model.py` and `multi_modal_model_check_1.py`.

## Overview

The unified model (`multi_modal_model_unified.py`) provides a comprehensive deepfake detection system with the following capabilities:

### Core Features from Original Model
- **Facial dynamics analysis** (FAU, micro-expressions, landmarks)
- **Physiological signal analysis** (heart rate, breathing patterns)
- **Visual artifact detection** (lighting, texture, frequency domain)
- **Audio analysis** (voice biometrics, MFCC, phoneme-viseme sync)
- **Siamese networks and autoencoders** for anomaly detection
- **Liveness detection modules**

### Advanced Features from Enhanced Model
- **Adversarial training modules** (FGSM, PGD attacks)
- **Self-supervised learning** (contrastive learning, masked autoencoders)
- **Curriculum learning** with difficulty scheduling
- **Active learning strategies**
- **Model optimization** (quantization, pruning, knowledge distillation)
- **Real-time processing capabilities**
- **Ensemble methods** with Bayesian uncertainty
- **Comprehensive forensic analysis modules**

## Architecture

### Core Components

1. **Visual Backbone**
   - EfficientNet-B0 or Swin Transformer V2
   - Pretrained on ImageNet
   - Configurable feature dimensions

2. **Audio Backbone**
   - Wav2Vec2 or HuBERT
   - Pretrained on speech data
   - Support for spectrogram processing

3. **Fusion Modules**
   - `AttentionFusion`: Cross-modal attention mechanism
   - `TemporalAttention`: Self-attention for temporal sequences
   - `StatsPooling`: Statistical pooling (mean + std)

4. **Forensic Analysis**
   - `ForensicConsistencyModule`: Lighting and texture consistency
   - `AudioVisualSyncDetector`: Synchronization detection

### Advanced Components

1. **Adversarial Robustness**
   - `AdversarialNoise`: FGSM and PGD attacks
   - `GradientMaskingDefense`: Gradient masking
   - `AdversarialTraining`: Integrated adversarial training

2. **Self-Supervised Learning**
   - `SelfSupervisedPretrainer`: Contrastive learning
   - `MaskedAutoencoderPretraining`: Masked reconstruction

3. **Curriculum Learning**
   - `CurriculumLearningScheduler`: Difficulty progression
   - `ProgressiveTraining`: Gradual complexity increase

4. **Active Learning**
   - `ActiveLearningSelector`: Uncertainty and diversity sampling

5. **Model Optimization**
   - `ModelQuantization`: Dynamic/static quantization
   - `ModelPruning`: Structured/unstructured pruning
   - `KnowledgeDistillation`: Teacher-student learning

6. **Real-Time Processing**
   - `SlidingWindowInference`: Streaming inference
   - `FrameBufferManager`: Frame buffering
   - `AdaptiveResolutionScaling`: Dynamic resolution adjustment

7. **Ensemble Methods**
   - `MultiHeadEnsemble`: Multiple model voting
   - `BayesianUncertaintyEstimation`: Monte Carlo dropout

## Usage

### Basic Usage

```python
from multi_modal_model_unified import create_unified_model

# Create basic model
model = create_unified_model(
    num_classes=2,
    backbone_visual='efficientnet',
    backbone_audio='wav2vec2'
)

# Forward pass
output = model(video_frames, audio_waveform, spectrogram)
predictions = output['predictions']
```

### Advanced Configuration

```python
# Create advanced model with all features
advanced_model = create_unified_model(
    num_classes=2,
    backbone_visual='swin',
    backbone_audio='wav2vec2',
    enable_adversarial_training=True,
    enable_self_supervised=True,
    enable_curriculum_learning=True,
    enable_active_learning=True,
    enable_real_time_optimization=True,
    enable_ensemble=True,
    adversarial_epsilon=0.02,
    curriculum_progression_epochs=50,
    pruning_ratio=0.3
)
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_classes` | Number of output classes | 2 |
| `backbone_visual` | Visual backbone ('efficientnet', 'swin') | 'efficientnet' |
| `backbone_audio` | Audio backbone ('wav2vec2', 'hubert') | 'wav2vec2' |
| `enable_adversarial_training` | Enable adversarial robustness | False |
| `enable_self_supervised` | Enable self-supervised pretraining | False |
| `enable_curriculum_learning` | Enable curriculum learning | False |
| `enable_active_learning` | Enable active learning | False |
| `enable_quantization` | Enable model quantization | False |
| `enable_pruning` | Enable model pruning | False |
| `enable_real_time_optimization` | Enable real-time processing | False |
| `enable_ensemble` | Enable ensemble methods | False |
| `adversarial_epsilon` | Adversarial perturbation bound | 0.01 |
| `curriculum_progression_epochs` | Curriculum learning epochs | 100 |
| `pruning_ratio` | Model pruning ratio | 0.2 |
| `window_size` | Sliding window size | 16 |
| `buffer_size` | Frame buffer size | 32 |
| `target_fps` | Target FPS for real-time processing | 30 |

### Feature Management

```python
# Enable/disable features dynamically
model.enable_feature('adversarial_training')
model.disable_feature('ensemble')

# Get model information
info = get_model_info(model)
print(info['features_enabled'])
```

## Input/Output Format

### Input
- `video_frames`: `[batch_size, seq_len, channels, height, width]`
- `audio_waveform`: `[batch_size, audio_length]`
- `spectrogram`: `[batch_size, 1, freq_bins, time_steps]` (optional)

### Output
- `predictions`: Main classification predictions
- `features`: Combined feature representation
- `visual_features`: Visual feature embeddings
- `audio_features`: Audio feature embeddings
- `forensic_features`: Forensic consistency features
- `av_sync_score`: Audio-visual synchronization score
- `deepfake_type_predictions`: Deepfake type classification (if enabled)
- `uncertainty`: Prediction uncertainty (if ensemble enabled)

## Migration from Separate Models

### From `multi_modal_model.py`
```python
# Old usage
from multi_modal_model import MultiModalDeepfakeModel
old_model = MultiModalDeepfakeModel(num_classes=2)

# New usage
from multi_modal_model_unified import create_unified_model
new_model = create_unified_model(num_classes=2)
```

### From `multi_modal_model_check_1.py`
```python
# Old usage
from multi_modal_model_check_1 import MultiModalDeepfakeModel
old_model = MultiModalDeepfakeModel(
    enable_adversarial_training=True,
    enable_self_supervised=True
)

# New usage
from multi_modal_model_unified import create_unified_model
new_model = create_unified_model(
    enable_adversarial_training=True,
    enable_self_supervised=True
)
```

## Training Pipelines

### Standard Training
```python
# Standard supervised training
model = create_unified_model(num_classes=2)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for batch in dataloader:
    video, audio, spec, targets = batch
    output = model(video, audio, spec)
    loss = criterion(output['predictions'], targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Adversarial Training
```python
# Adversarial training
model = create_unified_model(enable_adversarial_training=True)
adversarial_trainer = model.adversarial_training

for batch in dataloader:
    video, audio, spec, targets = batch
    clean_output, adv_output = adversarial_trainer(
        torch.cat([video, audio, spec], dim=1), 
        targets, 
        training=True
    )
    # Compute combined loss
    loss = criterion(clean_output, targets) + criterion(adv_output, targets)
```

### Self-Supervised Pretraining
```python
# Self-supervised pretraining
model = create_unified_model(enable_self_supervised=True)
pretrainer = model.self_supervised_pretrainer

for batch in unlabeled_dataloader:
    video1, audio1, spec1 = batch
    video2, audio2, spec2 = augment_batch(batch)  # Apply augmentations
    
    loss, h1, h2 = pretrainer(
        torch.cat([video1, audio1, spec1], dim=1),
        torch.cat([video2, audio2, spec2], dim=1)
    )
    # Optimize contrastive loss
```

## Performance Optimization

### Quantization
```python
# Apply quantization for deployment
model = create_unified_model(enable_quantization=True)
quantized_model = model.quantization.quantize_model(model)
```

### Pruning
```python
# Apply pruning to reduce model size
model = create_unified_model(enable_pruning=True)
pruned_model = model.pruning.prune_model(model)
```

### Real-Time Processing
```python
# Configure for real-time processing
model = create_unified_model(
    enable_real_time_optimization=True,
    window_size=8,
    buffer_size=16,
    target_fps=30
)

# Use sliding window inference
sliding_window = model.sliding_window
predictions = sliding_window.process_sequence(model, video_sequence)
```

## Troubleshooting

### Common Issues

1. **Memory Usage**: Disable unnecessary features or reduce batch size
2. **Slow Inference**: Enable quantization and pruning
3. **Poor Performance**: Try different backbone combinations
4. **Training Instability**: Adjust learning rates and enable gradient clipping

### Debug Mode
```python
# Enable debug mode for detailed logging
model = create_unified_model(debug=True)
```

## Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Torchvision >= 0.15.0
- Mediapipe >= 0.10.0
- Librosa >= 0.9.2
- Timm >= 0.9.2
- Numpy >= 1.22.0
- Scipy >= 1.8.0

### Optional Dependencies
- dlib (for advanced face detection)
- facenet-pytorch (for face recognition)
- onnx (for model export)
- tensorrt (for optimized inference)
- scikit-learn (for advanced active learning)

## License

This model builds upon pretrained models and libraries with their respective licenses:
- Wav2Vec2: Facebook Research License
- EfficientNet: Apache 2.0
- Swin Transformer: MIT License
- Mediapipe: Apache 2.0