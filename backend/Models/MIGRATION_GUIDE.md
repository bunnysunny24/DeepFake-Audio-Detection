# Migration Guide: Unified Multimodal Deepfake Detection Model

This guide helps you migrate from the separate `multi_modal_model.py` and `multi_modal_model_check_1.py` files to the unified `multi_modal_model_unified.py` model.

## Overview

The unified model combines all features from both original models while maintaining backward compatibility and adding new configuration options.

## Key Changes

### 1. Import Changes

**Before:**
```python
# From multi_modal_model.py
from multi_modal_model import MultiModalDeepfakeModel

# From multi_modal_model_check_1.py
from multi_modal_model_check_1 import MultiModalDeepfakeModel
```

**After:**
```python
# Unified import
from multi_modal_model_unified import (
    MultiModalDeepfakeModel,
    create_unified_model,
    UnifiedModelConfig,
    get_model_info
)
```

### 2. Model Initialization

**Before (multi_modal_model.py):**
```python
model = MultiModalDeepfakeModel(
    num_classes=2,
    video_feature_dim=1024,
    audio_feature_dim=1024,
    transformer_dim=768,
    num_transformer_layers=4,
    enable_face_mesh=True,
    enable_explainability=True,
    fusion_type='attention',
    backbone_visual='efficientnet',
    backbone_audio='wav2vec2',
    use_spectrogram=True,
    detect_deepfake_type=True,
    num_deepfake_types=7,
    debug=False
)
```

**After (Unified):**
```python
# Method 1: Direct initialization (maintains compatibility)
model = MultiModalDeepfakeModel(
    num_classes=2,
    video_feature_dim=1024,
    audio_feature_dim=1024,
    transformer_dim=768,
    num_transformer_layers=4,
    enable_face_mesh=True,
    enable_explainability=True,
    fusion_type='attention',
    backbone_visual='efficientnet',
    backbone_audio='wav2vec2',
    use_spectrogram=True,
    detect_deepfake_type=True,
    num_deepfake_types=7,
    debug=False
)

# Method 2: Using factory function (recommended)
model = create_unified_model(
    num_classes=2,
    video_feature_dim=1024,
    audio_feature_dim=1024,
    transformer_dim=768,
    num_transformer_layers=4,
    enable_face_mesh=True,
    enable_explainability=True,
    fusion_type='attention',
    backbone_visual='efficientnet',
    backbone_audio='wav2vec2',
    use_spectrogram=True,
    detect_deepfake_type=True,
    num_deepfake_types=7,
    debug=False
)
```

**Before (multi_modal_model_check_1.py):**
```python
model = MultiModalDeepfakeModel(
    num_classes=2,
    # ... basic parameters ...
    enable_adversarial_training=True,
    enable_self_supervised=True,
    enable_curriculum_learning=True,
    enable_active_learning=True,
    enable_quantization=True,
    enable_pruning=True,
    enable_real_time_optimization=True,
    enable_ensemble=True
)
```

**After (Unified):**
```python
model = create_unified_model(
    num_classes=2,
    # ... basic parameters ...
    enable_adversarial_training=True,
    enable_self_supervised=True,
    enable_curriculum_learning=True,
    enable_active_learning=True,
    enable_quantization=True,
    enable_pruning=True,
    enable_real_time_optimization=True,
    enable_ensemble=True
)
```

### 3. Configuration Management

**New Feature - Configuration Class:**
```python
# Create configuration object
config = UnifiedModelConfig(
    num_classes=2,
    backbone_visual='efficientnet',
    backbone_audio='wav2vec2',
    enable_adversarial_training=True,
    enable_self_supervised=False,
    adversarial_epsilon=0.02,
    curriculum_progression_epochs=50
)

# Use configuration
model = MultiModalDeepfakeModel(config=config)

# Update configuration
model.update_config(num_classes=3, adversarial_epsilon=0.01)
```

### 4. Feature Management

**New Feature - Dynamic Feature Control:**
```python
# Enable/disable features at runtime
model.enable_feature('adversarial_training')
model.disable_feature('ensemble')

# Get model information
info = get_model_info(model)
print(f"Enabled features: {info['features_enabled']}")
```

## Forward Pass Changes

### Input/Output Compatibility

**Before:**
```python
# Both models used similar input format
output = model(video_frames, audio_waveform, spectrogram)
```

**After:**
```python
# Same input format maintained
output = model(video_frames, audio_waveform, spectrogram)

# Enhanced output dictionary
predictions = output['predictions']
features = output['features']
visual_features = output['visual_features']
audio_features = output['audio_features']
forensic_features = output['forensic_features']
av_sync_score = output['av_sync_score']

# Additional outputs (if enabled)
if 'deepfake_type_predictions' in output:
    deepfake_types = output['deepfake_type_predictions']
    
if 'uncertainty' in output:
    uncertainty = output['uncertainty']
```

## Training Pipeline Migration

### 1. Standard Training

**Before:**
```python
for batch in dataloader:
    video, audio, spec, targets = batch
    output = model(video, audio, spec)
    loss = criterion(output, targets)  # Direct output
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**After:**
```python
for batch in dataloader:
    video, audio, spec, targets = batch
    output = model(video, audio, spec)
    loss = criterion(output['predictions'], targets)  # Use predictions key
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 2. Advanced Training Features

**New - Adversarial Training:**
```python
# Enable adversarial training
model = create_unified_model(enable_adversarial_training=True)

for batch in dataloader:
    video, audio, spec, targets = batch
    # Adversarial training returns clean and adversarial outputs
    clean_output, adv_output = model.adversarial_training(
        torch.cat([video, audio, spec], dim=1), 
        targets, 
        training=True
    )
    loss = criterion(clean_output, targets) + criterion(adv_output, targets)
```

**New - Curriculum Learning:**
```python
# Enable curriculum learning
model = create_unified_model(enable_curriculum_learning=True)

for epoch in range(num_epochs):
    for batch in dataloader:
        video, audio, spec, targets = batch
        # Progressive training considers epoch for difficulty
        output = model.progressive_training(
            torch.cat([video, audio, spec], dim=1), 
            targets, 
            epoch=epoch
        )
```

## Component Migration

### 1. Shared Components

These components exist in both models and are unified:

- `AttentionFusion` → Enhanced with better initialization
- `TemporalAttention` → Improved with configurable heads
- `StatsPooling` → Unchanged
- `ForensicConsistencyModule` → Enhanced with more features
- `AudioVisualSyncDetector` → Improved architecture

### 2. New Components (from check_1)

These are new additions from the enhanced model:

- `AdversarialNoise` and `AdversarialTraining`
- `SelfSupervisedPretrainer` and `MaskedAutoencoderPretraining`
- `CurriculumLearningScheduler` and `ProgressiveTraining`
- `ActiveLearningSelector`
- `ModelQuantization`, `ModelPruning`, `KnowledgeDistillation`
- `SlidingWindowInference`, `FrameBufferManager`, `AdaptiveResolutionScaling`
- `MultiHeadEnsemble`, `BayesianUncertaintyEstimation`

### 3. Deprecated Components

None - all components from both models are preserved.

## File Structure Changes

### Before
```
backend/Models/
├── multi_modal_model.py           # Basic model
├── multi_modal_model_check_1.py   # Enhanced model
└── ...
```

### After
```
backend/Models/
├── multi_modal_model_unified.py         # Unified model
├── UNIFIED_MODEL_DOCUMENTATION.md       # Documentation
├── MIGRATION_GUIDE.md                   # This file
├── test_unified_model_simple.py         # Component tests
├── multi_modal_model.py                 # Original (can be kept for reference)
├── multi_modal_model_check_1.py         # Original (can be kept for reference)
└── ...
```

## Breaking Changes

### 1. Output Format

**Breaking Change:** Model output is now always a dictionary.

**Before:**
```python
predictions = model(video, audio, spec)  # Direct tensor
```

**After:**
```python
output = model(video, audio, spec)
predictions = output['predictions']  # Dictionary access
```

### 2. Configuration Parameter Names

Some parameters have been renamed for clarity:

| Old Name | New Name |
|----------|----------|
| `debug` | `debug` (unchanged) |
| `enable_explainability` | `enable_explainability` (unchanged) |

## Step-by-Step Migration

### Step 1: Update Imports
Replace all imports from the separate models with the unified model import.

### Step 2: Update Model Creation
Use the `create_unified_model` factory function or update constructor calls.

### Step 3: Update Forward Pass
Change direct output usage to dictionary access: `output['predictions']`.

### Step 4: Update Training Loop
Modify training loops to use the new output format.

### Step 5: Test Functionality
Run tests to ensure all functionality works as expected.

### Step 6: Enable New Features
Gradually enable new features like adversarial training or curriculum learning.

## Testing Migration

### 1. Basic Functionality Test
```python
# Test basic model creation and forward pass
model = create_unified_model(num_classes=2)
video = torch.randn(1, 8, 3, 224, 224)
audio = torch.randn(1, 16000)
spec = torch.randn(1, 1, 128, 128)

output = model(video, audio, spec)
assert 'predictions' in output
assert output['predictions'].shape == (1, 2)
```

### 2. Advanced Features Test
```python
# Test advanced features
model = create_unified_model(
    num_classes=2,
    enable_adversarial_training=True,
    enable_curriculum_learning=True
)
# Test forward pass and training
```

### 3. Configuration Test
```python
# Test configuration management
config = UnifiedModelConfig(num_classes=3)
model = MultiModalDeepfakeModel(config=config)
assert model.get_config().num_classes == 3
```

## Performance Considerations

### 1. Memory Usage
The unified model may use more memory due to additional components. Disable unused features:

```python
# Minimal configuration for memory efficiency
model = create_unified_model(
    num_classes=2,
    enable_adversarial_training=False,
    enable_self_supervised=False,
    enable_ensemble=False
)
```

### 2. Training Speed
Some features like adversarial training and self-supervised learning may slow down training. Enable progressively:

```python
# Start with basic training
model = create_unified_model(num_classes=2)

# Later enable advanced features
model.enable_feature('adversarial_training')
```

### 3. Inference Speed
For production deployment, use optimization features:

```python
# Optimized model for inference
model = create_unified_model(
    num_classes=2,
    enable_quantization=True,
    enable_pruning=True,
    enable_real_time_optimization=True
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch size or disable unused features
3. **Training Instability**: Start with basic configuration and add features gradually
4. **Slow Inference**: Enable quantization and pruning

### Debug Mode
Enable debug mode for detailed logging:
```python
model = create_unified_model(debug=True)
```

## Support

For issues or questions regarding migration:
1. Check the test files for usage examples
2. Review the documentation for configuration options
3. Start with basic configuration and add features incrementally
4. Use debug mode to identify issues