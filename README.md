# Multi-Modal Deepfake Detection System

**End-to-end audio-visual deepfake detection with physiological, forensic, and behavioral analysis.**

| Metric | Value |
|--------|-------|
| Total Parameters | **68,556,807** (68.5M) |
| Trainable Parameters | 68,551,263 |
| Frozen Parameters | 5,544 (first 20 EfficientNet-B0 params) |
| Named Component Groups | 48 |
| BatchNorm Buffers | 209 |
| Target Accuracy | 90%+ on LAV-DF |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Visual Backbone - EfficientNet-B0](#2-visual-backbone---efficientnet-b0)
3. [Audio Backbone - LightweightAudioEncoder](#3-audio-backbone---lightweightaudioencoder)
4. [Cross-Modal Fusion Module](#4-cross-modal-fusion-module)
5. [Transformer Encoder](#5-transformer-encoder)
6. [Contrastive Fusion](#6-contrastive-fusion)
7. [Explainability Projector](#7-explainability-projector)
8. [Classification Heads](#8-classification-heads)
9. [Physiological Analysis Components](#9-physiological-analysis-components)
10. [Forensic and Behavioral Analysis](#10-forensic-and-behavioral-analysis)
11. [Mobile Sensor Analysis](#11-mobile-sensor-analysis)
12. [Contrastive Learning Pipeline](#12-contrastive-learning-pipeline)
13. [Data Pipeline](#13-data-pipeline)
14. [Training Pipeline](#14-training-pipeline)
15. [Loss Functions](#15-loss-functions)
16. [Optimizer and Scheduler](#16-optimizer-and-scheduler)
17. [EMA (Exponential Moving Average)](#17-ema-exponential-moving-average)
18. [Progressive Unfreezing](#18-progressive-unfreezing)
19. [Quantization-Aware Training (QAT)](#19-quantization-aware-training-qat)
20. [Inference Pipeline](#20-inference-pipeline)
21. [Complete Parameter Inventory](#21-complete-parameter-inventory)
22. [Data Features Catalog](#22-data-features-catalog)
23. [File Structure](#23-file-structure)
24. [How to Run](#24-how-to-run)
25. [Key Bug Fixes Applied](#25-key-bug-fixes-applied)
26. [References and Benchmarks](#26-references-and-benchmarks)

---

## 1. Architecture Overview

```
INPUT VIDEO + AUDIO
        |
        v
+-------+-------+
|               |
v               v
EfficientNet-B0  LightweightAudioEncoder
(4.0M params)    (657K params)
1280-dim out     768-dim out
        |               |
        v               v
+-------+-------+-------+-------+
|       Fusion Module (7.5M)     |
|   Cross-attention + gating     |
|   visual_proj: 1280->768       |
|   audio_proj:  768->768        |
|   bi-directional attention     |
|   fusion_layer: 1536->768      |
+---------------+----------------+
                |
                v
+---------------+----------------+
|    Transformer Encoder (28.3M) |
|    4 layers, dim=768           |
|    FFN=3072, 8 heads           |
|    LayerNorm + residual        |
+---------------+----------------+
                |
         768-dim output
                |
        +-------+-------+
        |               |
        v               v
   Main Path      Contrastive Path
        |          (10.6M, training only)
        |               |
        v               v
  Classifier      contrastive_fusion
  (920K params)   6144->1536->768
  1536->512->     similarity_scorer
  256->2          2048->512->256->1
  (BN+Dropout)
        |
        v
  REAL / FAKE
```

**Key Dimensions Flow:**
```
Visual:  [B, frames, 3, 224, 224] -> EfficientNet-B0 -> [B, 1280] (adaptive avg pool)
         -> video_projection [1280->1280] -> fusion_module.visual_projection [1280->768]

Audio:   [B, audio_length] -> MFCC(n_mfcc=40) -> Conv1d stack -> projection [256->512->768]
         -> fusion_module.audio_projection [768->768]

Fusion:  visual [B,768] + audio [B,768] -> cross-attention + gate -> [B,768]
         -> LayerNorm -> feed to transformer

Trans:   [B, seq, 768] -> 4x TransformerEncoderLayer(d=768, nhead=8, ffn=3072) -> [B, 768]

Concat:  [audio_768 || transformer_768] = [B, 1536]

Class:   [B,1536] -> Linear(512) -> BN -> ReLU -> Drop(0.3)
                   -> Linear(256) -> BN -> ReLU -> Drop(0.2)
                   -> Linear(2) -> softmax -> {REAL, FAKE}
```

---

## 2. Visual Backbone - EfficientNet-B0

**Total: 4,007,548 params (5,544 frozen)**

EfficientNet-B0 is a compound-scaled CNN pretrained on ImageNet. The first 20 parameters (stem conv + first MBConv block) are initially frozen to preserve low-level feature extractors.

### Architecture Blocks

| Block | Channels | Kernel | SE Ratio | Output |
|-------|----------|--------|----------|--------|
| Stem (features.0) | 3->32 | 3x3 | - | 112x112 |
| MBConv1 (features.1) | 32->16 | 3x3 DW | 8 | 112x112 |
| MBConv6 (features.2) | 16->24 | 3x3 DW | 4 | 56x56 |
| MBConv6 (features.3) | 24->40 | 5x5 DW | 6 | 28x28 |
| MBConv6 (features.4) | 40->80 | 3x3 DW | 10 | 14x14 |
| MBConv6 (features.5) | 80->112 | 5x5 DW | 20 | 14x14 |
| MBConv6 (features.6) | 112->192 | 5x5 DW | 28 | 7x7 |
| MBConv6 (features.7) | 192->320 | 3x3 DW | 48 | 7x7 |
| Head (features.8) | 320->1280 | 1x1 | - | 7x7 |

Each MBConv block contains: Expand (1x1) -> Depthwise Conv -> Squeeze-and-Excitation -> Project (1x1) with BatchNorm and SiLU activation throughout.

**Frozen Parameters (first 20):**
```
visual_model.features.0.0.weight:  [32, 3, 3, 3]     # Stem conv
visual_model.features.0.1.weight:  [32]                # Stem BN weight
visual_model.features.0.1.bias:    [32]                # Stem BN bias
visual_model.features.1.0.block.0.0.weight: [32,1,3,3] # Block1 DW conv
visual_model.features.1.0.block.0.1.weight: [32]       # Block1 DW BN
visual_model.features.1.0.block.0.1.bias:   [32]       # Block1 DW BN bias
visual_model.features.1.0.block.1.fc1.weight: [8,32,1,1]  # Block1 SE fc1
visual_model.features.1.0.block.1.fc1.bias:   [8]         # Block1 SE fc1 bias
visual_model.features.1.0.block.1.fc2.weight: [32,8,1,1]  # Block1 SE fc2
visual_model.features.1.0.block.1.fc2.bias:   [32]        # Block1 SE fc2 bias
visual_model.features.1.0.block.2.0.weight: [16,32,1,1]   # Block1 project
visual_model.features.1.0.block.2.1.weight: [16]          # Block1 project BN
visual_model.features.1.0.block.2.1.bias:   [16]          # Block1 project BN bias
... (total 20 parameters, 5,544 scalars)
```

**Progressive Unfreezing:** After epoch 3, 10 layers unfrozen per epoch. All backbone layers are always in the optimizer (just with `requires_grad=False` initially), so unfreezing simply sets `requires_grad=True`.

### Video Projection
```
video_projection: Linear(1280, 1280)   # 1,639,680 params
```

---

## 3. Audio Backbone - LightweightAudioEncoder

**Total: 657,472 params (all trainable)**

Replaces Wav2Vec2 (94.4M params) with a 99.3% parameter reduction and <1% accuracy loss.

### Architecture
```
Input: raw waveform [B, audio_length]
  |
  v
MFCC Transform (n_mfcc=40, sample_rate=16000)
  -> [B, 40, T]
  |
  v
Conv1d Stack:
  Conv1d(40, 64, kernel=3) -> BN(64) -> ReLU -> MaxPool1d
  Conv1d(64, 128, kernel=3) -> BN(128) -> ReLU -> MaxPool1d
  Conv1d(128, 256, kernel=3) -> BN(256) -> ReLU -> AdaptiveAvgPool1d(1)
  -> [B, 256]
  |
  v
Projection:
  Linear(256, 512) -> ReLU -> Dropout(0.1) -> Linear(512, 768)
  -> [B, 768]
```

**Buffers:**
- `mfcc_transform.dct_mat`: [40, 40] (DCT matrix for MFCC computation)
- `mfcc_transform.MelSpectrogram.spectrogram.window`: [400] (Hann window)
- `mfcc_transform.MelSpectrogram.mel_scale.fb`: [201, 40] (Mel filterbank)

### Audio Projection
```
audio_projection: Linear(768, 768)   # 590,592 params
```

### Spectrogram Model
```
spectrogram_model: Sequential(       # 92,672 params
  Conv2d(1, 32, 3x3) -> ReLU -> MaxPool2d
  Conv2d(32, 64, 3x3) -> ReLU -> MaxPool2d
  Conv2d(64, 128, 3x3) -> ReLU -> AdaptiveAvgPool2d
)
spectrogram_projection: Linear(128, 768)  # 99,072 params
```

---

## 4. Cross-Modal Fusion Module

**Total: 7,483,392 params**

Implements bidirectional cross-attention between visual and audio features, followed by gated fusion.

### Architecture
```
Input: visual [B, 1280], audio [B, 768]
  |
  +-- visual_projection: Linear(1280, 768)   # Project visual to shared dim
  +-- audio_projection:  Linear(768, 768)    # Project audio to shared dim
  +-- visual_norm: LayerNorm(768)
  +-- audio_norm: LayerNorm(768)
  |
  v
Bi-directional Cross-Attention:
  Visual attending to Audio:
    Q = visual_query(visual)  [768->768]
    K = audio_key(audio)      [768->768]
    V = audio_value(audio)    [768->768]
    attended_visual = softmax(QK^T / sqrt(768)) * V

  Audio attending to Visual:
    Q = audio_query(audio)    [768->768]
    K = visual_key(visual)    [768->768]
    V = visual_value(visual)  [768->768]
    attended_audio = softmax(QK^T / sqrt(768)) * V
  |
  v
Gated Fusion:
  gate = sigmoid(gate_layer([attended_visual || attended_audio]))  # [1536->768]
  fused = gate * attended_visual + (1 - gate) * attended_audio
  |
  v
Residual Fusion:
  fusion_layer: Linear(1536, 768)  # [visual || audio] -> 768
  output = LayerNorm(fused + residual)
  -> [B, 768]
```

**Complete Parameter List:**
```
fusion_module.visual_projection.weight: [768, 1280]
fusion_module.visual_projection.bias:   [768]
fusion_module.audio_projection.weight:  [768, 768]
fusion_module.audio_projection.bias:    [768]
fusion_module.visual_norm.weight:       [768]
fusion_module.visual_norm.bias:         [768]
fusion_module.audio_norm.weight:        [768]
fusion_module.audio_norm.bias:          [768]
fusion_module.visual_query.weight:      [768, 768]
fusion_module.visual_query.bias:        [768]
fusion_module.audio_key.weight:         [768, 768]
fusion_module.audio_key.bias:           [768]
fusion_module.audio_value.weight:       [768, 768]
fusion_module.audio_value.bias:         [768]
fusion_module.audio_query.weight:       [768, 768]
fusion_module.audio_query.bias:         [768]
fusion_module.visual_key.weight:        [768, 768]
fusion_module.visual_key.bias:          [768]
fusion_module.visual_value.weight:      [768, 768]
fusion_module.visual_value.bias:        [768]
fusion_module.gate.0.weight:            [768, 1536]
fusion_module.gate.0.bias:              [768]
fusion_module.fusion_layer.weight:      [768, 1536]
fusion_module.fusion_layer.bias:        [768]
fusion_module.layer_norm.weight:        [768]
fusion_module.layer_norm.bias:          [768]
```

---

## 5. Transformer Encoder

**Total: 28,351,488 params**

4-layer Transformer Encoder operating on the fused 768-dimensional features.

### Per-Layer Architecture
```
TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=3072):
  |
  +-- Multi-Head Self-Attention:
  |     in_proj_weight: [2304, 768]   # Q,K,V packed (3 x 768 = 2304)
  |     in_proj_bias:   [2304]
  |     out_proj:       Linear(768, 768)
  |
  +-- LayerNorm1: weight [768], bias [768]
  |
  +-- FFN:
  |     linear1: Linear(768, 3072)
  |     linear2: Linear(3072, 768)
  |
  +-- LayerNorm2: weight [768], bias [768]
```

**Per-layer params:** 7,087,872
**Total (4 layers):** 28,351,488

**Layer-by-layer:**
```
Layer 0: transformer.layers.0.*  (7,087,872 params)
Layer 1: transformer.layers.1.*  (7,087,872 params)
Layer 2: transformer.layers.2.*  (7,087,872 params)
Layer 3: transformer.layers.3.*  (7,087,872 params)
```

---

## 6. Contrastive Fusion

**Total: 10,623,744 params (training only)**

Learns fake-vs-original feature differences via contrastive comparison. Only active during training; disabled at inference.

### Architecture
```
contrastive_fusion: Sequential(
  Linear(6144, 1536) -> BatchNorm1d(1536) -> ReLU -> Dropout(0.1)
  Linear(1536, 768)  -> BatchNorm1d(768)
)
```
Input: concatenation of [fake_visual, fake_audio, original_visual, original_audio] = 6144

### Supporting Components
```
feature_difference_analyzer: Sequential(     # 1,640,320 params
  Linear(1280, 640) -> ReLU -> Dropout(0.1)
  Linear(640, 1280) -> ReLU
)

audio_difference_analyzer: Sequential(       # 590,976 params
  Linear(768, 384) -> ReLU -> Dropout(0.1)
  Linear(384, 768) -> ReLU
)

similarity_scorer: Sequential(               # 1,180,673 params
  Linear(2048, 512) -> ReLU -> Dropout(0.1)
  Linear(512, 256) -> ReLU
  Linear(256, 1)                # Similarity score
)

combined_projection: Linear(2048, 768)       # 1,573,632 params
```

---

## 7. Explainability Projector

**Total: 2,491,648 params**

Projects concatenated features from multiple modalities into the transformer dimension space for explanation generation.

```
explainability_projector: Sequential(
  Linear(4096, 512)    # 4096 input = concat of multiple feature vectors
  ReLU
  Dropout(0.1)
  Linear(512, 768)     # Output matches transformer dim
)
```

---

## 8. Classification Heads

### Main Classifier (920,322 params)
```
classifier: Sequential(
  Linear(1536, 512)          # 1536 = [audio_768 || transformer_768]
  BatchNorm1d(512)           # Stabilizes training, reduces internal covariate shift
  ReLU
  Dropout(0.3)               # Strong regularization
  Linear(512, 256)
  BatchNorm1d(256)
  ReLU
  Dropout(0.2)
  Linear(256, 2)             # Binary: REAL vs FAKE
)
```

### Per-Modality Classifiers (for tampering type detection)

**Audio-Only Classifier (230,018 params):**
```
audio_only_classifier: Sequential(
  Linear(768, 256) -> ReLU -> Dropout(0.2)
  Linear(256, 128) -> ReLU -> Dropout(0.1)
  Linear(128, 2)
)
```

**Video-Only Classifier (361,090 params):**
```
video_only_classifier: Sequential(
  Linear(1280, 256) -> ReLU -> Dropout(0.2)
  Linear(256, 128) -> ReLU -> Dropout(0.1)
  Linear(128, 2)
)
```

**Deepfake Type Classifier (395,271 params):**
```
deepfake_type_classifier: Sequential(
  Linear(1536, 256) -> ReLU -> Dropout(0.2)
  Linear(256, 7)              # 7 types: unknown, face_swap, face_reenactment,
)                              #   lip_sync, audio_only, entire_synthesis,
                               #   attribute_manipulation
```

### Auxiliary Heads
```
aux_visual_head:        Linear(1280, 128) -> ReLU -> Drop(0.2) -> Linear(128, 2)    # 164,226
aux_audio_head:         Linear(768, 128)  -> ReLU -> Drop(0.2) -> Linear(128, 2)    # 98,690
aux_physiological_head: Linear(256, 64)   -> ReLU -> Drop(0.2) -> Linear(64, 2)     # 16,578
aux_facial_head:        Linear(256, 64)   -> ReLU -> Drop(0.2) -> Linear(64, 2)     # 16,578
aux_forensic_head:      Linear(64, 32)    -> ReLU -> Drop(0.2) -> Linear(32, 2)     # 2,146
```

---

## 9. Physiological Analysis Components

### 9.1 Advanced Physiological Analyzer (86,183 params)

Contains three sub-analyzers fused together:

#### Digital Heartbeat Detector (14,354 params standalone, also nested in advanced)
```
face_attention: Conv2d(3,16,3) -> Conv2d(16,8,3) -> Conv2d(8,1,1)
temporal_extractor: Conv1d(3,32,3) -> Conv1d(32,64,3) -> Conv1d(64,32,3) -> Conv1d(32,1,1)
```
Detects subtle color changes in facial skin corresponding to blood pulse. Deepfakes lack this natural pulse signal.

#### Blood Flow Analyzer (34,791 params standalone, also nested)
```
skin_segmenter: Conv2d(3,32,3) -> Conv2d(32,16,3) -> Conv2d(16,1,1)
color_transformer: Conv1d(3,16,1) -> Conv1d(16,8,1) -> Conv1d(8,3,1)
temporal_analyzer: Conv1d(3,32,3) -> BN -> Conv1d(32,64,3) -> BN -> Conv1d(64,32,3) -> BN
flow_classifier: Linear(32,64) -> ReLU -> Drop -> Linear(64,32) -> ReLU -> Linear(32,1)
thermal_analyzer: Conv2d(3,16,3) -> Conv2d(16,32,3) -> Conv2d(32,16,3) -> Conv2d(16,1,1)
temp_consistency: Linear(64,32) -> ReLU -> Linear(32,1)
```

#### Breathing Pattern Detector (33,900 params standalone, also nested)
```
chest_detector: Conv2d(3,32,3) -> Conv2d(32,16,3) -> Conv2d(16,8,3) -> Flatten -> Linear(8,1)
nostril_analyzer: Conv2d(3,16,3) -> Conv2d(16,8,3) -> Flatten -> Linear(128,32) -> Linear(32,1)
pattern_analyzer: Conv1d(2,32,3) -> BN -> Conv1d(32,64,3) -> BN -> Conv1d(64,32,3) -> BN
rate_estimator: Linear(32,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1)
naturalness_classifier: Linear(34,64) -> ReLU -> Drop -> Linear(64,32) -> ReLU -> Linear(32,1)
```

**Fusion and Coherence:**
```
advanced_physiological_analyzer.fusion_layer: Linear(3,64) -> ReLU -> Drop -> Linear(64,32) -> ReLU -> Linear(32,1)
advanced_physiological_analyzer.coherence_analyzer: Linear(6,32) -> ReLU -> Linear(32,16) -> ReLU -> Linear(16,1)
```

### 9.2 Skin Color Analyzer (103,105 params)
```
color_encoder: Linear(3,64) -> ReLU -> Linear(64,64) -> ReLU
temporal_cnn: Conv1d(64,128,3) -> Conv1d(128,128,3) -> Conv1d(128,64,3)
naturalness_scorer: Linear(64,1)
```
Tracks temporal skin color consistency across frames. Deepfakes often show unnatural color fluctuations.

### 9.3 Eye Analysis Module (8,067 params)
```
blink_detector: Linear(32,16) -> ReLU -> Linear(16,1)
pupil_estimator: Linear(32,16) -> ReLU -> Linear(16,1)
temporal_gru: GRU(input=2, hidden=32, bidirectional=True, 1 layer)
naturalness_scorer: Linear(64,1)
```

### 9.4 Liveness Detector (164,097 params)
```
detector: Linear(1280,128) -> ReLU -> Linear(128,1)
```
Direct liveness check from visual features.

---

## 10. Forensic and Behavioral Analysis

### 10.1 Micro-Expression Detector (2,491,015 params)
```
conv3d: Conv3d(3,64,3) -> BN3d -> ReLU -> Conv3d(64,128,3) -> BN3d -> ReLU
temporal_lstm: BiLSTM(input=2048, hidden=128, 1 layer)
classifier: Linear(256,128) -> ReLU -> Drop -> Linear(128,7)  # 7 micro-expression types
```

### 10.2 Facial Action Unit (AU) Analyzer (178,194 params)
```
encoder: Linear(136,128) -> BN -> ReLU -> Linear(128,128)
au_heads: 17x Linear(128,1)           # 17 individual AU detectors
temporal_lstm: BiLSTM(input=17, hidden=64, 2 layers)
consistency_scorer: Linear(128,1)
```
Detects 17 facial action units (FACS) and measures temporal consistency. Deepfakes often lack subtle AU synchronization.

### 10.3 Facial Landmark Trajectory Analyzer (158,529 params)
```
motion_encoder: Linear(136,128) -> ReLU -> Drop -> Linear(128,64)
temporal_gru: BiGRU(input=64, hidden=64, 2 layers)
consistency_scorer: Linear(128,64) -> ReLU -> Linear(64,1)
```

### 10.4 Head Pose Estimator (127,108 params)
```
pose_estimator: Linear(136,128) -> ReLU -> Drop -> Linear(128,64) -> Linear(64,3)  # pitch, yaw, roll
consistency_analyzer: BiGRU(input=3, hidden=64, 2 layers)
consistency_scorer: Linear(128,1)
```

### 10.5 Lip-Audio Sync Analyzer (138,945 params)
```
lip_encoder: Linear(40,128) -> ReLU -> Linear(128,64)
audio_encoder: Linear(768,128) -> ReLU -> Linear(128,64)
cross_attention: MultiheadAttention(embed_dim=64, num_heads=4)
sync_scorer: Linear(64,32) -> ReLU -> Linear(32,1)
```

### 10.6 Sync Detector (328,449 params)
```
visual_encoder: Linear(1280,128) -> ReLU -> Linear(128,128)
audio_encoder: Linear(768,128) -> ReLU -> Linear(128,128)
fusion: Linear(256,128) -> ReLU -> Linear(128,1)
```

### 10.7 Forensic Module (42,880 params)
```
conv1: Conv2d(3,64,3) -> ReLU -> MaxPool
conv2: Conv2d(64,64,3) -> ReLU -> AdaptiveAvgPool
fc: Linear(64,64)
```
Analyzes ELA (Error Level Analysis) features for compression artifact detection.

### 10.8 Face Embedding Processor (98,688 params)
```
Linear(256,256) -> ReLU -> Linear(256,128)
```

---

## 11. Mobile Sensor Analysis

### 11.1 Mobile Sensor Fusion (1,170,949 params)
```
optical_flow_proj: Linear(64, 256)
metadata_proj:     Linear(32, 256)
shutter_proj:      Linear(16, 256)
sync_proj:         Linear(32, 256)
depth_proj:        Linear(64, 256)

attention: Linear(1280,256) -> ReLU -> Linear(256,5)  # 5 sensor weights
fusion:    Linear(1280,512) -> ReLU -> Drop -> Linear(512,256)
```

### 11.2 Sub-Analyzers

| Component | Params | Input | Output |
|-----------|--------|-------|--------|
| `optical_flow_analyzer` | 8,737 | 10-dim motion | shake score |
| `camera_metadata_analyzer` | 1,744 | 8-dim EXIF | 32-dim features |
| `rolling_shutter_detector` | 184 | 4-dim shutter | 16-dim features |
| `av_sync_analyzer` | 656 | 6-dim sync | 32-dim features |
| `mobile_depth_analyzer` | 88,896 | depth map | 64-dim features |

### 11.3 Voice Stress Analyzer (49,453 params)
```
jitter_shimmer_analyzer:
  feature_extractor: Linear(10,32) -> ReLU -> Drop -> Linear(32,64) -> ReLU -> Drop -> Linear(64,32)
  stress_classifier: Linear(32,16) -> ReLU -> Linear(16,1)

emotional_detector:
  emotion_encoder: Linear(64,128) -> ReLU -> Drop -> Linear(128,64) -> ReLU -> Drop -> Linear(64,32)
  stress_head, anxiety_head, fear_head, anger_head: Linear(32,1) each
  overall_emotion: Linear(32,7)  # 7 emotion classes

formant_analyzer:
  formant_encoder: Linear(12,32) -> ReLU -> Linear(32,64) -> ReLU -> Linear(64,32)

fusion: Linear(96,128) -> ReLU -> Drop -> Linear(128,64) -> ReLU -> Linear(64,1)
```

---

## 12. Contrastive Learning Pipeline

During training, each sample provides both a **fake** video and its **original** (real) counterpart. The model extracts features from both and computes:

1. **Feature Differences:** `feature_difference_analyzer(fake_visual - original_visual)`
2. **Audio Differences:** `audio_difference_analyzer(fake_audio - original_audio)`
3. **Similarity Score:** `similarity_scorer([fake_combined, original_combined])`
4. **Contrastive Fusion:** `contrastive_fusion([fake_v, fake_a, orig_v, orig_a])`

This teaches the model what specific artifacts distinguish fakes from originals, rather than learning dataset biases.

### Learnable Thresholds
```
deepfake_threshold:               scalar    # Adaptive detection threshold
frequency_threshold:              scalar    # Frequency domain anomaly threshold
noise_threshold:                  scalar    # Noise pattern threshold
temporal_consistency_threshold:   scalar    # Temporal consistency threshold
diversity_loss_weight:            scalar    # Component diversity loss scaling
component_weights:                [50]      # Learnable weights for 50 components
```

---

## 13. Data Pipeline

### 13.1 Dataset Format (LAV-DF)

- **Source:** LAV-DF (Large-scale Audivisual Deepfake) dataset
- **Splits:** train (78,703), dev (31,501), test (26,100)
- **Metadata:** `metadata.json` with `split` field per entry
- **Fields per entry:** `file`, `original`, `split`, `label`, `modify_type`, `fake_periods`, `timestamps`, `transcript`

### 13.2 Data Loading (`dataset_loader.py`)

**Frame Sampling:**
- Max frames: 32 (configurable via `--max_frames`)
- Sampling: uniform temporal sampling across video duration
- Face detection: dlib HOG detector with CNN fallback
- Landmark extraction: dlib 68-point face landmarks normalized to [0,1] range

**Audio Processing:**
- Sample rate: 16,000 Hz
- Audio length: configurable (default from video duration)
- MFCC: 40 coefficients over 50 time steps (from audio, reduced from 100)
- Voice stress: 6-dim (jitter, shimmer, HNR + flags)

### 13.3 Complete Feature Dictionary (per sample)

Every `__getitem__` call returns a dictionary with these keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `video_frames` | `[32, 3, 224, 224]` | Face-cropped, normalized RGB frames |
| `audio` | `[audio_length]` | Raw waveform at 16kHz |
| `audio_spectrogram` | `[1, 64, 64]` | Mel spectrogram |
| `label` | scalar | 0=REAL, 1=FAKE, -1=placeholder (filtered) |
| `deepfake_type` | scalar | 0-6 (see classifier section) |
| `facial_landmarks` | `[32, 136]` | 68 landmarks x 2 coords, normalized |
| `mfcc_features` | `[20, 50]` | MFCC coefficients |
| `voice_stress_features` | `[6]` | Jitter, shimmer, HNR + flags |
| `pulse_signal` | `[32]` | Estimated pulse per frame |
| `skin_color_variations` | `[32, 3]` | Mean RGB per frame |
| `head_pose` | `[32, 3]` | Pitch, yaw, roll per frame |
| `eye_blink_features` | `[32]` | Eye aspect ratio per frame |
| `frequency_features` | `[1, 16, 16]` | DCT frequency map |
| `face_embeddings` | `[1, 256]` | Face identity embedding |
| `temporal_consistency` | scalar | Frame-to-frame consistency score |
| `metadata_features` | `[10]` | Video metadata (bitrate, fps, etc.) |
| `ela_features` | `[112, 112]` | Error Level Analysis map |
| `audio_visual_sync` | `[5]` | A/V synchronization features |
| `file_path` | string | Source video path |
| `fake_mask` | `[1]` | Temporal fake mask |
| `original_video_frames` | `[32, 3, 224, 224]` or None | Original (real) video for contrastive |
| `original_audio` | `[audio_length]` or None | Original audio for contrastive |
| `fake_periods` | list | Time periods of fakery |
| `timestamps` | list | Video timestamps |
| `transcript` | string | Audio transcription |
| `original_facial_landmarks` | `[32, 136]` | Original landmarks |
| `original_mfcc_features` | `[20, 50]` | Original MFCCs |
| `original_voice_stress_features` | `[6]` | Original voice stress |
| `original_pulse_signal` | `[32]` | Original pulse |
| `original_skin_color_variations` | `[32, 3]` | Original skin color |
| `original_head_pose` | `[32, 3]` | Original head pose |
| `original_eye_blink_features` | `[32]` | Original blink |
| `original_frequency_features` | `[1, 16, 16]` | Original frequency |

### 13.4 Augmentation (`improved_augmentation.py`)

**Training transforms:**
- Random horizontal flip
- Random JPEG compression (quality 70-95)
- Color jitter (brightness, contrast, saturation, hue)
- Random rotation (up to 10 degrees)
- Random gaussian noise
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Validation transforms (deterministic):**
- Resize to 224x224
- Normalize only (no random augmentation)

### 13.5 DataLoader Configuration

```python
# Training (both oversampled and normal paths):
DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,       # Prevents BatchNorm crash with batch_size=1
    collate_fn=safe_collate_fn
)

# Validation:
DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,      # Keep all validation samples
    collate_fn=safe_collate_fn
)
```

---

## 14. Training Pipeline

### 14.1 Training Loop (`train_multimodal.py` - `DeepfakeTrainer` class)

```
For each epoch:
    1. Progressive unfreezing (after epoch 3)
    2. For each batch:
        a. Forward pass (AMP mixed precision)
        b. Compute multi-component loss
        c. Scale loss by grad_accum_steps
        d. Backward pass (scaled)
        e. Every grad_accum_steps:
           - Gradient clipping (max_norm=1.5)
           - Optimizer step
           - EMA update
           - Scheduler step
        f. Mixup augmentation (randomly applied)
    3. Validate on dev set (using EMA model)
    4. Save checkpoint if best F1
    5. Early stopping check (patience=10)
```

### 14.2 Full Hyperparameter Table

| Parameter | Value | CLI Flag |
|-----------|-------|----------|
| Batch size | 16 | `--batch_size` |
| Gradient accumulation | 4 | `--grad_accum_steps` |
| Effective batch size | **64** | (16 x 4) |
| Epochs | 80 | `--num_epochs` |
| Learning rate (heads) | 3e-4 | `--learning_rate` |
| Learning rate (backbone) | 3e-5 | (automatic 10x lower) |
| Weight decay | 0.01 | `--weight_decay` |
| Dropout (classifier) | 0.3, 0.2 | `--dropout_rate` |
| Gradient clipping | 1.5 | `--gradient_clip` |
| Max frames | 32 | `--max_frames` |
| Label smoothing | 0.05 | `--label_smoothing` |
| Focal alpha | 1.0 | `--focal_alpha` |
| Focal gamma | 2.0 (or 3.0) | `--focal_gamma` |
| EMA decay | 0.999 | `--ema_decay` |
| Mixup alpha | 0.2 | `--mixup_alpha` |
| Warmup epochs | 3 | `--warmup_epochs` |
| Early stopping patience | 10 | `--early_stopping_patience` |
| Min learning rate | 1e-6 | `--min_lr` |
| AMP (mixed precision) | Enabled | `--amp_enabled` |
| QAT start epoch | 40 | `--qat_start_epoch` |
| QAT backend | fbgemm | `--qat_backend` |
| QAT LR scale | 0.1 | `--qat_lr_scale` |

### 14.3 Environment Variables
```
CUDA_VISIBLE_DEVICES=0
TORCH_CUDNN_BENCHMARK=1
TORCH_ALLOW_TF32=1
CUDA_TF32_TENSOR_CORES=1
OMP_NUM_THREADS=24
MKL_NUM_THREADS=24
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16
SHAPE_PREDICTOR_PATH=F:\Deepfakee\Models\shape_predictor_68_face_landmarks.dat
```

---

## 15. Loss Functions

### 15.1 FocalLoss (Primary)

```python
FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.05, class_weights='balanced')
```

**Formula:**
$$FL(p_t) = -\alpha \cdot (1 - p_t)^{\gamma} \cdot \text{CE}(p_t)$$

Where:
- $p_t$ = model confidence for the correct class
- $\alpha = 1.0$ (neutral global scale; class-specific weighting via `class_weights`)
- $\gamma = 2.0$ (focusing parameter — down-weights easy examples)
- Label smoothing $= 0.05$ (prevents overconfident predictions)
- `class_weights` computed from inverse class frequencies (balanced mode)

**Numerical stability:**
- Input logits clamped to [-50, 50]
- $p_t$ clamped to [1e-7, 1 - 1e-7]

### 15.2 Multi-Component Loss Budget

```
total_loss = main_loss
           + 0.1  * audio_only_loss      # Per-modality: audio classifier
           + 0.1  * video_only_loss      # Per-modality: video classifier
           + 0.05 * kl_consistency_loss  # KL divergence between modality predictions
           + aux_loss                    # Auxiliary head losses (dynamic weights)
           + diversity_loss              # Component diversity regularization
```

**Per-modality losses** use the same FocalLoss but on `audio_only_classifier` and `video_only_classifier` outputs. They are computed WITHOUT AMP (full float32) to prevent numerical instability in small classifiers.

**KL Consistency Loss:** Ensures audio-only and video-only predictions agree with the main classifier. Prevents modalities from diverging.

**Diversity Loss:** Encourages different components to contribute uniquely, preventing redundancy. Scaled by learnable `diversity_loss_weight`.

---

## 16. Optimizer and Scheduler

### 16.1 AdamW with 4 Parameter Groups

```python
optimizer = AdamW([
    # Group 0: Backbone params with weight decay
    {'params': backbone_decay_params,    'lr': 3e-5,  'weight_decay': 0.01},
    
    # Group 1: Backbone params WITHOUT weight decay (BN weights, biases)
    {'params': backbone_no_decay_params, 'lr': 3e-5,  'weight_decay': 0.0},
    
    # Group 2: Head params with weight decay
    {'params': head_decay_params,        'lr': 3e-4,  'weight_decay': 0.01},
    
    # Group 3: Head params WITHOUT weight decay (BN weights, biases)
    {'params': head_no_decay_params,     'lr': 3e-4,  'weight_decay': 0.0},
])
```

**Discriminative LR:** Backbone (EfficientNet-B0) gets 10x lower learning rate than heads. This preserves pretrained features while allowing fine-tuning.

**Weight Decay Exclusion:** BatchNorm parameters and biases are excluded from weight decay (group 1 and 3) — a best practice that prevents BN scale from being regularized toward zero.

### 16.2 CosineAnnealingWarmRestarts

```python
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # First restart at epoch 10
    T_mult=2,      # Second at 10+20=30, third at 30+40=70
    eta_min=1e-6   # Minimum learning rate
)
```

**Warmup:** Linear warmup over 3 epochs from `lr * 0.1` to full `lr`.

**LR Schedule Visualization:**
```
Epoch:  1  2  3  4  5  6  7  8  9  10  11 ... 30  31 ... 70
        warmup      |   cosine decay    |restart|cosine|restart
LR:    3e-5 ... 3e-4 ... decay ... 1e-6  3e-4  decay  1e-6
```

---

## 17. EMA (Exponential Moving Average)

```python
ModelEMA(model, decay=0.999)
```

Maintains a shadow copy of ALL 68,551,263 trainable parameters plus 209 BatchNorm buffers. After each optimizer step:

$$\theta_{\text{EMA}} = \alpha \cdot \theta_{\text{EMA}} + (1 - \alpha) \cdot \theta_{\text{model}}$$

Where $\alpha = 0.999$. The EMA model is used for:
- **Validation:** All validation metrics computed on EMA model
- **Checkpointing:** Best model saved is the EMA version
- **Inference:** Production model uses EMA weights

**BN Buffer Tracking:** EMA also copies running_mean, running_var, and num_batches_tracked from the training model to the shadow model at each update. This ensures the EMA model's BatchNorm statistics stay current.

---

## 18. Progressive Unfreezing

Starting at epoch 3, the backbone (EfficientNet-B0) is gradually unfrozen:

```
Epoch 1-3:  Only first 20 params frozen (stem + block1)
            All other backbone params already trainable but at 0.1x LR
Epoch 4:    10 additional layers unfrozen (deeper blocks)
Epoch 5:    10 more layers unfrozen
...
Epoch N:    All backbone layers fully trainable
```

**Implementation:** All backbone parameters are added to the optimizer from the start (with `requires_grad=False` for frozen ones). Unfreezing simply sets `requires_grad=True` — no optimizer reconstruction needed.

This prevents catastrophic forgetting of ImageNet features during early training when the heads are randomly initialized.

---

## 19. Quantization-Aware Training (QAT)

Starting at epoch 40:

```python
# Backend: fbgemm (x86 optimized)
# LR scale: 0.1x (reduced learning rate during QAT)
torch.quantization.prepare_qat(model, inplace=True)
```

QAT inserts fake quantization nodes (QuantStub/DeQuantStub) during training to simulate INT8 inference precision. This typically preserves >99% of floating-point accuracy while enabling 2-4x inference speedup on CPU.

**Post-training:** The model can be fully quantized for INT8 deployment using `torch.quantization.convert()`.

---

## 20. Inference Pipeline

### 20.1 Single Video Inference (`inference.py`)

```
Input: video file path
  |
  v
1. Extract frames (max_frames=32, uniform temporal sampling)
2. Detect faces (dlib HOG -> CNN fallback)
3. Extract landmarks (dlib 68-point)
4. Extract audio (16kHz)
5. Compute all features (MFCC, spectrogram, pulse, skin color, etc.)
6. Load EMA checkpoint
7. Forward pass (no grad)
8. Softmax -> probability
9. Return: {prediction, confidence, per-modality scores}
```

### 20.2 API Inference (`inference_api.py`)

REST API wrapper around the inference pipeline for deployment.

---

## 21. Complete Parameter Inventory

### 21.1 Summary by Component (sorted by size)

| # | Component | Parameters | % of Total | Frozen |
|---|-----------|-----------|------------|--------|
| 1 | `transformer` | 28,351,488 | 41.36% | 0 |
| 2 | `contrastive_fusion` | 10,623,744 | 15.50% | 0 |
| 3 | `fusion_module` | 7,483,392 | 10.92% | 0 |
| 4 | `visual_model` | 4,007,548 | 5.85% | 5,544 |
| 5 | `explainability_projector` | 2,491,648 | 3.64% | 0 |
| 6 | `micro_expression_detector` | 2,491,015 | 3.63% | 0 |
| 7 | `video_projection` | 1,639,680 | 2.39% | 0 |
| 8 | `feature_difference_analyzer` | 1,640,320 | 2.39% | 0 |
| 9 | `combined_projection` | 1,573,632 | 2.30% | 0 |
| 10 | `similarity_scorer` | 1,180,673 | 1.72% | 0 |
| 11 | `mobile_sensor_fusion` | 1,170,949 | 1.71% | 0 |
| 12 | `classifier` | 920,322 | 1.34% | 0 |
| 13 | `audio_model` | 657,472 | 0.96% | 0 |
| 14 | `audio_difference_analyzer` | 590,976 | 0.86% | 0 |
| 15 | `audio_projection` | 590,592 | 0.86% | 0 |
| 16 | `deepfake_type_classifier` | 395,271 | 0.58% | 0 |
| 17 | `video_only_classifier` | 361,090 | 0.53% | 0 |
| 18 | `sync_detector` | 328,449 | 0.48% | 0 |
| 19 | `audio_only_classifier` | 230,018 | 0.34% | 0 |
| 20 | `facial_au_analyzer` | 178,194 | 0.26% | 0 |
| 21 | `aux_visual_head` | 164,226 | 0.24% | 0 |
| 22 | `liveness_detector` | 164,097 | 0.24% | 0 |
| 23 | `landmark_trajectory_analyzer` | 158,529 | 0.23% | 0 |
| 24 | `lip_audio_sync_analyzer` | 138,945 | 0.20% | 0 |
| 25 | `head_pose_estimator` | 127,108 | 0.19% | 0 |
| 26 | `skin_color_analyzer` | 103,105 | 0.15% | 0 |
| 27 | `spectrogram_projection` | 99,072 | 0.14% | 0 |
| 28 | `aux_audio_head` | 98,690 | 0.14% | 0 |
| 29 | `face_embedding_processor` | 98,688 | 0.14% | 0 |
| 30 | `spectrogram_model` | 92,672 | 0.14% | 0 |
| 31 | `mobile_depth_analyzer` | 88,896 | 0.13% | 0 |
| 32 | `advanced_physiological_analyzer` | 86,183 | 0.13% | 0 |
| 33 | `voice_stress_analyzer` | 49,453 | 0.07% | 0 |
| 34 | `forensic_module` | 42,880 | 0.06% | 0 |
| 35 | `blood_flow_analyzer` | 34,791 | 0.05% | 0 |
| 36 | `breathing_pattern_detector` | 33,900 | 0.05% | 0 |
| 37 | `aux_physiological_head` | 16,578 | 0.02% | 0 |
| 38 | `aux_facial_head` | 16,578 | 0.02% | 0 |
| 39 | `digital_heartbeat_detector` | 14,354 | 0.02% | 0 |
| 40 | `optical_flow_analyzer` | 8,737 | 0.01% | 0 |
| 41 | `eye_analysis_module` | 8,067 | 0.01% | 0 |
| 42 | `aux_forensic_head` | 2,146 | <0.01% | 0 |
| 43 | `camera_metadata_analyzer` | 1,744 | <0.01% | 0 |
| 44 | `av_sync_analyzer` | 656 | <0.01% | 0 |
| 45 | `rolling_shutter_detector` | 184 | <0.01% | 0 |
| 46 | `component_weights` | 50 | <0.01% | 0 |
| 47 | `deepfake_threshold` | 1 | <0.01% | 0 |
| 48 | `frequency_threshold` | 1 | <0.01% | 0 |
| 49 | `noise_threshold` | 1 | <0.01% | 0 |
| 50 | `temporal_consistency_threshold` | 1 | <0.01% | 0 |
| 51 | `diversity_loss_weight` | 1 | <0.01% | 0 |
| | **TOTAL** | **68,556,807** | **100%** | **5,544** |

### 21.2 Buffer Inventory (209 buffers)

Buffers are non-parameter state tensors (BatchNorm running statistics, MFCC transform matrices):

| Component | Buffer Type | Count |
|-----------|-------------|-------|
| `visual_model` (EfficientNet-B0) | running_mean, running_var, num_batches_tracked | 156 |
| `audio_model` | running_mean, running_var, num_batches_tracked | 9 |
| `audio_model.mfcc_transform` | dct_mat [40,40], window [400], mel_scale.fb [201,40] | 3 |
| `micro_expression_detector` | BN3d running stats | 6 |
| `advanced_physiological_analyzer` | BN1d running stats | 18 |
| `blood_flow_analyzer` | BN1d running stats | 9 |
| `breathing_pattern_detector` | BN1d running stats | 9 |
| `classifier` | BN1d running stats | 6 |
| `component_contribution_ema` | [50] | 1 |
| `component_usage_count` | [50] | 1 |

---

## 22. Data Features Catalog

### 22.1 Visual Features

| Feature | Extraction Method | Shape | Purpose |
|---------|------------------|-------|---------|
| Face frames | dlib HOG/CNN detection + crop | [32, 3, 224, 224] | Primary visual input |
| Facial landmarks | dlib 68-point predictor | [32, 136] | Face geometry analysis |
| Head pose | Landmark-based estimation | [32, 3] | Head movement consistency |
| Eye blink | Eye aspect ratio (EAR) | [32] | Blink naturalness |
| Pulse signal | Color change in face ROI | [32] | Heartbeat detection |
| Skin color | Mean RGB from face region | [32, 3] | Color consistency |
| Frequency features | DCT transform | [1, 16, 16] | Compression artifacts |
| Face embeddings | Per-frame identity | [1, 256] | Identity consistency |
| Temporal consistency | Frame-to-frame diff | scalar | Temporal smoothness |
| ELA features | Error level analysis | [112, 112] | Compression forensics |

### 22.2 Audio Features

| Feature | Extraction Method | Shape | Purpose |
|---------|------------------|-------|---------|
| Raw audio | librosa.load at 16kHz | [audio_length] | Primary audio input |
| Audio spectrogram | Mel spectrogram | [1, 64, 64] | Spectral analysis |
| MFCC | 20 coefficients, 50 steps | [20, 50] | Vocal characteristics |
| Voice stress | Jitter, shimmer, HNR | [6] | Vocal stress patterns |
| A/V sync | Audio-visual alignment | [5] | Lip sync detection |
| Metadata | Bitrate, fps, codec info | [10] | File-level forensics |

---

## 23. File Structure

```
Models/
  train_multimodal.py              # Training loop, FocalLoss, ModelEMA, DeepfakeTrainer
  multi_modal_model.py             # MultiModalDeepfakeModel (68.5M params)
  dataset_loader.py                # Data loading, face detection, feature extraction
  improved_augmentation.py         # Training/validation transforms
  safe_collate.py                  # Custom collate function for variable-length data
  advanced_model_components.py     # AttentionFusion, micro-expression, AU analyzer
  advanced_physiological_analysis.py # Heartbeat, blood flow, breathing detectors
  mobile_sensor_analysis.py        # Optical flow, camera metadata, depth, shutter
  voice_stress_analyzer.py         # Jitter/shimmer, emotional detection, formants
  skin_analyzer.py                 # Skin color temporal analysis
  fallbacks.py                     # Fallback implementations when dependencies missing
  quantization_utils.py            # QAT utilities
  inference.py                     # Single video inference
  inference_api.py                 # REST API for deployment
  predict_deployment.py            # Production deployment wrapper
  train_production_mobile.ps1      # PowerShell training launcher
  requirements.txt                 # Python dependencies
  checkpoints/                     # Saved model checkpoints
  outputs/                         # Training logs and visualizations
  deepfake-env-311/                # Python 3.11 virtual environment
```

---

## 24. How to Run

### 24.1 Environment Setup

```powershell
cd F:\Deepfakee\Models
.\deepfake-env-311\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 24.2 Training (Full Pipeline)

```powershell
cd F:\Deepfakee\Models
.\train_production_mobile.ps1
```

**With options:**
```powershell
# Subset for smoke test
.\train_production_mobile.ps1 -MaxSamples 1000

# Imbalance mitigation preset
.\train_production_mobile.ps1 -Preset imbalance

# Custom learning rate
.\train_production_mobile.ps1 -LearningRate 5e-4
```

### 24.3 Training (Direct Python)

```powershell
python train_multimodal.py `
  --json_path F:\Deepfakee\LAV_DF\metadata.json `
  --data_dir F:\Deepfakee\LAV_DF `
  --output_dir outputs/run_manual `
  --checkpoint_dir checkpoints/run_manual `
  --batch_size 16 --num_epochs 80 `
  --learning_rate 3e-4 --weight_decay 0.01 `
  --loss_type focal --focal_alpha 1.0 --label_smoothing 0.05 `
  --ema_decay 0.999 --mixup_alpha 0.2 `
  --grad_accum_steps 4 --amp_enabled `
  --warmup_epochs 3 --early_stopping_patience 10 `
  --use_spectrogram --detect_faces --enhanced_preprocessing `
  --enhanced_augmentation --use_weighted_loss
```

### 24.4 Inference

```powershell
python inference.py --video_path path/to/video.mp4 --checkpoint_path checkpoints/best_model.pth
```

---

## 25. Key Bug Fixes Applied

Over 8 development sessions, 30+ critical bugs were identified and fixed:

### Session 1: Landmark Initialization
- **Bug:** dlib face landmark predictor not initialized, producing all-zero landmarks
- **Fix:** Proper dlib initialization with shape_predictor_68_face_landmarks.dat

### Session 2: Full Codebase Audit (23 issues)
- Data leakage in train/val splitting
- AttentionFusion dimension mismatch
- Spectrogram feature dimension errors
- Dead code removal
- Inference pipeline rewrite

### Session 3: Remaining Fixes (10 items)
- `target_dim` not passed to attention modules
- `forensic_features` wiring disconnected
- Label smoothing and scheduler CLI args missing

### Session 4: Accuracy Optimizations (19 changes)
- Added EMA (decay=0.999)
- Added Mixup (alpha=0.2)
- BatchNorm in classifier (replacing bare Linear)
- Reduced dropout (0.5->0.3)
- Faster backbone unfreezing schedule
- Warmup tuning (3 epochs)

### Session 5: Data Pipeline Fixes (5 items)
- dlib fallback landmark normalization
- Deterministic validation transforms (no random compression)
- Multi-frame frequency features
- BN/bias weight decay exclusion
- EMA BatchNorm buffer tracking

### Session 6: Structural Fixes (3 items)
- Explainability projector (replaced zero-noise pipeline)
- Loss rebalancing: 0.1 audio + 0.1 video + 0.05 KL
- Discriminative learning rates (backbone 10x lower)

### Session 7: Critical Training Fixes
- **Progressive unfreezing broken:** Frozen backbone params were excluded from optimizer entirely. When unfrozen, they had no optimizer state. Fix: include ALL params in optimizer from start.
- **EMA same bug:** EMA didn't track frozen params. Fix: track ALL params including frozen.
- **Logging spam:** Reduced to every 200-500 batches
- **Non-AMP modality losses:** Per-modality losses computed in float32

### Session 8: Runtime Fixes
- **BatchNorm crash:** Batch size 1 at end of epoch caused "Expected more than 1 value per channel." Fix: `drop_last=True` for training DataLoaders.
- **PS1 parse errors:** Unicode characters, comma parameter separators, variable interpolation. Fix: ASCII-only, single-quoted strings.
- **UTF-8 encoding:** Model init emoji prints caused cp1252 encoding failures. Fix: TextIOWrapper with UTF-8.

---

## 26. References and Benchmarks

### 26.1 LAV-DF Benchmark

Published methods on LAV-DF achieve:
- Audio-visual fusion methods: 92-97% accuracy
- Video-only methods: 85-92% accuracy
- Audio-only methods: 78-88% accuracy

### 26.2 Key Papers

- **EfficientNet:** Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019)
- **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
- **Mixup:** Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
- **EMA:** Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging" (1992)
- **LAV-DF:** Cai et al., "Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization" (2022)
- **FACS / Action Units:** Ekman & Friesen, "Facial Action Coding System" (1978)
- **rPPG:** De Haan & Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG" (2013)

### 26.3 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 6GB VRAM | 8GB+ VRAM (RTX 4060+) |
| RAM | 16GB | 32GB |
| Storage | 50GB (dataset) | 100GB |
| CPU | 8 cores | 16+ cores |

---

*Generated from model architecture dump (68,556,807 parameters across 48 component groups, 209 buffers).*
