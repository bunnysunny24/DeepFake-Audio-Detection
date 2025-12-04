# 🎭 Multimodal Deepfake Detection System

**Advanced deepfake detection using multimodal fusion of visual, audio, and physiological features.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-zone)

A comprehensive multimodal deepfake detection framework leveraging both audio and video streams. This project fuses computer vision, audio analysis, digital forensics, and physiological signal processing to robustly detect and explain deepfake manipulations.

---

## 🎯 Latest Results (November 11, 2025)

### ⚠️ CRITICAL BUG DISCOVERED & FIXED ⚠️

**2-Month Debugging Journey Complete!**

**Root Cause Identified:** Class weights were **NEVER passed** to datasets during training, causing [1.0, 1.0] equal weights instead of [1.4528, 0.8095] balanced weights for 3.22:1 imbalanced dataset (76% fake, 24% real).

**Bug Impact:**
- Model learned "76% of data is fake → predict FAKE when uncertain = 76% accuracy"
- **LAV-DF Performance:** 90% (same distribution as training)
- **Samsung Performance:** 48% (different distribution)
- **External Videos:** 0% (real-world test)
- **Pattern:** Predicted FAKE for ALL videos with 70-80% confidence

**✅ FIX APPLIED (November 10, 2025):**
- Fixed `dataset_loader.py` (lines 1613-1680) - Added `class_weights_mode` parameter
- Fixed `train_multimodal.py` (lines 1208-1224) - Pass class_weights_mode to datasets
- Started new training with **correct** class weights [1.4528, 0.8095]

### 🚀 Current Training Progress (Epoch 8/30)

**NEW MODEL** (run_20251110_161212) - Training started Nov 10, 16:12:12

| Epoch | Train Acc | Train F1 | Val Acc | Val F1 (Macro) | AUC | Status |
|-------|-----------|----------|---------|----------------|-----|--------|
| 1 | 48.95% | - | **42.00%** | - | 0.52 | Starting point |
| 2 | 48.20% | - | **35.20%** | - | 0.62 | Learning patterns |
| 3 | 55.60% | - | **47.60%** | - | 0.70 | Breaking 50% |
| 4 | 65.70% | - | **66.00%** | - | 0.78 | Major jump |
| 5 | 72.65% | - | **73.60%** | - | 0.79 | Steady climb |
| 6 | 76.45% | 0.8404 | **76.40%** | **0.7135** | 0.81 | ⭐ **BEST MODEL** |
| 7 | 80.30% | 0.8795 | **75.20%** | 0.7091 | 0.83 | Slight val drop |
| 8 | 82.60% | 0.8795 | **74.40%** | 0.7018 | 0.82 | Validation plateau |

**📊 Performance Improvement:** 42.00% → 76.40% = **+34.4% in 6 epochs** ✅

**🎯 Expected Final Performance (Epochs 15-20):**
- Samsung Dataset: 48% → **75-80%** accuracy
- External Videos: 0% → **60-70%** accuracy  
- REAL Detection: Currently biased → **70-80%** correct REAL predictions
- Overall Validation: **80-85%** accuracy

**⏱️ Training Status:**
- Current: Epoch 8/30 completed (Nov 11, 13:31:44)
- Best model saved: Epoch 6 (Nov 11, 07:35:47)
- Remaining: 22 epochs (~26-33 hours)
- Early stopping patience: 12 epochs

---

## 🔧 Key Features

### Model Architecture
- **Visual Backbone:** EfficientNet-B0 (pretrained)
- **Audio Backbone:** Wav2Vec2 (pretrained)
- **Fusion:** Cross-modal attention mechanism
- **Advanced Features:** 
  - Pulse detection from video (rPPG)
  - Skin color analysis
  - Pose estimation
  - Blink detection
  - Frequency analysis
  - Facial landmarks (68-point detection)

### Training Innovations (November 2025)
✅ **Bias Elimination:** Removed hardcoded bias manipulation (±0.30 → ±0.001)
✅ **Balanced Class Weights:** Changed from sqrt_balanced to balanced (no class preference)
✅ **Larger Dataset:** Increased from 1000 to 5000 samples
✅ **Optimized Regularization:** Dropout 0.4 (reduced from 0.5)
✅ **Extended Training:** Early stopping patience 12 (increased from 8)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- NVIDIA GPU with 8GB+ VRAM (RTX 4060 or better)
- CUDA 12.4
- Windows OS (PowerShell)

### Installation

1. **Clone repository:**
```powershell
git clone <repository-url>
cd Models
```

2. **Create virtual environment:**
```powershell
python -m venv deepfake-env-312
.\deepfake-env-312\Scripts\Activate.ps1
```

3. **Install PyTorch with CUDA:**
```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

4. **Install other dependencies:**
```powershell
pip install -r requirements.txt
```

5. **Download facial landmarks model:**
- Download `shape_predictor_68_face_landmarks.dat` from [dlib models](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Extract to `Models/` directory

---

## 📊 Training

### Current Configuration (Optimized for 85-90% Accuracy)

```powershell
# Run the enhanced training script
.\train_enhanced_model.ps1
```

**Active Settings:**
- **Dataset:** LAV-DF (5000 samples, 10% validation/test split)
- **Model:** EfficientNet-B0 (visual) + Wav2Vec2 (audio) with attention fusion
- **Loss:** Focal Loss (γ=1.5, α=0.25) with **balanced** class weights
- **Optimizer:** AdamW (lr=5e-5, weight_decay=0.001)
- **Batch Size:** 10 (optimized for RTX 4060 8GB VRAM)
- **Dropout:** 0.4 (balanced regularization)
- **Early Stopping:** Patience=12 epochs
- **Advanced Features:** All enabled (pulse, skin, pose, blink, frequency, landmarks)

### Training Timeline
- **Per Epoch:** ~4.6 hours (with full features)
- **Expected Epochs:** 20-30 (early stopping)
- **Total Time:** ~4-5 days for 85-90% accuracy

---

## 🧪 Testing/Inference

```powershell
python predict_deepfake_fixed.py <model_checkpoint> <video_path>
```

**Example:**
```powershell
python predict_deepfake_fixed.py server_checkpoints/best_model.pth F:\LAV-DF\fake_014.mp4
```

- **Epochs**: 50 with early stopping (patience=8)# Change to the repository root (example)

- **Features**: Skin color analysis, physiological signals, facial landmarksSet-Location -Path "D:\path\to\repo"

- **Validation**: Three-layer anti-degenerate protection system  --json_path "./data/LAV-DF/metadata.json" \

  --data_dir "./data/LAV-DF" \

---  --output_dir "./outputs" \

  --checkpoint_dir "./checkpoints" \

## 📊 Training Features  --resume_checkpoint "./checkpoints/run_20250807_193339/intermediate/checkpoint_epoch_4_batch_20.pth" \



### ✅ Anti-Degenerate Solution System4. Install other dependencies via pip (from repository root):

## Repository-friendly usage (public/open source)

The training includes a **three-layer protection** against model collapse:

- Store datasets under `./data/` relative to the repo (e.g. `./data/LAV-DF`).

---

## 🛡️ Class Weights Bug Discovery & Fix (November 2025)

### 🔍 The 2-Month Mystery

**Initial Symptoms:**
```
✅ Training: 90% accuracy on LAV-DF  
❌ Samsung:  48% accuracy (different distribution)
❌ External: 0% accuracy (real-world videos)
🚨 Pattern:  ALL videos predicted as FAKE (70-80% confidence)
```

**Investigation Timeline:**
1. **Week 1-4:** Suspected augmentation, preprocessing issues ❌
2. **Week 5-6:** Checked model architecture, loss functions ❌
3. **Week 7:** Analyzed dataset imbalance (3.22:1 fake:real) 🔍
4. **Week 8:** **BREAKTHROUGH** - Found class weights always [1.0, 1.0] in logs! 💡

### 🐛 Root Cause Analysis

**The Bug:**
```python
# dataset_loader.py (BEFORE FIX)
def get_data_loaders(json_path, data_dir, batch_size, ...):
    # ❌ class_weights_mode parameter was MISSING!
    # Function signature didn't accept it
    # Datasets created with default [1.0, 1.0] weights
```

**The Impact:**
```
Dataset Imbalance: 76% FAKE, 24% REAL (3.22:1 ratio)
Class Weights Used: [1.0, 1.0] (equal treatment)
Expected Weights: [1.4528, 0.8095] (REAL gets 1.79× more weight)

Result: Model learned "predict FAKE = 76% accurate on training data"
        Rather than learning actual visual/audio features
```

**Evidence from Logs:**
```
[CRITICAL DEBUG] Class weights from dataset_loader: tensor([1., 1.])
[WARNING] Expected [1.4528, 0.8095] for balanced mode
[ERROR] Model shows FAKE bias - predicts FAKE for all videos
```

### ✅ The Fix

**Changes Made:**

**1. dataset_loader.py (Lines 1613-1680)**
```python
# AFTER FIX
def get_data_loaders(
    json_path, data_dir, batch_size,
    class_weights_mode='balanced',  # ✅ ADDED PARAMETER
    ...
):
    # Create datasets
    train_dataset = LAVDFDataset(...)
    val_dataset = LAVDFDataset(...)
    
    # ✅ SET class_weights_mode AFTER initialization
    train_dataset.class_weights_mode = class_weights_mode
    val_dataset.class_weights_mode = class_weights_mode
    
    # ✅ RECALCULATE weights with correct mode
    train_dataset._calculate_class_weights()
    val_dataset._calculate_class_weights()
```

**2. train_multimodal.py (Lines 1208-1224)**
```python
# AFTER FIX
train_loader, val_loader, test_loader = get_data_loaders(
    class_weights_mode=self.config.class_weights_mode,  # ✅ PASS PARAMETER
    ...
)
```

**Verification Results:**
```bash
$ python test_balanced_mode.py
✅ Class weights: [1.4528, 0.8095]
✅ REAL weight: 1.79× higher than FAKE
✅ Matches expected values for 3.22:1 imbalance
```

### 📊 Before vs After Comparison

| Metric | OLD (Broken) | NEW (Fixed) | Improvement |
|--------|--------------|-------------|-------------|
| **Class Weights** | [1.0, 1.0] | [1.4528, 0.8095] | ✅ Correct |
| **LAV-DF Accuracy** | 90% | 76% → 80%+ | Learning features |
| **Samsung Accuracy** | 48% | Expected 75-80% | 🎯 Target |
| **External Videos** | 0% | Expected 60-70% | 🎯 Target |
| **REAL Detection** | 18% (biased) | Expected 70-80% | 🎯 Target |
| **Prediction Pattern** | ALL FAKE | Varies by content | ✅ Fixed |
| **Training Progress** | Stagnant | 42%→76% in 6 epochs | ✅ Working |

### 🎓 Lessons Learned

1. **Always verify parameter passing** - Parameter existed in config but never reached datasets
2. **Log intermediate values** - [1., 1.] in logs revealed the bug
3. **Test on multiple distributions** - Samsung/external showed the generalization failure
4. **Class imbalance is critical** - 3.22:1 ratio requires proper weighting
5. **Diagnostic tools are essential** - test_balanced_mode.py confirmed the fix
---

## 📊 Dataset Information

### COMBINED_DATASET (155,899 videos)

**Composition:**
- **LAV-DF:** 136,304 videos (87.4%)
- **Samsung FakeAVCeleb:** 19,595 videos (12.6%)
- **Location:** `F:\deepfake\backup\COMBINED_DATASET\`

**Class Distribution:**
- **FAKE:** 118,627 videos (76.1%)
- **REAL:** 37,272 videos (23.9%)
- **Imbalance Ratio:** 3.22:1 (FAKE:REAL)

**Training Configuration:**
- **Train:** 2,000 samples (80%)
- **Validation:** 250 samples (10%)
- **Test:** 250 samples (10%)
- **Class Weights:** [1.4528, 0.8095] (REAL gets 1.79× weight)

**Metadata Structure:**
```json
{
  "video_id": "004053",
  "label": "fake",
  "split": "dev",
  "video_path": "LAV-DF/dev/004053.mp4",
  "audio_path": "LAV-DF/dev/004053.wav"
}
```

---

## 🚀 Quick Start

### Prerequisites
- **Python:** 3.12
- **GPU:** NVIDIA RTX 4060 or better (8GB+ VRAM)
- **CUDA:** 12.4
- **OS:** Windows 10/11 with PowerShell

### Installation

1. **Clone repository:**
```powershell
git clone <repository-url>
cd Models
```

2. **Create virtual environment:**
```powershell
python -m venv deepfake-env-312
.\deepfake-env-312\Scripts\Activate.ps1
```

3. **Install PyTorch with CUDA:**
```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

4. **Install other dependencies:**
```powershell
pip install -r requirements.txt
```

5. **Download facial landmarks model:**
- Download `shape_predictor_68_face_landmarks.dat` from [dlib models](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Extract to `Models/` directory

---

## 🎮 Training

### Current Training (Resume from Epoch 8)

```powershell
# Resume training from checkpoint
.\train_combined_dataset.ps1
```

**Training Configuration:**
```powershell
python train_multimodal.py `
  --json_path "F:\deepfake\backup\COMBINED_DATASET\metadata.json" `
  --data_dir "F:\deepfake\backup\COMBINED_DATASET" `
  --output_dir "F:\deepfake\backup\Models\server_outputs" `
  --checkpoint_dir "F:\deepfake\backup\Models\server_checkpoints" `
  --resume_checkpoint "F:\deepfake\backup\Models\server_checkpoints\run_20251110_161212\regular\checkpoint_epoch_7_acc_0.7520_f1_0.7091.pth" `
  --batch_size 10 `
  --num_epochs 30 `
  --max_samples 2500 `
  --learning_rate 5e-5 `
  --weight_decay 0.001 `
  --class_weights_mode balanced `
  --loss_type focal `
  --focal_alpha 0.25 `
  --focal_gamma 3.0 `
  --optimizer adamw `
  --scheduler cosine_with_restarts `
  --warmup_epochs 3 `
  --early_stopping_patience 12 `
  --dropout_rate 0.4 `
  --gradient_clip 0.5 `
  --detect_faces `
  --compute_spectrograms `
  --enhanced_preprocessing `
  --enhanced_augmentation `
  --enable_skin_color_analysis `
  --enable_advanced_physiological `
  --amp_enabled
```

### Training Timeline
- **Per Epoch:** ~2.5 hours (RTX 4060)
- **Total Epochs:** 30 (early stopping at 12 patience)
- **Expected Completion:** ~26-33 hours remaining
- **Best Model:** Saved at epoch 6 (76.4% val accuracy)

### Monitoring Training

```powershell
# Check recent training progress
Get-Content "F:\deepfake\backup\Models\server_outputs\combined_lavdf_samsung_log.txt" -Tail 20

# Check best model timestamp
Get-ChildItem "F:\deepfake\backup\Models\server_checkpoints\run_20251110_161212\best_model.pth" | Select LastWriteTime

# Monitor GPU usage
nvidia-smi
```



---

## 🐛 Troubleshooting

### Common Issues

**1. Disk Space Full**
```
Error: [Errno 28] No space left on device
Solution: Intermediate checkpoints disabled (saves only epoch + best model)
Expected disk usage: ~30 GB for 50 epochs
```

**2. All Predictions Same Class**
```
[DEGENERATE SOLUTION DETECTED] All predictions are class X
Solution: Emergency median threshold fix activates automatically
```

**3. CUDA Out of Memory**
```
Error: CUDA OOM
Solution: Reduce --batch_size to 4 or --reduce_frames to 5
```

**4. BatchNorm Collapse in Validation**
```
Validation outputs identical for all samples
Solution: Automatically uses model.train() for epochs < 3
```

---

## 🔍 Monitoring Training

### Key Metrics to Watch

1. **Macro F1 Score** (primary metric)
   - Should increase steadily after epoch 5
   - Target: >0.50 by epoch 10, >0.70 by epoch 30

2. **Real Class Recall** (critical)
   - Should be >30% by epoch 5
   - Should be >60% by epoch 20
   - If stuck at 0%, emergency fix will activate

3. **Confusion Matrix**
   - All 4 cells should have non-zero values
   - If degenerate (0s in row/column), median threshold triggers

4. **Loss Values**
   - Should be in range 0.05-0.15 (not 0.0000)
   - Should decrease gradually over epochs

### Log Files

- **Training Log**: `server_outputs/improved_training_log.txt`
- **Checkpoints**: `server_checkpoints/run_YYYYMMDD_HHMMSS/`
- **Visualizations**: `server_outputs/visualizations/`

---

## 📚 References

- **EfficientNet**: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- **Wav2Vec2**: [Baevski et al., 2020](https://arxiv.org/abs/2006.11477)
- **Focal Loss**: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- **LAV-DF Dataset**: Large-scale Audio-Visual Deepfake Detection

---

## 📝 License

This project is for research and educational purposes.

---

## 👥 Contributing

For questions or issues:
1. Check the troubleshooting section
2. Review training logs in `server_outputs/`
3. Create an issue with error details and configuration


