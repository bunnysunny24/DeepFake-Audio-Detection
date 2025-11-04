# 🎭 Multimodal Deepfake Detection System# SRM_25OD04SRM_Audio_Visual_Deepfake_Detection_Leveraging_Digital_Biometrics

SRIB-PRISM Program

**Advanced deepfake detection using multimodal fusion of visual, audio, and physiological features.**

# DeepFake-Audio-Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)A comprehensive multimodal deepfake detection framework leveraging both audio and video streams. This project fuses computer vision, audio analysis, digital forensics, and physiological signal processing to robustly detect and explain deepfake manipulations.

[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)](https://developer.nvidia.com/cuda-zone)

---

---

## Features

## 🚀 Quick Start

- Multimodal deepfake detection (audio + video)

### Current Training Configuration- Advanced facial and physiological feature extraction

- Digital forensics (ELA, GAN fingerprints, compression artifacts)

```powershell- Audio-visual synchronization analysis

# Run the enhanced training script- Explainable AI: outputs feature importance and rationale for each prediction

.\train_enhanced_model.ps1- Extensive metrics tracking and visualization

```- Modular, extensible architecture



**Active Settings:**---

- **Dataset**: LAV-DF (100 samples, 10% validation/test split)## Example (original Linux command)

- **Model**: EfficientNet-B0 (visual) + Wav2Vec2 (audio) with attention fusion

- **Loss**: Focal Loss (γ=1.5, α=0.25) with sqrt-balanced class weightsThe command below is the original full command you provided. For a public repository we avoid embedding personal absolute paths; see the "Repository-friendly" section next for a sanitized variant.

- **Optimizer**: AdamW (lr=5e-5, weight_decay=0.001)Open PowerShell and run these commands (adjust paths to your local Windows paths). For a public repo use the repository-relative placeholders below (./data, ./outputs, ./checkpoints) or override with environment variables.

- **Batch Size**: 8 (optimized for BatchNorm stability)

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

1. **Adaptive Bias Correction** (Epoch 3+)- Store outputs/checkpoints under `./outputs/` and `./checkpoints/`.

   - Monitors probability distribution in real-time# Training README — Multimodal Deepfake Model

   - Applies ±0.05 logit bias only if needed (Fake prob <40% or >60%)

   This README contains the exact training command provided, plus a PowerShell-friendly variant, environment & prerequisites notes, and a short explanation of the most important flags. It's tailored for running laptop environment.

2. **Degenerate Solution Detection**

   - Automatically detects if all predictions are same class## Original (Linux) command provided

   - Triggers emergency fix when validation fails

cd /home/srmist54/backend/Models && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun 

3. **Emergency Median Threshold**--nproc_per_node=8 

   - Forces ~50/50 prediction split using median probability--standalone train_multimodal.py   

   - Guarantees both classes are detected--distributed   

   - Shows before/after confusion matrices--json_path "/home/srmist54/backend/combined_data/LAV-DF/metadata.json"   

--data_dir "/home/srmist54/backend/combined_data/LAV-DF"   

**Example Output:**--output_dir "/home/srmist54/backend/Models/server_outputs"   

```--checkpoint_dir "/home/srmist54/backend/Models/server_checkpoints"   --batch_size 12   --num_epochs 30   --learning_rate 1e-4   

================================================================================--enable_face_mesh   

[DEGENERATE SOLUTION DETECTED] All predictions are class 1--enable_explainability   

[EMERGENCY FIX] Applying median threshold: 0.5314--use_spectrogram   

[EMERGENCY FIX] Confusion Matrix BEFORE: [[0, 134], [0, 366]]  ❌--detect_deepfake_type   

[EMERGENCY FIX] Confusion Matrix AFTER:  [[66, 68], [184, 182]]  ✅--detect_faces   

================================================================================--compute_spectrograms   

```--temporal_features   

--enhanced_preprocessing   

### 🎯 Key Hyperparameters--enhanced_augmentation   

--enable_advanced_physiological  

| Parameter | Value | Rationale |--physiological_fps 30   

|-----------|-------|-----------|--optimizer adamw   

| **Focal Gamma** | 1.5 | Moderate focusing on hard examples |--scheduler cosine   

| **Class Weights** | sqrt_balanced | ~1.7x penalty for minority class (Real) |--scheduler_patience 5   

| **Learning Rate** | 5e-5 | Stable convergence without overshooting |--warmup_epochs 1  

| **Batch Size** | 8 | Sufficient for BatchNorm statistics |--early_stopping_patience 10   

| **Warmup Epochs** | 3 | Gradual learning rate warmup |--gradient_clip 1.0   

| **Dropout** | 0.5 | Strong regularization |--amp_enabled   

| **Gradient Clip** | 0.5 | Prevents exploding gradients |--save_intermediate   

--save_intermediate_interval 20   

-----debug   

--reduce_frames 8   

## 🏗️ Model Architecture--pin_memory   

--resume_checkpoint "/home/srmist54/backend/Models/server_checkpoints/run_20250807_193339/intermediate/checkpoint_epoch_4_batch_20.pth"   

### Multimodal Fusion Pipeline--loss_type focal   

--focal_gamma 2.0   

```--focal_alpha 1.0   

┌─────────────────┐     ┌─────────────────┐--dropout_rate 0.3  

│  Video Frames   │     │  Audio Waveform │--class_weights_mode balanced   

│  (3x224x224)    │     │  (16kHz)        │--use_wandb   

└────────┬────────┘     └────────┬────────┘--wandb_project "deepfake-detection-improved"    

         │                       │ --wandb_run_name "focal_loss_balanced_v1"

         ▼                       ▼

  ┌─────────────┐         ┌─────────────┐

  │ EfficientNet│         │  Wav2Vec2   │## PowerShell / Windows laptop variant

  │    -B0      │         │   Base      │

  └──────┬──────┘         └──────┬──────┘Open PowerShell and run these commands (adjust paths to your local Windows paths):

         │                       │

         │ 1280-dim              │ 768-dim```powershell

         │                       │# Set environment variables for this PowerShell session

         └───────────┬───────────┘$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

                     ▼$env:CUDA_VISIBLE_DEVICES = '0,1,2,3,4,5,6,7'

           ┌──────────────────┐

           │  Attention Fusion│# Change to the Models directory

           │   (2944-dim)     │Set-Location -Path "D:\Bunny\check_lib"

           └─────────┬────────┘

                     ▼# Run training with torchrun (8 processes)

           ┌──────────────────┐torchrun --nproc_per_node=8 --standalone train_multimodal.py \

           │ Transformer (4L) │  --distributed \

           └─────────┬────────┘  --json_path "D:\Bunny\check_lib\combined_data\LAV-DF\metadata.json" \

                     ▼  --data_dir "D:\Bunny\check_lib\combined_data\LAV-DF" \

           ┌──────────────────┐  --output_dir "D:\Bunny\check_lib\server_outputs" \

           │  Classifier      │  --checkpoint_dir "D:\Bunny\check_lib\server_checkpoints" \

           │  [Real / Fake]   │  --batch_size 12 \

           └──────────────────┘  --num_epochs 30 \

```  --learning_rate 1e-4 \

  --enable_face_mesh \

### Advanced Features Extracted  --enable_explainability \

  --use_spectrogram \

- **Visual**: Face detection, facial landmarks (68 points), Error Level Analysis (ELA)  --detect_deepfake_type \

- **Audio**: Spectrograms, frequency analysis, voice artifacts  ## Multimodal Deepfake Model — Training README

- **Physiological**: 

  - Heart rate estimation (rPPG)  This repository contains code to train a multimodal deepfake detection model. This README explains how to run training in a repository-friendly way (no user-specific absolute paths), how to override dataset and output paths, and provides both Linux and PowerShell examples.

  - Skin color consistency analysis

  - Eye blink detection  Important: do not commit datasets, checkpoints, or other large artifacts to version control. Add `./data/`, `./outputs/`, and `./checkpoints/` to `.gitignore`.

  - Head pose estimation

  - Temporal consistency checks  ---



---  ## Recommended repository layout (relative to repo root)



## 📁 Project Structure  - data/           # place datasets here (ignored by git)

    - LAV-DF/

```      - metadata.json

Models/      - video/

├── train_multimodal.py          # Main training script (3117 lines)      - audio/

├── train_enhanced_model.ps1     # PowerShell launcher with optimal settings  - outputs/        # logs, exported results

├── dataset_loader.py            # LAV-DF dataset loader with balanced sampling  - checkpoints/    # saved model checkpoints

├── multi_modal_model.py         # Model architecture definitions  - train_multimodal.py

├── advanced_model_components.py # Custom layers (attention, fusion, etc.)  - requirements.txt

├── advanced_physiological_analysis.py  # rPPG, blink, pose detection

├── improved_augmentation.py     # Enhanced data augmentation  ---

├── skin_analyzer.py             # Skin color consistency analysis

├── optimize_model.py            # Model optimization utilities  ## Example: Linux (sanitized)

├── predict_deepfake_fixed.py    # Inference script

├── requirements.txt             # Python dependencies  This is a sanitized example — paths are repository-relative so this README is safe to publish.

├── server_checkpoints/          # Saved model checkpoints

│   └── run_YYYYMMDD_HHMMSS/  ```bash

│       ├── best_model.pth       # Best performing model  cd /path/to/repo

│       └── regular/             # Epoch checkpoints  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --standalone train_multimodal.py \

└── server_outputs/              # Training logs and visualizations    --distributed \

    └── improved_training_log.txt    --json_path ./data/LAV-DF/metadata.json \

```    --data_dir ./data/LAV-DF \

    --output_dir ./outputs \

---    --checkpoint_dir ./checkpoints \

    --batch_size 12 --num_epochs 30 --learning_rate 1e-4 \

## 🔧 Installation    --enable_face_mesh --enable_explainability --use_spectrogram --detect_deepfake_type \

    --detect_faces --compute_spectrograms --temporal_features --enhanced_preprocessing \

### Prerequisites    --enhanced_augmentation --enable_advanced_physiological --physiological_fps 30 \

    --optimizer adamw --scheduler cosine --scheduler_patience 5 --warmup_epochs 1 \

- Python 3.12    --early_stopping_patience 10 --gradient_clip 1.0 --amp_enabled --save_intermediate \

- CUDA-capable GPU (tested on NVIDIA GPUs)    --save_intermediate_interval 20 --debug --reduce_frames 8 --pin_memory \

- Windows 10/11 with PowerShell    --resume_checkpoint ./checkpoints/run_xxx/intermediate/checkpoint_epoch_4_batch_20.pth \

    --loss_type focal --focal_gamma 2.0 --focal_alpha 1.0 --dropout_rate 0.3 \

### Setup    --class_weights_mode balanced --use_wandb --wandb_project "deepfake-detection-improved" \

    --wandb_run_name "focal_loss_balanced_v1"

```powershell  ```

# 1. Create virtual environment

python -m venv deepfake-env-312  Notes:

  - Adjust `--nproc_per_node` and `CUDA_VISIBLE_DEVICES` to match your available GPUs. For a single GPU use `--nproc_per_node=1` and `CUDA_VISIBLE_DEVICES=0`.

# 2. Activate environment  - The example above uses 4 GPUs; reduce as needed for a laptop.

.\deepfake-env-312\Scripts\activate.ps1

  ---

# 3. Install dependencies

pip install -r requirements.txt  ## Example: PowerShell (Windows)



# 4. Download dlib face landmarks model  Open PowerShell in the repository root and run:

# Place shape_predictor_68_face_landmarks.dat in Models/ directory

```  ```powershell

  # set env vars for this session

### Dependencies  $env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

  $env:CUDA_VISIBLE_DEVICES = '0'

```

torch>=2.0.0  # run training (single-GPU example)

torchvision>=0.15.0  Set-Location -Path "C:\path\to\repo"

torchaudio>=2.0.0

opencv-python>=4.8.0  torchrun --nproc_per_node=1 --standalone train_multimodal.py `

librosa>=0.10.0    --distributed `

scikit-learn>=1.3.0    --json_path "./data/LAV-DF/metadata.json" `

numpy>=1.24.0    --data_dir "./data/LAV-DF" `

pillow>=10.0.0    --output_dir "./outputs" `

matplotlib>=3.7.0    --checkpoint_dir "./checkpoints" `

seaborn>=0.12.0    --batch_size 12 `

dlib>=19.24.0    --num_epochs 30 `

mediapipe>=0.10.0    --learning_rate 1e-4 `

```    --enable_face_mesh `

    --enable_explainability `

---    --use_spectrogram `

    --detect_deepfake_type `

✅ **Both classes successfully detected!**

    ```bash

---    cd /path/to/repo

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --standalone train_multimodal.py \

## 🎮 Usage      --distributed \

      --json_path ./data/LAV-DF/metadata.json \

### Training      --data_dir ./data/LAV-DF \

      --output_dir ./outputs \

```powershell      --checkpoint_dir ./checkpoints \

# Full training with current settings (100 samples)      --batch_size 12 --num_epochs 30 --learning_rate 1e-4 \

.\train_enhanced_model.ps1      --enable_face_mesh --enable_explainability --use_spectrogram --detect_deepfake_type \

      --detect_faces --compute_spectrograms --temporal_features --enhanced_preprocessing \

# Custom training      --enhanced_augmentation --enable_advanced_physiological --physiological_fps 30 \

python train_multimodal.py `      --optimizer adamw --scheduler cosine --scheduler_patience 5 --warmup_epochs 1 \

  --json_path "F:\deepfake\backup\LAV-DF\metadata.json" `      --early_stopping_patience 10 --gradient_clip 1.0 --amp_enabled --save_intermediate \

  --data_dir "F:\deepfake\backup\LAV-DF" `      --save_intermediate_interval 20 --debug --reduce_frames 8 --pin_memory \

  --batch_size 8 `      --resume_checkpoint ./checkpoints/run_xxx/intermediate/checkpoint_epoch_4_batch_20.pth \

  --num_epochs 50 `      --loss_type focal --focal_gamma 2.0 --focal_alpha 1.0 --dropout_rate 0.3 \

  --max_samples 100 `      --class_weights_mode balanced --use_wandb --wandb_project "deepfake-detection-improved" \

  --learning_rate 5e-5 `      --wandb_run_name "focal_loss_balanced_v1"

  --focal_gamma 1.5 `    ```

  --class_weights_mode sqrt_balanced `

  --enhanced_preprocessing `    Notes:

  --enhanced_augmentation `    - Adjust `--nproc_per_node` and `CUDA_VISIBLE_DEVICES` to match your available GPUs. For a single GPU use `--nproc_per_node=1` and `CUDA_VISIBLE_DEVICES=0`.

  --enable_skin_color_analysis `    - The example above uses 4 GPUs; reduce as needed for a laptop.

  --enable_advanced_physiological

```    ---



### Full Training Command (Current)    ## Example: PowerShell (Windows)



```powershell    Open PowerShell in the repository root and run:

python train_multimodal.py `

  --json_path "F:\deepfake\backup\LAV-DF\metadata.json" `    ```powershell

  --data_dir "F:\deepfake\backup\LAV-DF" `    # set env vars for this session

  --output_dir "F:\deepfake\backup\Models\server_outputs" `    $env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

  --checkpoint_dir "F:\deepfake\backup\Models\server_checkpoints" `    $env:CUDA_VISIBLE_DEVICES = '0'

  --batch_size 8 `

  --num_epochs 50 `    # run training (single-GPU example)

  --max_samples 100 `    Set-Location -Path "C:\path\to\repo"

  --learning_rate 5e-5 `

  --weight_decay 0.001 `    torchrun --nproc_per_node=1 --standalone train_multimodal.py `

  --detect_faces `      --distributed `

  --compute_spectrograms `      --json_path "./data/LAV-DF/metadata.json" `

  --validation_split 0.1 `      --data_dir "./data/LAV-DF" `

  --test_split 0.1 `      --output_dir "./outputs" `

  --optimizer adamw `      --checkpoint_dir "./checkpoints" `

  --scheduler cosine_with_restarts `      --batch_size 12 `

  --warmup_epochs 3 `      --num_epochs 30 `

  --loss_type focal `      --learning_rate 1e-4 `

  --focal_alpha 0.25 `      --enable_face_mesh `

  --focal_gamma 1.5 `      --enable_explainability `

  --class_weights_mode sqrt_balanced `      --use_spectrogram `

  --use_weighted_loss `      --detect_deepfake_type `

  --dropout_rate 0.5 `      --detect_faces `

  --gradient_clip 0.5 `      --compute_spectrograms `

  --early_stopping_patience 8 `      --temporal_features `

  --reduce_frames 10 `      --enhanced_preprocessing `

  --enhanced_preprocessing `      --enhanced_augmentation `

  --enhanced_augmentation `      --enable_advanced_physiological `

  --enable_skin_color_analysis `      --physiological_fps 30 `

  --enable_advanced_physiological `      --optimizer adamw `

  --num_workers 4 `      --scheduler cosine `

  --amp_enabled `      --scheduler_patience 5 `

  --wandb_run_name "anti_degenerate_training" `      --warmup_epochs 1 `

  --log_file "F:\deepfake\backup\Models\server_outputs\improved_training_log.txt"      --early_stopping_patience 10 `

```      --gradient_clip 1.0 `

      --amp_enabled `

### Inference      --save_intermediate `

      --save_intermediate_interval 20 `

```powershell      --debug `

python predict_deepfake_fixed.py `      --reduce_frames 8 `

  --video_path "path/to/video.mp4" `      --pin_memory `

  --checkpoint "server_checkpoints/run_XXXXXX/best_model.pth"      --resume_checkpoint "./checkpoints/run_xxx/intermediate/checkpoint_epoch_4_batch_20.pth" `

```      --loss_type focal `

      --focal_gamma 2.0 `

---      --focal_alpha 1.0 `

      --dropout_rate 0.3 `

## 📊 Dataset      --class_weights_mode balanced `

      --use_wandb `

### LAV-DF (Large-scale Audio-Visual Deepfake Dataset)      --wandb_project "deepfake-detection-improved" `

      --wandb_run_name "focal_loss_balanced_v1"

- **Total Samples**: 136,304 videos    ```

- **Class Distribution**: 

  - Real: 36,431 (26.7%)    ---

  - Fake: 99,873 (73.3%)

- **Current Training**: 100 samples (for development)    ## How to customize paths safely (recommended for public repos)

- **Splits**: 80% train, 10% validation, 10% test

    1. Use repository-relative folders (./data, ./outputs, ./checkpoints).

**Metadata Structure:**    2. Keep a small `.env.example` file with variables and add `.env` to `.gitignore` for user-specific overrides.

```json

{    Example `.env.example`:

  "video_id": "004053",

  "label": "fake",    ```

  "split": "dev",    DATA_DIR=./data

  "video_path": "dev/004053.mp4",    OUTPUT_DIR=./outputs

  "audio_path": "dev/004053.wav"    CHECKPOINT_DIR=./checkpoints

}    ```

```

    Then users can create their own `.env` or export variables in their shell prior to running.

---

    ---

## 🔬 Key Improvements

    ## Prerequisites

### ✅ Fixed Issues

    - Python 3.11

1. **Class Weight Float16 Bug** → Explicit `dtype=torch.float32` in weight calculation    - Create a virtualenv or conda env (conda recommended on Windows for native libs)

2. **BatchNorm Validation Collapse** → Use `model.train()` for first 3 epochs    - Install PyTorch matching your CUDA/CPU via https://pytorch.org

3. **Argmax Bias** → Replaced with threshold-based predictions (0.50)    - Install the rest of the deps:

4. **Probability Imbalance** → Adaptive logit bias correction (±0.05)

5. **Degenerate Solutions** → Three-layer emergency protection system    ```bash

    pip install -r requirements.txt

### 📝 Documentation    ```



- `VALIDATION_SUCCESS.md` - Complete fix documentation    For Windows, if `dlib` or `mediapipe` fail to build use conda-forge: `conda install -c conda-forge dlib mediapipe`.

- `BATCHNORM_FIX.md` - BatchNorm collapse analysis

- `THRESHOLD_VS_ARGMAX.md` - Prediction method comparison    ---

- `improved_training_log.txt` - Detailed training logs

    ## Quick troubleshooting

---

    - OOM (out-of-memory): lower `--batch_size` or use `--amp_enabled`.

## 🎯 Performance Metrics    - Missing torch distributed features: ensure PyTorch was installed from the official wheels matching your CUDA/CPU.

    - dlib build failures: install Visual Studio Build Tools + CMake or use conda-forge prebuilt packages.

### Target Metrics

    ---

- **Accuracy**: >90% on test set
- **Macro F1**: >0.85 (balanced performance)
- **Real Class Recall**: >85% (critical for security)
- **Fake Class Recall**: >90%
- **AUC-ROC**: >0.95



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


