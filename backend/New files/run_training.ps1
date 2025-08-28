# MEMORY-OPTIMIZED TRAINING FOR 8GB GPU
# Fixes CUDA out of memory with mandatory physiological features

Write-Host "MEMORY-OPTIMIZED MODE FOR 8GB GPU" -ForegroundColor Red
Write-Host "   DETECTED: 8GB GPU (not 15.8GB as expected)" -ForegroundColor Yellow
Write-Host "   MANDATORY: Digital heartbeat + Blood flow + Breathing" -ForegroundColor Magenta
Write-Host "   OPTIMIZED: For 8GB GPU memory limit" -ForegroundColor Yellow
Write-Host "   TARGET: 25-35 minutes per epoch" -ForegroundColor Green

# Aggressive memory optimization
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:64"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OMP_NUM_THREADS = "14"
$env:MKL_NUM_THREADS = "14"
$env:NUMEXPR_NUM_THREADS = "14"
$env:OPENBLAS_NUM_THREADS = "14"

# Change to Models directory
Set-Location "F:\deepfake\backup\Models"

# Activate Python 3.12 virtual environment
Write-Host "Activating Python 3.12 virtual environment..." -ForegroundColor Cyan
.\deepfake-env-312\Scripts\Activate.ps1

Write-Host "MEMORY-OPTIMIZED Configuration:" -ForegroundColor Red
Write-Host "   Samples: 1,500 (ultra-conservative for 8GB)" -ForegroundColor Yellow
Write-Host "   Epochs: 15 (good balance)" -ForegroundColor Yellow
Write-Host "   Batch Size: 6 (ultra-safe for 8GB GPU)" -ForegroundColor Yellow
Write-Host "   Frames: 3 per video (memory optimized)" -ForegroundColor Yellow
Write-Host "   MANDATORY: Heartbeat + Blood flow + Breathing patterns" -ForegroundColor Magenta
Write-Host "   Memory: Optimized for 8GB GPU limit" -ForegroundColor Green

# Create output directories
New-Item -ItemType Directory -Force -Path "F:\deepfake\backup\Models\memory_optimized_outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "F:\deepfake\backup\Models\memory_optimized_checkpoints" | Out-Null

# MEMORY-OPTIMIZED training with mandatory physiological features
python train_multimodal.py `
  --json_path "F:\deepfake\backup\LAV-DF\metadata.json" `
  --data_dir "F:\deepfake\backup\LAV-DF" `
  --output_dir "F:\deepfake\backup\Models\memory_optimized_outputs" `
  --checkpoint_dir "F:\deepfake\backup\Models\memory_optimized_checkpoints" `
  --max_samples 2000 `
  --batch_size 6 `
  --num_epochs 15 `
  --learning_rate 2e-4 `
  --weight_decay 1e-4 `
  --dropout_rate 0.3 `
  --enable_face_mesh `
  --detect_deepfake_type `
  --detect_faces `
  --compute_spectrograms `
  --temporal_features `
  --enhanced_preprocessing `
  --enable_advanced_physiological `
  --enable_skin_color_analysis `
  --physiological_fps 12 `
  --optimizer adamw `
  --scheduler cosine `
  --scheduler_patience 3 `
  --warmup_epochs 1 `
  --early_stopping_patience 5 `
  --gradient_clip 0.5 `
  --label_smoothing 0.1 `
  --amp_enabled `
  --reduce_frames 3 `
  --num_workers 10 `
  --pin_memory `
  --persistent_workers `
  --prefetch_factor 6 `
  --loss_type focal `
  --focal_gamma 2.0 `
  --focal_alpha 1.0 `
  --use_wandb `
  --save_intermediate `
  --save_intermediate_interval 20 `
  --wandb_project "deepfake-detection-8gb-optimized" `
  --wandb_run_name "8gb_ultra_conservative_1500_samples_batch6"

Write-Host "Memory-optimized training with mandatory physiological features completed!" -ForegroundColor Green
Write-Host "Features included: Digital heartbeat + Blood flow + Breathing patterns" -ForegroundColor Magenta
Write-Host "Optimized for 8GB GPU memory limit" -ForegroundColor Yellow
