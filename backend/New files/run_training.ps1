# Deepfake Detection Training Script for Windows PC with Python 3.12
# Adapted from server paths to local Windows paths

Write-Host "Starting deepfake detection training with Python 3.12..." -ForegroundColor Green

# Set environment variables for CUDA
# $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"  # Commented out - not supported on Windows
$env:CUDA_VISIBLE_DEVICES = "0"

# Change to Models directory
Set-Location "F:\deepfake\backup\Models"

# Activate Python 3.12 virtual environment
Write-Host "Activating Python 3.12 virtual environment..." -ForegroundColor Cyan
.\deepfake-env-312\Scripts\Activate.ps1

# Run OPTIMIZED training with multiprocessing and all advanced features
Write-Host "🚀 Starting OPTIMIZED deepfake training with:" -ForegroundColor Yellow
Write-Host "   - 16 CPU cores with 8 worker processes for 5x faster data loading" -ForegroundColor Cyan
Write-Host "   - RTX 4060 8GB GPU optimization" -ForegroundColor Cyan
Write-Host "   - Advanced physiological analysis enabled" -ForegroundColor Cyan
Write-Host "   - All comprehensive features activated" -ForegroundColor Cyan
Write-Host ""

python train_multimodal.py `
  --json_path "F:\deepfake\backup\LAV-DF\metadata.json" `
  --data_dir "F:\deepfake\backup\LAV-DF" `
  --output_dir "F:\deepfake\backup\Models\optimized_outputs" `
  --checkpoint_dir "F:\deepfake\backup\Models\optimized_checkpoints" `
  --batch_size 6 `
  --num_workers 8 `
  --persistent_workers `
  --prefetch_factor 4 `
  --num_epochs 30 `
  --learning_rate 1e-4 `
  --enable_face_mesh `
  --enable_explainability `
  --use_spectrogram `
  --detect_deepfake_type `
  --detect_faces `
  --compute_spectrograms `
  --temporal_features `
  --enhanced_preprocessing `
  --enhanced_augmentation `
  --enable_advanced_physiological `
  --physiological_fps 30 `
  --optimizer adamw `
  --scheduler cosine `
  --scheduler_patience 5 `
  --warmup_epochs 1 `
  --early_stopping_patience 10 `
  --gradient_clip 1.0 `
  --amp_enabled `
  --save_intermediate `
  --save_intermediate_interval 20 `
  --debug `
  --reduce_frames 8 `
  --pin_memory `
  --loss_type focal `
  --focal_gamma 2.0 `
  --focal_alpha 1.0 `
  --dropout_rate 0.3 `
  --class_weights_mode balanced `
  --use_wandb `
  --wandb_project "deepfake-detection-improved" `
  --wandb_run_name "focal_loss_balanced_pc_training_v1"

Write-Host "Training completed!" -ForegroundColor Green
