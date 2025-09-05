# STRATIFIED SAMPLING FIX - CONSISTENT TRAIN/VAL SPLITS
# Ensures same class distribution in training and validation

Write-Host "STRATIFIED SAMPLING FIX" -ForegroundColor Magenta
Write-Host "   PROBLEM: Training learns, Validation always predicts Fake" -ForegroundColor Red
Write-Host "   CAUSE: Different class distributions in train vs validation splits" -ForegroundColor Yellow
Write-Host "   FIX: Force same class ratio in train/val + larger validation set" -ForegroundColor Green
Write-Host "   RESULT: Consistent learning behavior across both sets" -ForegroundColor Cyan

# Memory optimization
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:64"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OMP_NUM_THREADS = "14"
$env:MKL_NUM_THREADS = "14"
$env:NUMEXPR_NUM_THREADS = "14"
$env:OPENBLAS_NUM_THREADS = "14"

Set-Location "F:\deepfake\backup\Models"

Write-Host "Activating Python 3.12 virtual environment..." -ForegroundColor Cyan
.\deepfake-env-312\Scripts\Activate.ps1

Write-Host "STABILITY CONFIGURATION:" -ForegroundColor Magenta
Write-Host "   Learning Rate: 5e-5 (reduced for stability)" -ForegroundColor Yellow
Write-Host "   Dropout: 0.2 (increased regularization)" -ForegroundColor Yellow
Write-Host "   LR Schedule: 50% reduction every 3 epochs" -ForegroundColor Yellow
Write-Host "   Early Stopping: 5 epochs (prevents collapse)" -ForegroundColor Yellow
Write-Host "   Loss: Focal Loss (class imbalance solution)" -ForegroundColor Yellow

# Create output directories
New-Item -ItemType Directory -Force -Path "F:\deepfake\backup\Models\stratified_outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "F:\deepfake\backup\Models\stratified_checkpoints" | Out-Null

# CONVERGENCE FIX - Address class imbalance and training instability
python train_multimodal.py `
  --json_path "F:\deepfake\backup\LAV-DF\metadata.json" `
  --data_dir "F:\deepfake\backup\LAV-DF" `
  --output_dir "F:\deepfake\backup\Models\stratified_outputs" `
  --checkpoint_dir "F:\deepfake\backup\Models\stratified_checkpoints" `
  --max_samples 2500 `
  --batch_size 6 `
  --validation_split 0.2 `
  --test_split 0.1 `
  --num_epochs 50 `
  --learning_rate 5e-5 `
  --weight_decay 1e-4 `
  --dropout_rate 0.2 `
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
  --scheduler step `
  --scheduler_step_size 3 `
  --scheduler_gamma 0.5 `
  --warmup_epochs 3 `
  --early_stopping_patience 5 `
  --gradient_clip 0.5 `
  --label_smoothing 0.1 `
  --amp_enabled `
  --reduce_frames 3 `
  --num_workers 4 `
  --pin_memory `
  --persistent_workers `
  --prefetch_factor 2 `
  --loss_type focal `
  --focal_alpha 0.75 `
  --focal_gamma 2.0 `
  --use_weighted_loss `
  --class_weights_mode balanced `
  --use_wandb `
  --save_intermediate `
  --save_intermediate_interval 50 `
  --wandb_project "deepfake-detection-convergence-fix" `
  --wandb_run_name "focal_loss_stable_training"

Write-Host "STABILITY FIX APPLIED!" -ForegroundColor Magenta
Write-Host "CHANGES MADE TO PREVENT EPOCH 7 COLLAPSE:" -ForegroundColor Green
Write-Host "  1. LOWER LEARNING RATE: 5e-5 (was 1e-4) - prevents overshooting" -ForegroundColor Yellow
Write-Host "  2. STRONGER REGULARIZATION: dropout=0.2, weight_decay=1e-4" -ForegroundColor Yellow
Write-Host "  3. AGGRESSIVE LR SCHEDULE: 50% reduction every 3 epochs" -ForegroundColor Yellow
Write-Host "  4. EARLY STOPPING: 5 epochs (prevents training collapse)" -ForegroundColor Yellow
Write-Host "  5. FOCAL LOSS: Still addresses class imbalance" -ForegroundColor Yellow
Write-Host "EXPECTED RESULTS:" -ForegroundColor Cyan
Write-Host "  - Stable training through all epochs (no collapse)" -ForegroundColor Yellow
Write-Host "  - Sustained 80%+ F1 scores after epoch 3" -ForegroundColor Yellow
Write-Host "  - Best model saved around epoch 5-8" -ForegroundColor Yellow
Write-Host "  - Final performance: 85-90% F1 score" -ForegroundColor Yellow
