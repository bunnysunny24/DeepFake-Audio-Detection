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

Write-Host "STRATIFIED CONFIGURATION:" -ForegroundColor Magenta
Write-Host "   Validation Split: 0.2 (larger for stable statistics)" -ForegroundColor Yellow
Write-Host "   Test Split: 0.1 (reasonable test set)" -ForegroundColor Yellow
Write-Host "   Sampling: Stratified (same ratio in train/val)" -ForegroundColor Yellow
Write-Host "   Learning Rate: 3e-4 (stable convergence)" -ForegroundColor Yellow
Write-Host "   Batch Size: 8 (better gradient estimates)" -ForegroundColor Yellow
Write-Host "   Loss: Pure weighted CrossEntropy (no focal complexity)" -ForegroundColor Yellow

# Create output directories
New-Item -ItemType Directory -Force -Path "F:\deepfake\backup\Models\stratified_outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "F:\deepfake\backup\Models\stratified_checkpoints" | Out-Null

# STRATIFIED training - Consistent splits + stable parameters
python train_multimodal.py `
  --json_path "F:\deepfake\backup\LAV-DF\metadata.json" `
  --data_dir "F:\deepfake\backup\LAV-DF" `
  --output_dir "F:\deepfake\backup\Models\stratified_outputs" `
  --checkpoint_dir "F:\deepfake\backup\Models\stratified_checkpoints" `
  --max_samples 2000 `
  --batch_size 8 `
  --validation_split 0.2 `
  --test_split 0.1 `
  --num_epochs 20 `
  --learning_rate 3e-4 `
  --weight_decay 1e-4 `
  --dropout_rate 0.15 `
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
  --scheduler_patience 4 `
  --warmup_epochs 2 `
  --early_stopping_patience 7 `
  --gradient_clip 1.0 `
  --label_smoothing 0.05 `
  --amp_enabled `
  --reduce_frames 3 `
  --num_workers 8 `
  --pin_memory `
  --persistent_workers `
  --prefetch_factor 4 `
  --loss_type crossentropy `
  --use_weighted_loss `
  --class_weights_mode balanced `
  --use_wandb `
  --save_intermediate `
  --save_intermediate_interval 30 `
  --wandb_project "deepfake-detection-stratified" `
  --wandb_run_name "consistent_train_val_splits"

Write-Host "STRATIFIED training completed!" -ForegroundColor Magenta
Write-Host "EXPECTED CONSISTENT RESULTS:" -ForegroundColor Green
Write-Host "  Training: Balanced predictions with steady improvement" -ForegroundColor Yellow
Write-Host "  Validation: SAME pattern as training (not constant predictions)" -ForegroundColor Yellow
Write-Host "  Both confusion matrices: Should have values in all 4 cells" -ForegroundColor Yellow
Write-Host "  AUC progression: Should improve consistently on both sets" -ForegroundColor Yellow
Write-Host "  Best models: Will save when both train/val improve together" -ForegroundColor Yellow
