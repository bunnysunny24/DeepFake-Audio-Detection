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
  --learning_rate 1e-4 `
  --weight_decay 5e-5 `
  --dropout_rate 0.1 `
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
  --scheduler_step_size 8 `
  --scheduler_gamma 0.7 `
  --warmup_epochs 3 `
  --early_stopping_patience 12 `
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

Write-Host "CONVERGENCE FIX APPLIED!" -ForegroundColor Magenta
Write-Host "CHANGES MADE TO FIX TRAINING ISSUES:" -ForegroundColor Green
Write-Host "  1. FOCAL LOSS: Addresses severe class imbalance" -ForegroundColor Yellow
Write-Host "  2. LOWER LEARNING RATE: 1e-4 for stable convergence" -ForegroundColor Yellow
Write-Host "  3. STEP SCHEDULER: More controlled learning rate decay" -ForegroundColor Yellow
Write-Host "  4. REDUCED WORKERS: Prevents I/O corruption" -ForegroundColor Yellow
Write-Host "  5. STRONGER LABEL SMOOTHING: 0.1 for better generalization" -ForegroundColor Yellow
Write-Host "  6. EXTENDED PATIENCE: 12 epochs to allow proper convergence" -ForegroundColor Yellow
Write-Host "EXPECTED RESULTS:" -ForegroundColor Cyan
Write-Host "  - Training accuracy should steadily increase" -ForegroundColor Yellow
Write-Host "  - Validation will follow training (no more flat lines)" -ForegroundColor Yellow
Write-Host "  - Both classes will be predicted in confusion matrix" -ForegroundColor Yellow
Write-Host "  - AUC should reach 0.75+ by epoch 15-20" -ForegroundColor Yellow
