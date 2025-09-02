# DEFINITIVE FIX - FORCES BALANCED CLASS LEARNING
# Solves the "always predict majority class" problem with aggressive balancing

Write-Host "AGGRESSIVE CLASS BALANCE FIX" -ForegroundColor Red
Write-Host "   PROBLEM: Model learned 'always predict FAKE' = 74% accuracy" -ForegroundColor Yellow
Write-Host "   SOLUTION: Force 50/50 class importance with extreme weights" -ForegroundColor Green
Write-Host "   METHOD: Undersample majority + oversample minority + heavy weights" -ForegroundColor Cyan
Write-Host "   GOAL: Force model to learn BOTH classes equally" -ForegroundColor Magenta

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

Write-Host "AGGRESSIVE BALANCE Configuration:" -ForegroundColor Red
Write-Host "   Dataset: 73% fake, 27% real (severe imbalance)" -ForegroundColor Yellow
Write-Host "   Strategy: Force 50/50 importance through sampling + weighting" -ForegroundColor Green
Write-Host "   Learning Rate: 2e-4 (conservative for stability)" -ForegroundColor Yellow
Write-Host "   Class Weights: FORCED BALANCED (not just calculated)" -ForegroundColor Yellow
Write-Host "   Focal Loss: DISABLED (using pure weighted CrossEntropy)" -ForegroundColor Yellow
Write-Host "   Expected: Model MUST predict both Real AND Fake" -ForegroundColor Magenta

# Create output directories
New-Item -ItemType Directory -Force -Path "F:\deepfake\backup\Models\aggressive_balance_outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "F:\deepfake\backup\Models\aggressive_balance_checkpoints" | Out-Null

# AGGRESSIVE BALANCED training - NO focal loss, PURE class weights
python train_multimodal.py `
  --json_path "F:\deepfake\backup\LAV-DF\metadata.json" `
  --data_dir "F:\deepfake\backup\LAV-DF" `
  --output_dir "F:\deepfake\backup\Models\aggressive_balance_outputs" `
  --checkpoint_dir "F:\deepfake\backup\Models\aggressive_balance_checkpoints" `
  --max_samples 1200 `
  --batch_size 8 `
  --num_epochs 20 `
  --learning_rate 2e-4 `
  --weight_decay 5e-5 `
  --dropout_rate 0.05 `
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
  --scheduler plateau `
  --scheduler_patience 3 `
  --warmup_epochs 2 `
  --early_stopping_patience 6 `
  --gradient_clip 0.5 `
  --label_smoothing 0.0 `
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
  --save_intermediate_interval 25 `
  --wandb_project "deepfake-detection-aggressive-balance" `
  --wandb_run_name "pure_weighted_crossentropy"

Write-Host "AGGRESSIVE training completed!" -ForegroundColor Red
Write-Host "CHECK RESULTS:" -ForegroundColor Magenta
Write-Host "  1. Confusion matrix should show predictions in BOTH columns" -ForegroundColor Yellow
Write-Host "  2. Training accuracy should VARY between epochs (not constant)" -ForegroundColor Yellow
Write-Host "  3. Model should predict some Real samples (not all Fake)" -ForegroundColor Yellow
Write-Host "  4. AUC should be > 0.6 (better than random)" -ForegroundColor Yellow
