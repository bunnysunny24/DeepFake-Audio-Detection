# =============================================================================
# Enhanced Training Configuration Script for Deepfake Detection
# =============================================================================
# This script uses optimal hyperparameters and automatically integrates:
# - Advanced Model Components (SelfAttentionPooling, TemporalConsistencyDetector,
#   EnhancedCrossModalFusion, PeriodicalFeatureExtractor, MultiScaleFeatureFusion)
# - Improved Augmentation (from improved_augmentation.py)
# - Enhanced Preprocessing (facial landmarks, physiological features)
# 
# UPDATED: October 27, 2025 - ANTI-DEGENERATE SOLUTION SETTINGS
# - Focal Loss with gamma=3.0 (aggressive)
# - Extreme class weights (10:1 Real:Fake ratio)
# - Lower learning rate (3e-5) to prevent early convergence
# - Gradient clipping for stability
# =============================================================================

Write-Host "ENHANCED TRAINING - BALANCED & STABLE" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Optimizations applied:" -ForegroundColor Green
Write-Host "  [OK] Focal Loss gamma=2.0 - Standard focusing" -ForegroundColor White
Write-Host "  [OK] Sqrt-balanced weights (~1.7x) - Moderate penalty" -ForegroundColor White
Write-Host "  [OK] Learning Rate 1e-4 - Normal convergence speed" -ForegroundColor White
Write-Host "  [OK] Batch size 8 - Stable BatchNorm statistics" -ForegroundColor White
Write-Host "  [OK] Warmup 2 epochs - Gradual learning" -ForegroundColor White
Write-Host ""
Write-Host "Strategy:" -ForegroundColor Yellow
Write-Host "  >> Prevent model collapse with balanced hyperparameters" -ForegroundColor White
Write-Host "  >> Let model learn for multiple epochs (10+)" -ForegroundColor White
Write-Host "  >> Monitor Real class recall improvement across epochs" -ForegroundColor White
Write-Host ""
# Activate the virtual environment
& "F:\deepfake\backup\Models\deepfake-env-312\Scripts\activate.ps1"

# Memory / CPU / CUDA tuning
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:64'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '14'
$env:MKL_NUM_THREADS = '14'
$env:NUMEXPR_NUM_THREADS = '14'
$env:OPENBLAS_NUM_THREADS = '14'

# Environment variables for PyTorch Distributed (single-node, single-process with DDP)
$env:MASTER_ADDR = '127.0.0.1'
$env:MASTER_PORT = '29500'



$env:RANK = '0'
$env:WORLD_SIZE = '1'
$env:LOCAL_RANK = '0'

# Run training with improved configuration
python train_multimodal.py `
  --json_path "F:\deepfake\backup\LAV-DF\metadata.json" `
  --data_dir "F:\deepfake\backup\LAV-DF" `
  --output_dir "F:\deepfake\backup\Models\server_outputs" `
  --checkpoint_dir "F:\deepfake\backup\Models\server_checkpoints" `
  --batch_size 8 `
  --num_epochs 15 `
  --max_samples 1500 `
  --learning_rate 5e-5 `
  --weight_decay 0.001 `
  --detect_faces `
  --compute_spectrograms `
  --validation_split 0.1 `
  --test_split 0.1 `
  --optimizer adamw `
  --scheduler cosine_with_restarts `
  --warmup_epochs 3 `
  --loss_type focal `
  --focal_alpha 0.25 `
  --focal_gamma 1.5 `
  --class_weights_mode sqrt_balanced `
  --use_weighted_loss `
  --dropout_rate 0.5 `
  --gradient_clip 0.5 `
  --early_stopping_patience 8 `
  --reduce_frames 10 `
  --save_intermediate `
  --save_intermediate_interval 5 `
  --enhanced_preprocessing `
  --enhanced_augmentation `
  --enable_skin_color_analysis `
  --enable_advanced_physiological `
  --num_workers 4 `
  --amp_enabled `
  --wandb_run_name "anti_degenerate_training" `
  --log_file "F:\deepfake\backup\Models\server_outputs\improved_training_log.txt"

Write-Host ""
Write-Host "[OK] Training completed or stopped" -ForegroundColor Green
Write-Host ""
Write-Host "Check results:" -ForegroundColor Cyan
Write-Host "  >> Confusion matrix should have non-zero values in all cells" -ForegroundColor White
Write-Host "  >> Real class F1 should be greater than 0.30 by epoch 1" -ForegroundColor White
Write-Host "  >> Loss should be visible in range 1.0-2.0, not 0.0000" -ForegroundColor White
Write-Host "  [!] If degenerate warning appears, increase focal_gamma to 4.0 or 5.0" -ForegroundColor Yellow
Write-Host ""
