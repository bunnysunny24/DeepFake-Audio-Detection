# =============================================================================
# Enhanced Training Configuration Script for Deepfake Detection
# =============================================================================
# This script uses optimal hyperparameters and automatically integrates:
# - Advanced Model Components (SelfAttentionPooling, TemporalConsistencyDetector,
#   EnhancedCrossModalFusion, PeriodicalFeatureExtractor, MultiScaleFeatureFusion)
# - Improved Augmentation (from improved_augmentation.py)
# - Enhanced Preprocessing (facial landmarks, physiological features)
# 
# Last Updated: October 9, 2025
# =============================================================================

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
  --batch_size 6 `
  --num_epochs 50 `
  --learning_rate 3e-5 `
  --detect_faces `
  --compute_spectrograms `
  --max_samples 80 `
  --validation_split 0.15 `
  --test_split 0.1 `
  --optimizer adamw `
  --loss_type focal `
  --focal_alpha 0.65 `
  --focal_gamma 2.5 `
  --class_weights_mode manual_extreme `
  --oversample_minority `
  --scheduler cosine_with_restarts `
  --warmup_epochs 2 `
  --early_stopping_patience 15 `
  --gradient_clip 0.5 `
  --reduce_frames 10 `
  --dropout_rate 0.5 `
  --weight_decay 5e-4 `
  --use_weighted_loss `
  --save_intermediate `
  --save_intermediate_interval 25 `
  --enhanced_preprocessing `
  --enhanced_augmentation `
  --enable_skin_color_analysis `
  --enable_advanced_physiological `
  --wandb_run_name "improved_deepfake_detector_with_advanced_components" `
  --log_file "F:\deepfake\backup\Models\server_outputs\improved_training_log.txt"

# =============================================================================
# Advanced Components Status:
# - If advanced_model_components.py is present, the following will be enabled:
#   ✅ SelfAttentionPooling (visual & audio)
#   ✅ TemporalConsistencyDetector
#   ✅ EnhancedCrossModalFusion
#   ✅ PeriodicalFeatureExtractor
#   ✅ MultiScaleFeatureFusion
#
# - If improved_augmentation.py is present, the following will be enabled:
#   ✅ get_advanced_video_transforms
#   ✅ get_advanced_audio_transforms
#   ✅ mix_up_augmentation (when --enhanced_augmentation is set)
#
# Check the training log for confirmation messages:
#   "✅ Successfully imported advanced model components"
#   "✅ Integrating advanced model components..."
#   "✅ Using MixUp/CutMix augmentation from improved_augmentation.py"
# =============================================================================
