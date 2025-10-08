# Extreme balancing script for severe class imbalance
# This script uses very aggressive balancing techniques for severe class imbalance
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

# Run training with extreme class balancing for severe imbalance
python train_multimodal.py `
  --json_path "F:\deepfake\backup\LAV-DF\metadata.json" `
  --data_dir "F:\deepfake\backup\LAV-DF" `
  --output_dir "F:\deepfake\backup\Models\server_outputs" `
  --checkpoint_dir "F:\deepfake\backup\Models\server_checkpoints" `
  --batch_size 4 `
  --num_epochs 30 `
  --learning_rate 5e-5 `
  --detect_faces `
  --compute_spectrograms `
  --max_samples 40 `
  --validation_split 0.2 `
  --test_split 0.1 `
  --disable_skin_analysis `
  --disable_advanced_physio `
  --optimizer adamw `
  --loss_type focal `
  --focal_alpha 0.75 `
  --focal_gamma 2.0 `
  --class_weights_mode manual_extreme `
  --oversample_minority `
  --scheduler cosine `
  --warmup_epochs 1 `
  --early_stopping_patience 10 `
  --gradient_clip 0.1 `
  --reduce_frames 8 `
  --dropout_rate 0.4 `
  --weight_decay 1e-4 `
  --use_weighted_loss `
  --wandb_run_name "extreme_balancing_v1"
