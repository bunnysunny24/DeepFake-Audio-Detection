# Minimal Windows launcher (single RTX 4060, num_workers=0)
Set-Location "F:\deepfake\backup\Models"

# Memory / CPU / CUDA tuning
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:64'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '14'
$env:MKL_NUM_THREADS = '14'
$env:NUMEXPR_NUM_THREADS = '14'
$env:OPENBLAS_NUM_THREADS = '14'

# Activate virtualenv
.\deepfake-env-312\Scripts\Activate.ps1

# Run training (single-process Python on Windows; num_workers forced to 0)
python train_multimodal.py `
  --json_path "F:\deepfake\backup\LAV-DF\metadata.json" `
  --data_dir "F:\deepfake\backup\LAV-DF" `
  --output_dir "F:\deepfake\backup\Models\server_outputs" `
  --checkpoint_dir "F:\deepfake\backup\Models\server_checkpoints" `
  --batch_size 4 `
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
  --wandb_run_name "focal_loss_balanced_v1"
