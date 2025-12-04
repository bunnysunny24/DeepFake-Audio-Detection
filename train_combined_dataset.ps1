# LAV-DF Training - Deepfake Detection
# 99,873 paired samples (fake + real comparison)

Write-Host "Starting LAV-DF Training..." -ForegroundColor Cyan

& "D:\Bunny\Deepfake\backend\Models\deepfake-env\Scripts\activate.ps1"

$env:CUDA_VISIBLE_DEVICES = '0'
$env:TORCH_CUDNN_BENCHMARK = '1'
$env:TORCH_ALLOW_TF32 = '1'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:512'

python train_multimodal.py `
  --json_path "D:\Bunny\Deepfake\backend\LAV-DF\metadata.json" `
  --data_dir "D:\Bunny\Deepfake\backend\LAV-DF" `
  --output_dir "D:\Bunny\Deepfake\backend\Models\outputs" `
  --checkpoint_dir "D:\Bunny\Deepfake\backend\Models\checkpoints" `
  --log_file "D:\Bunny\Deepfake\backend\Models\outputs\training_log.txt" `
  --batch_size 4 `
  --num_epochs 10 `
  --max_samples 50 `
  --learning_rate 1e-5 `
  --weight_decay 0.0001 `
  --detect_faces `
  --compute_spectrograms `
  --use_spectrogram `
  --validation_split 0.1 `
  --optimizer adamw `
  --scheduler cosine_with_restarts `
  --warmup_epochs 3 `
  --loss_type focal `
  --focal_alpha 0.25 `
  --focal_gamma 2.0 `
  --class_weights_mode manual_extreme `
  --use_weighted_loss `
  --dropout_rate 0.3 `
  --gradient_clip 1.0 `
  --early_stopping_patience 12 `
  --reduce_frames 6 `
  --enhanced_preprocessing `
  --enhanced_augmentation `
  --enable_skin_color_analysis `
  --enable_advanced_physiological `
  --enable_face_mesh `
  --num_workers 4 `
  --pin_memory 

Write-Host "Training completed" -ForegroundColor Green
# Remove-Item Env:CUDA_CACHE_MAXSIZE

Write-Host ""
Write-Host "="*70 -ForegroundColor Green
Write-Host "TRAINING COMPLETE" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Test on Samsung dataset again:" -ForegroundColor White
Write-Host "   python test_samsung_random.py" -ForegroundColor Yellow
Write-Host '   Expected: 23% -> 70-80% accuracy' -ForegroundColor Green
Write-Host ""
Write-Host "2. Test on external videos:" -ForegroundColor White
Write-Host "   python test_folder.py '$basePath\TESTING\deepfake videos'" -ForegroundColor Yellow
Write-Host '   Expected: 40% -> 70-80% accuracy' -ForegroundColor Green
Write-Host ""
Write-Host "3. Model trained on COMBINED_DATASET v3:" -ForegroundColor White
Write-Host "   - LAV-DF: 136,304 videos (lipsync manipulation)" -ForegroundColor Gray
Write-Host "   - Samsung: 19,595 videos (face-swap + manipulated audio)" -ForegroundColor Gray
Write-Host '   - Total: 155,899 videos (99.9% with original refs)' -ForegroundColor Gray
Write-Host ""
Write-Host "Evaluation metrics:" -ForegroundColor Cyan
Write-Host '  >> Confusion matrix should have non-zero values in all cells' -ForegroundColor White
Write-Host '  >> Real class F1 should be greater than 0.30 by epoch 1' -ForegroundColor White
Write-Host '  >> Loss should be visible in range 1.0-2.0, not 0.0000' -ForegroundColor White
Write-Host '  [!] If degenerate warning appears, increase focal_gamma to 4.0 or 5.0' -ForegroundColor Yellow
Write-Host ""
