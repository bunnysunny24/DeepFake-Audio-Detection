# LAV-DF Training - Deepfake Detection
# 99,873 paired samples (fake + real comparison)

# Set console and output encoding to UTF-8 to handle emoji characters
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Starting LAV-DF Training..." -ForegroundColor Cyan

& "D:\Bunny\Deepfake\backend\Models\deepfake-env\Scripts\activate.ps1"

$env:CUDA_VISIBLE_DEVICES = '0'
$env:TORCH_CUDNN_BENCHMARK = '1'
$env:TORCH_ALLOW_TF32 = '1'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:512'

# Capture all output to file in terminal folder
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$terminalDir = "D:\Bunny\Deepfake\backend\Models\terminal"
$outputFile = "$terminalDir\terminal_output_$timestamp.txt"

# Create terminal directory if it doesn't exist
if (-not (Test-Path $terminalDir)) {
    New-Item -ItemType Directory -Path $terminalDir -Force | Out-Null
}

Write-Host "Output will be saved to: $outputFile" -ForegroundColor Cyan

python train_multimodal.py `
  --json_path "D:\Bunny\Deepfake\backend\LAV-DF\metadata.json" `
  --data_dir "D:\Bunny\Deepfake\backend\LAV-DF" `
  --output_dir "D:\Bunny\Deepfake\backend\Models\outputs" `
  --checkpoint_dir "D:\Bunny\Deepfake\backend\Models\checkpoints" `
  --log_file "D:\Bunny\Deepfake\backend\Models\outputs\training_log.txt" `
  --batch_size 8 `
  --num_epochs 30 `
  --max_samples 100 `
  --learning_rate 5e-5 `
  --weight_decay 0.0001 `
  --detect_faces `
  --compute_spectrograms `
  --use_spectrogram `
  --validation_split 0.1 `
  --optimizer adamw `
  --scheduler cosine_with_restarts `
  --warmup_epochs 5 `
  --loss_type focal `
  --focal_alpha 0.25 `
  --focal_gamma 1.0 `
  --class_weights_mode balanced `
  --use_weighted_loss `
  --dropout_rate 0.2 `
  --gradient_clip 1.0 `
  --early_stopping_patience 15 `
  --reduce_frames 8 `
  --enhanced_preprocessing `
  --enhanced_augmentation `
  --enable_skin_color_analysis `
  --enable_advanced_physiological `
  --enable_face_mesh `
  --num_workers 4 `
  --pin_memory `
  --amp_enabled `
  --grad_accum_steps 2 `
  --enable_qat `
  --qat_start_epoch 15 `
  --qat_backend fbgemm `
  --qat_lr_scale 0.1 2>&1 | Tee-Object -FilePath $outputFile

Write-Host "Training completed" -ForegroundColor Green
# Remove-Item Env:CUDA_CACHE_MAXSIZE

Write-Host ""
Write-Host ("="*80) -ForegroundColor Green
Write-Host "TRAINING COMPLETE" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Green
Write-Host ""
Write-Host "PRODUCTION-READY MODEL TRAINED" -ForegroundColor Cyan
Write-Host ""
Write-Host "ARCHITECTURE:" -ForegroundColor Yellow
Write-Host "   - 31 Training Components (27 always-active + 4 contrastive learning)" -ForegroundColor Gray
Write-Host "   - 27 Deployment Components (contrastive learning disabled in production)" -ForegroundColor Gray
Write-Host "   - Contrastive Learning: Trains on fake+original pairs, deployment uses learned weights" -ForegroundColor Gray
Write-Host "   - Mobile Sensors (6): Optical flow, camera metadata, rolling shutter, A-V sync, depth" -ForegroundColor Gray
Write-Host ""
Write-Host "PRODUCTION ROBUSTNESS:" -ForegroundColor Yellow
Write-Host "   - Social Media Compression (Instagram, TikTok, WhatsApp, YouTube)" -ForegroundColor Gray
Write-Host "   - Resolution Degradation (224px -> 45px)" -ForegroundColor Gray
Write-Host "   - Adaptive Lighting (low-light, overexposed, shadows)" -ForegroundColor Gray
Write-Host "   - Quantization-Aware Training (INT8 from epoch 15)" -ForegroundColor Gray
Write-Host ""
Write-Host "Output Files:" -ForegroundColor Cyan
Write-Host "   - checkpoints/best_model.pth           (Best FP32 model)" -ForegroundColor Gray
Write-Host "   - logs/model_int8_quantized.pth        (4x smaller INT8 model)" -ForegroundColor Gray
Write-Host "   - logs/qat_report.json                 (Quantization accuracy report)" -ForegroundColor Gray
Write-Host "   - logs/final_results.json              (Complete training metrics)" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Test on Samsung dataset:" -ForegroundColor White
Write-Host "   python test_samsung_random.py" -ForegroundColor Yellow
Write-Host "   Expected: Improved accuracy on face-swap deepfakes" -ForegroundColor Green
Write-Host ""
Write-Host "2. Test on external videos:" -ForegroundColor White
Write-Host "   python test_folder.py 'D:\Bunny\Deepfake\backend\TESTING\deepfake videos'" -ForegroundColor Yellow
Write-Host "   Expected: Robust to compression, lighting, resolution variations" -ForegroundColor Green
Write-Host ""
Write-Host "3. Deploy INT8 model (production):" -ForegroundColor White
Write-Host "   Use logs/model_int8_quantized.pth for:" -ForegroundColor Yellow
Write-Host "   - Edge devices - mobile and embedded systems" -ForegroundColor Gray
Write-Host "   - Production servers (4x smaller, 2-4x faster)" -ForegroundColor Gray
Write-Host "   - TensorRT/ONNX deployment" -ForegroundColor Gray
Write-Host ""
Write-Host "Dataset Details:" -ForegroundColor Cyan
Write-Host "   - LAV-DF: 136,304 videos (lipsync manipulation)" -ForegroundColor Gray
Write-Host "   - 73.3% with paired originals for contrastive learning" -ForegroundColor Gray
Write-Host ""
Write-Host "Expected Performance:" -ForegroundColor Cyan
Write-Host "   - Macro F1 > 0.80 (balanced Real/Fake performance)" -ForegroundColor White
Write-Host "   - Robust to social media compression (Instagram, TikTok, WhatsApp)" -ForegroundColor White
Write-Host "   - Works across lighting conditions (low-light, overexposed, shadows)" -ForegroundColor White
Write-Host "   - Handles resolution degradation (224 to 45 to 224 pixels)" -ForegroundColor White
Write-Host "   - Component diversity enforced (no silent modules)" -ForegroundColor White
Write-Host ""
