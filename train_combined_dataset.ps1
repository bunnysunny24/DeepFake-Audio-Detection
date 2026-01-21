# Production Mobile Model Training
# 62M params, >75% accuracy target, <60MB INT8

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Starting Production Mobile Training..." -ForegroundColor Cyan

# Activate environment
& "D:\Bunny\Deepfake\backend\Models\deepfake-env\Scripts\activate.ps1"

# GPU optimizations
$env:CUDA_VISIBLE_DEVICES = '0'
$env:TORCH_CUDNN_BENCHMARK = '1'
$env:TORCH_ALLOW_TF32 = '1'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:512'

# Create output directories
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = "D:\Bunny\Deepfake\backend\Models\outputs\run_$timestamp"
$checkpointDir = "D:\Bunny\Deepfake\backend\Models\checkpoints\run_$timestamp"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null

$outputFile = "$runDir\training_log.txt"
Write-Host "Logs: $outputFile" -ForegroundColor Gray

$startTime = Get-Date

# Train
python train_multimodal.py `
  --json_path "D:\Bunny\Deepfake\backend\LAV-DF\metadata.json" `
  --data_dir "D:\Bunny\Deepfake\backend\LAV-DF" `
  --output_dir $runDir `
  --checkpoint_dir $checkpointDir `
  --max_samples 100 `
  --batch_size 4 `
  --num_epochs 30 `
  --validation_split 0.2 `
  --test_split 0.1 `
  --learning_rate 5e-5 `
  --weight_decay 0.0001 `
  --detect_faces `
  --compute_spectrograms `
  --use_spectrogram `
  --enhanced_preprocessing `
  --enhanced_augmentation `
  --enable_skin_color_analysis `
  --enable_advanced_physiological `
  --enable_face_mesh `
  --optimizer adamw `
  --scheduler cosine_with_restarts `
  --warmup_epochs 5 `
  --loss_type focal `
  --focal_alpha 0.25 `
  --focal_gamma 2.0 `
  --class_weights_mode balanced `
  --use_weighted_loss `
  --dropout_rate 0.3 `
  --gradient_clip 1.0 `
  --early_stopping_patience 10 `
  --reduce_frames 16 `
  --num_workers 2 `
  --pin_memory `
  --amp_enabled `
  --grad_accum_steps 2 `
  --enable_qat `
  --qat_start_epoch 15 `
  --qat_backend fbgemm `
  --qat_lr_scale 0.1 2>&1 | Tee-Object -FilePath $outputFile

$exitCode = $LASTEXITCODE
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "Training Completed: $($duration.Hours)h $($duration.Minutes)m" -ForegroundColor Green
    
    $resultsFile = "$runDir\logs\final_results.json"
    if (Test-Path $resultsFile) {
        $results = Get-Content $resultsFile | ConvertFrom-Json
        $testAcc = $results.test_accuracy * 100
        
        Write-Host "Test Accuracy: $([math]::Round($testAcc, 2))%" -ForegroundColor $(if ($testAcc -gt 75) { "Green" } else { "Yellow" })
        Write-Host "F1 Score: $([math]::Round($results.f1_score, 4))" -ForegroundColor Gray
        
        $fp32Path = "$checkpointDir\best_model.pth"
        $int8Path = "$runDir\logs\model_int8_quantized.pth"
        
        if (Test-Path $fp32Path) {
            $size = (Get-Item $fp32Path).Length / 1MB
            Write-Host "FP32 Model: $fp32Path ($([math]::Round($size, 1)) MB)" -ForegroundColor Gray
        }
        if (Test-Path $int8Path) {
            $size = (Get-Item $int8Path).Length / 1MB
            Write-Host "INT8 Model: $int8Path ($([math]::Round($size, 1)) MB)" -ForegroundColor Gray
        }
        
        if ($testAcc -gt 75) {
            Write-Host "`nNext: python validate_checkpoints.py" -ForegroundColor Cyan
        } else {
            Write-Host "`nAccuracy below target - review training log" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "Training Failed (Exit: $exitCode)" -ForegroundColor Red
    Write-Host "Check: $outputFile" -ForegroundColor Yellow
}

Write-Host ""
