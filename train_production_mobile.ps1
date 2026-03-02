#Requires -Version 5.1
<#
.SYNOPSIS
    Production training launcher for multimodal deepfake detection model.

.DESCRIPTION
    EfficientNet-B0 + LightweightAudio, ~41M params
    EMA (0.999) + Mixup (0.2) + Focal Loss + Discriminative LR
    Grad accumulation 4x (effective batch = 64)
    Progressive backbone unfreezing after epoch 3
    QAT from epoch 40 for INT8 deployment

.EXAMPLE
    .\train_production_mobile.ps1
    .\train_production_mobile.ps1 -MaxSamples 5000
    .\train_production_mobile.ps1 -Preset imbalance
#>

param(
    [int]$NumWorkers       = 4,
    [int]$MaxSamples       = 0,
    [switch]$DisableDetectFaces,
    [switch]$DisableSpectrograms,
    [switch]$ImbalanceMitigation,
    [int]$OversampleFactor = 2,
    [double]$FocalGamma    = 3.0,
    [double]$LearningRate  = 3e-4,
    [switch]$PrecomputeLandmarks,
    [string]$Preset        = ''
)

# -- UTF-8 output -------------------------------------------------------------
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

Write-Host ''
Write-Host '=== Production Deepfake Training ===' -ForegroundColor Cyan

# -- Presets -------------------------------------------------------------------
$ClassWeightsMode = 'balanced'

if ($Preset) {
    switch ($Preset.ToLower()) {
        'imbalance' {
            $ImbalanceMitigation = $true
            $OversampleFactor = 2
            $FocalGamma = 3.0
            $LearningRate = 3e-4
            Write-Host '[PRESET] Imbalance mitigation (oversample=2, gamma=3.0)' -ForegroundColor Cyan
        }
        'aggressive' {
            $ImbalanceMitigation = $true
            $OversampleFactor = 4
            $FocalGamma = 2.0
            $LearningRate = 5e-4
            $ClassWeightsMode = 'manual'
            Write-Host '[PRESET] Aggressive (oversample=4, gamma=2.0, lr=5e-4)' -ForegroundColor Cyan
        }
        'precompute' {
            $PrecomputeLandmarks = $true
            Write-Host '[PRESET] Precompute landmarks only' -ForegroundColor Cyan
        }
        'both' {
            $ImbalanceMitigation = $true
            $PrecomputeLandmarks = $true
            $OversampleFactor = 2
            $FocalGamma = 3.0
            $LearningRate = 3e-4
            Write-Host '[PRESET] Imbalance + precompute landmarks' -ForegroundColor Cyan
        }
        default {
            Write-Host "[WARN] Unknown preset: $Preset" -ForegroundColor Yellow
        }
    }
}

# -- Activate virtual environment ----------------------------------------------
$envScript = 'F:\Deepfakee\Models\deepfake-env-311\Scripts\Activate.ps1'
if (-not (Test-Path $envScript)) {
    Write-Host 'Virtual environment not found!' -ForegroundColor Red
    exit 1
}
. $envScript

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host 'Python not found after activation!' -ForegroundColor Red
    exit 1
}

# -- GPU and system optimizations ----------------------------------------------
$env:CUDA_VISIBLE_DEVICES    = '0'
$env:TORCH_CUDNN_BENCHMARK   = '1'
$env:TORCH_ALLOW_TF32        = '1'
$env:CUDA_TF32_TENSOR_CORES  = '1'
$env:OMP_NUM_THREADS          = '24'
$env:MKL_NUM_THREADS          = '24'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16'
$env:SHAPE_PREDICTOR_PATH    = 'F:\Deepfakee\Models\shape_predictor_68_face_landmarks.dat'

# -- Output directories -------------------------------------------------------
$timestamp     = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir        = "F:\Deepfakee\Models\outputs\run_$timestamp"
$checkpointDir = "F:\Deepfakee\Models\checkpoints\run_$timestamp"

New-Item -ItemType Directory -Path $runDir        -Force | Out-Null
New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null

$logFile = Join-Path $runDir 'training_log.txt'
Write-Host "Output : $runDir"        -ForegroundColor Gray
Write-Host "Ckpts  : $checkpointDir" -ForegroundColor Gray
Write-Host "Log    : $logFile"       -ForegroundColor Gray

$startTime = Get-Date

# -- Build python arguments ----------------------------------------------------
$pyArgs = @(
    'train_multimodal.py'

    # Paths
    '--json_path',       'F:\Deepfakee\LAV_DF\metadata.json'
    '--data_dir',        'F:\Deepfakee\LAV_DF'
    '--output_dir',      $runDir
    '--checkpoint_dir',  $checkpointDir

    # Architecture
    '--video_feature_dim',      '256'
    '--audio_feature_dim',      '256'
    '--transformer_dim',        '256'
    '--num_transformer_layers', '2'
    '--use_spectrogram'

    # Training
    '--batch_size',       '16'
    '--num_epochs',       '80'
    '--learning_rate',    $LearningRate.ToString()
    '--weight_decay',     '0.01'
    '--dropout_rate',     '0.3'
    '--gradient_clip',    '1.5'
    '--max_frames',       '32'

    # Optimizer and scheduler
    '--optimizer',        'adamw'
    '--scheduler',        'cosine_with_restarts'
    '--warmup_epochs',    '3'
    '--scheduler_gamma',  '0.5'
    '--min_lr',           '1e-6'

    # Loss
    '--loss_type',        'focal'
    '--focal_alpha',      '1.0'
    '--label_smoothing',  '0.05'
    '--use_weighted_loss'

    # Regularization
    '--ema_decay',        '0.999'
    '--mixup_alpha',      '0.2'

    # Augmentation and preprocessing
    '--enhanced_preprocessing'
    '--enhanced_augmentation'

    # Degenerate auto-mitigation
    '--enable_degenerate_auto_mitigation'
    '--degenerate_mitigation_threshold', '1'
    '--auto_mitigation_focal_gamma',     '3.0'

    # Early stopping
    '--early_stopping_patience', '10'

    # AMP + gradient accumulation
    '--amp_enabled'
    '--grad_accum_steps', '4'
    '--fast_mode'

    # Quantization-aware training
    '--enable_qat'
    '--qat_start_epoch',  '40'
    '--qat_backend',      'fbgemm'
    '--qat_lr_scale',     '0.1'
)

# Workers
$pyArgs += @('--num_workers', $NumWorkers.ToString())

# Optional: limit dataset size
if ($MaxSamples -gt 0) {
    $pyArgs += @('--max_samples', $MaxSamples.ToString())
}

# Feature flags
if (-not $DisableDetectFaces)    { $pyArgs += '--detect_faces' }
if (-not $DisableSpectrograms)   { $pyArgs += '--compute_spectrograms' }
if ($NumWorkers -gt 0)           { $pyArgs += '--pin_memory' }

# -- Imbalance mitigation (optional) ------------------------------------------
if ($ImbalanceMitigation) {
    Write-Host "[OPT] Imbalance: oversample=$OversampleFactor gamma=$FocalGamma weights=$ClassWeightsMode" -ForegroundColor Cyan
    $pyArgs += @(
        '--class_weights_mode',  $ClassWeightsMode
        '--oversample_minority'
        '--oversample_factor',   $OversampleFactor.ToString()
        '--focal_gamma',         $FocalGamma.ToString()
    )
}

# -- Precompute landmarks (optional) ------------------------------------------
if ($PrecomputeLandmarks) {
    Write-Host '[OPT] Precomputing landmarks...' -ForegroundColor Cyan
    & python precompute_landmarks.py `
        --json_path  F:\Deepfakee\LAV_DF\metadata.json `
        --data_dir   F:\Deepfakee\LAV_DF `
        --output_dir $runDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Landmark precomputation failed!' -ForegroundColor Red
        exit 1
    }
}

# -- Print config summary -----------------------------------------------------
Write-Host ''
Write-Host 'Config:' -ForegroundColor White
Write-Host "  LR=$LearningRate  batch=16  accum=4 (eff=64)  epochs=80" -ForegroundColor Gray
Write-Host '  EMA=0.999  Mixup=0.2  Focal+LS=0.05  WD=0.01' -ForegroundColor Gray
Write-Host '  Scheduler=cosine_restarts  Warmup=3  Patience=10' -ForegroundColor Gray
Write-Host "  QAT start=40  Workers=$NumWorkers  AMP=on" -ForegroundColor Gray
if ($MaxSamples -gt 0) {
    Write-Host "  MaxSamples=$MaxSamples (subset)" -ForegroundColor Yellow
} else {
    Write-Host '  Dataset=FULL (all samples)' -ForegroundColor Gray
}
Write-Host ''

# -- Run training --------------------------------------------------------------
Write-Host 'Starting training...' -ForegroundColor Green
& python @pyArgs 2>&1 | Tee-Object -FilePath $logFile

$exitCode = $LASTEXITCODE
$duration = (Get-Date) - $startTime
$elapsed  = '{0}h {1}m {2}s' -f $duration.Hours, $duration.Minutes, $duration.Seconds

Write-Host ''
Write-Host '-------------------------------------' -ForegroundColor Gray

if ($exitCode -eq 0) {
    Write-Host "Training completed in $elapsed" -ForegroundColor Green

    $resultsFile = Join-Path $runDir 'logs\final_results.json'
    if (Test-Path $resultsFile) {
        $results = Get-Content $resultsFile | ConvertFrom-Json
        $testAcc = [math]::Round($results.test_accuracy * 100, 2)
        $f1Val   = [math]::Round($results.f1_score, 4)

        if ($testAcc -ge 90) {
            Write-Host ('Test Accuracy: ' + $testAcc + '% -- TARGET MET!') -ForegroundColor Green
        } elseif ($testAcc -ge 80) {
            Write-Host ('Test Accuracy: ' + $testAcc + '% (close - needs tuning)') -ForegroundColor Yellow
        } else {
            Write-Host ('Test Accuracy: ' + $testAcc + '% (below target)') -ForegroundColor Red
        }
        Write-Host ('F1 Score: ' + $f1Val) -ForegroundColor Gray
    } else {
        Write-Host ('No results file found at: ' + $resultsFile) -ForegroundColor Yellow
    }
} else {
    Write-Host ('Training FAILED (exit code: ' + $exitCode + ')') -ForegroundColor Red
    Write-Host ('Check log: ' + $logFile) -ForegroundColor Yellow
}

Write-Host ''
