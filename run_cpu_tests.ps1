# CPU-BASED PIPELINE VERIFICATION - No GPU Needed
# Run these tests on your laptop to verify everything works

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  DEEPFAKE DETECTION PIPELINE - CPU VERIFICATION TESTS" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "[1/3] Activating virtual environment..." -ForegroundColor Cyan
& "D:\Bunny\Deepfake\backend\Models\deepfake-env\Scripts\Activate.ps1"
Write-Host "✅ Environment activated`n" -ForegroundColor Green

# Test 1: Dataset Feature Extraction
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  TEST 1: Dataset Feature Extraction" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "This tests if dataset extracts all 40+ features correctly..." -ForegroundColor Gray
Write-Host ""

python test_dataset_features.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Dataset test failed! Fix dataset extraction before proceeding." -ForegroundColor Red
    exit 1
}

Write-Host "`nPress Enter to continue to pipeline test..." -ForegroundColor Yellow
Read-Host

# Test 2: Complete Pipeline (Model + Priority System + Gradients)
Write-Host "`n"
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  TEST 2: Complete Pipeline Verification" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "This tests:" -ForegroundColor Gray
Write-Host "  - Model receives all dataset features" -ForegroundColor Gray
Write-Host "  - Priority system uses dataset features first" -ForegroundColor Gray
Write-Host "  - Forward pass works" -ForegroundColor Gray
Write-Host "  - Backward pass computes gradients" -ForegroundColor Gray
Write-Host "  - No hardcoded values block learning" -ForegroundColor Gray
Write-Host ""

python test_pipeline_cpu.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Pipeline test failed! Check error messages above." -ForegroundColor Red
    exit 1
}

# Summary
Write-Host "`n"
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  ALL TESTS COMPLETE" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Dataset extraction verified" -ForegroundColor Green
Write-Host "✅ Model pipeline verified" -ForegroundColor Green
Write-Host "✅ Priority system verified" -ForegroundColor Green
Write-Host "✅ Gradient flow verified" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Your model is READY for GPU training!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Transfer to GPU machine" -ForegroundColor Gray
Write-Host "  2. Run: .\train_combined_dataset.ps1" -ForegroundColor Gray
Write-Host "  3. Monitor 'Using dataset-provided' messages in logs" -ForegroundColor Gray
Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
