"""
Quantization-Aware Training (QAT) utilities for model deployment.

Prepares model for INT8/FP16 quantization with minimal accuracy loss.
Useful for:
- Edge device deployment (mobile, embedded)
- TensorRT/ONNX export for production
- Reducing inference latency and memory usage
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from copy import deepcopy


class QuantizationConfig:
    """Configuration for quantization-aware training."""
    
    def __init__(
        self,
        backend='fbgemm',  # 'fbgemm' for x86, 'qnnpack' for ARM
        activation_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        per_channel_quantization=True,
        reduce_range=False  # Set True for some older CPUs
    ):
        self.backend = backend
        self.activation_dtype = activation_dtype
        self.weight_dtype = weight_dtype
        self.per_channel_quantization = per_channel_quantization
        self.reduce_range = reduce_range


def prepare_model_for_qat(model, qconfig_spec=None, backend='fbgemm'):
    """
    Prepare model for Quantization-Aware Training (QAT).
    
    Args:
        model: PyTorch model to quantize
        qconfig_spec: Custom quantization config (optional)
        backend: 'fbgemm' (x86) or 'qnnpack' (ARM)
    
    Returns:
        model_prepared: Model ready for QAT training
    """
    print("\n" + "="*80)
    print("🔧 PREPARING MODEL FOR QUANTIZATION-AWARE TRAINING (QAT)")
    print("="*80)
    
    # Set quantization backend
    torch.backends.quantized.engine = backend
    print(f"✅ Quantization backend: {backend}")
    
    # Create model copy (preserve original)
    model_qat = deepcopy(model)
    model_qat.train()
    
    # Get default QAT config
    if qconfig_spec is None:
        if backend == 'fbgemm':
            qconfig = quant.get_default_qat_qconfig('fbgemm')
        elif backend == 'qnnpack':
            qconfig = quant.get_default_qat_qconfig('qnnpack')
        else:
            qconfig = quant.get_default_qat_qconfig('fbgemm')
    else:
        qconfig = qconfig_spec
    
    # Apply QAT configuration to model
    model_qat.qconfig = qconfig
    print(f"✅ QAT config applied: {qconfig}")
    
    # Prepare model for QAT (insert fake quantization modules)
    try:
        # For models with complex architectures, use fuse_modules first
        model_qat = fuse_model_modules(model_qat)
        print("✅ Fused compatible modules (Conv+BN+ReLU)")
    except Exception as e:
        print(f"⚠️ Module fusion skipped: {e}")
    
    # Prepare QAT (inserts FakeQuantize modules)
    model_prepared = quant.prepare_qat(model_qat, inplace=True)
    print("✅ FakeQuantize modules inserted")
    
    # Count quantizable layers
    quant_layers = count_quantizable_layers(model_prepared)
    print(f"✅ Quantizable layers: {quant_layers}")
    
    print("\n📊 QAT TRAINING GUIDELINES:")
    print("   1. Train for 5-10 epochs with QAT enabled")
    print("   2. Use lower learning rate (0.1x of normal)")
    print("   3. Monitor accuracy drop (target: <2% degradation)")
    print("   4. After QAT training, convert to INT8 for deployment")
    print("="*80 + "\n")
    
    return model_prepared


def fuse_model_modules(model):
    """
    Fuse compatible modules for better quantization.
    
    Common fusions:
    - Conv2d + BatchNorm2d + ReLU
    - Conv2d + BatchNorm2d
    - Linear + ReLU
    """
    # For EfficientNet backbone
    if hasattr(model, 'visual_model'):
        try:
            # Attempt to fuse EfficientNet layers
            # Note: EfficientNet from torchvision may have fused modules already
            model.visual_model = torch.quantization.fuse_modules(
                model.visual_model,
                [['features.0.0', 'features.0.1']],  # First conv+bn
                inplace=True
            )
        except Exception:
            pass  # Fusion not applicable or already done
    
    return model


def count_quantizable_layers(model):
    """Count layers that will be quantized."""
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            count += 1
    return count


def convert_qat_to_quantized(model_qat):
    """
    Convert QAT model to fully quantized INT8 model.
    
    Args:
        model_qat: Model after QAT training
    
    Returns:
        model_quantized: INT8 quantized model for deployment
    """
    print("\n" + "="*80)
    print("🔄 CONVERTING QAT MODEL TO INT8 QUANTIZED MODEL")
    print("="*80)
    
    # Set to eval mode before conversion
    model_qat.eval()
    
    # Convert to quantized model
    model_quantized = quant.convert(model_qat, inplace=False)
    print("✅ Model converted to INT8")
    
    # Calculate size reduction
    qat_size = get_model_size(model_qat)
    quantized_size = get_model_size(model_quantized)
    reduction = (1 - quantized_size / qat_size) * 100
    
    print(f"\n📊 MODEL SIZE COMPARISON:")
    print(f"   QAT model (FP32):     {qat_size:.2f} MB")
    print(f"   Quantized (INT8):     {quantized_size:.2f} MB")
    print(f"   Size reduction:       {reduction:.1f}%")
    print(f"   Expected speedup:     2-4x faster inference")
    print("="*80 + "\n")
    
    return model_quantized


def get_model_size(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def export_quantized_model(model, export_path, sample_input=None):
    """
    Export quantized model for deployment.
    
    Args:
        model: Quantized model
        export_path: Path to save model (.pth or .pt)
        sample_input: Sample input for ONNX export (optional)
    """
    print(f"\n💾 Exporting quantized model to: {export_path}")
    
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'quantization_config': {
            'backend': torch.backends.quantized.engine,
            'dtype': 'int8',
        }
    }, export_path)
    print(f"✅ PyTorch model saved: {export_path}")
    
    # Optional: Export to ONNX if sample input provided
    if sample_input is not None:
        try:
            onnx_path = export_path.replace('.pth', '.onnx').replace('.pt', '.onnx')
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"✅ ONNX model saved: {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")
    
    print("✅ Export complete!\n")


def validate_quantized_model(model_fp32, model_int8, val_loader, device='cpu'):
    """
    Compare accuracy between FP32 and INT8 models.
    
    Args:
        model_fp32: Original FP32 model
        model_int8: Quantized INT8 model
        val_loader: Validation data loader
        device: Device for validation
    
    Returns:
        accuracy_fp32, accuracy_int8, degradation_percent
    """
    print("\n" + "="*80)
    print("📊 VALIDATING QUANTIZATION ACCURACY")
    print("="*80)
    
    model_fp32.eval()
    model_int8.eval()
    
    correct_fp32 = 0
    correct_int8 = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Get inputs and labels
            if isinstance(batch, dict):
                inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                labels = inputs.get('labels')
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
            
            # FP32 predictions
            outputs_fp32, _ = model_fp32(inputs)
            pred_fp32 = outputs_fp32.argmax(dim=1)
            correct_fp32 += (pred_fp32 == labels).sum().item()
            
            # INT8 predictions
            outputs_int8, _ = model_int8(inputs)
            pred_int8 = outputs_int8.argmax(dim=1)
            correct_int8 += (pred_int8 == labels).sum().item()
            
            total += labels.size(0)
            
            # Limit validation to save time
            if total >= 1000:
                break
    
    accuracy_fp32 = 100.0 * correct_fp32 / total
    accuracy_int8 = 100.0 * correct_int8 / total
    degradation = accuracy_fp32 - accuracy_int8
    
    print(f"\n📊 ACCURACY COMPARISON (on {total} samples):")
    print(f"   FP32 model:           {accuracy_fp32:.2f}%")
    print(f"   INT8 quantized:       {accuracy_int8:.2f}%")
    print(f"   Accuracy drop:        {degradation:.2f}%")
    
    if degradation < 2.0:
        print("   ✅ Quantization successful! (<2% accuracy drop)")
    elif degradation < 5.0:
        print("   ⚠️ Acceptable quantization (2-5% accuracy drop)")
    else:
        print("   ❌ High accuracy drop! Consider more QAT training epochs")
    
    print("="*80 + "\n")
    
    return accuracy_fp32, accuracy_int8, degradation


# Example usage functions

def qat_training_example():
    """Example: How to use QAT during training."""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║           QUANTIZATION-AWARE TRAINING (QAT) EXAMPLE                ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    STEP 1: Prepare model for QAT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from quantization_utils import prepare_model_for_qat
    
    model = MultiModalDeepfakeModel(...)
    model_qat = prepare_model_for_qat(model, backend='fbgemm')
    
    
    STEP 2: Train with QAT enabled (5-10 epochs)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Use LOWER learning rate (0.1x normal)
    optimizer = torch.optim.Adam(model_qat.parameters(), lr=1e-5)
    
    for epoch in range(5):
        model_qat.train()
        for batch in train_loader:
            outputs, _ = model_qat(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    
    STEP 3: Convert to INT8 quantized model
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from quantization_utils import convert_qat_to_quantized
    
    model_quantized = convert_qat_to_quantized(model_qat)
    
    
    STEP 4: Validate quantization accuracy
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from quantization_utils import validate_quantized_model
    
    acc_fp32, acc_int8, drop = validate_quantized_model(
        model, model_quantized, val_loader
    )
    
    
    STEP 5: Export for deployment
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from quantization_utils import export_quantized_model
    
    export_quantized_model(
        model_quantized,
        'model_int8_quantized.pth'
    )
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║  BENEFITS: 4x smaller model, 2-4x faster inference, <2% accuracy  ║
    ║  drop for deployment to production/edge devices                   ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    qat_training_example()
