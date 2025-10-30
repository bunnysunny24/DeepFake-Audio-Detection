"""
Verification script to check if advanced components are properly integrated
Run this before training to ensure everything is set up correctly
"""

import sys
import os

print("=" * 80)
print("ADVANCED COMPONENTS VERIFICATION")
print("=" * 80)
print()

# 1. Check if advanced_model_components.py exists
print("1. Checking for advanced_model_components.py...")
if os.path.exists("advanced_model_components.py"):
    print("   ✅ File found!")
    try:
        from advanced_model_components import (
            SelfAttentionPooling,
            TemporalConsistencyDetector,
            EnhancedCrossModalFusion,
            PeriodicalFeatureExtractor,
            MultiScaleFeatureFusion
        )
        print("   ✅ All components imported successfully!")
        print("      - SelfAttentionPooling")
        print("      - TemporalConsistencyDetector")
        print("      - EnhancedCrossModalFusion")
        print("      - PeriodicalFeatureExtractor")
        print("      - MultiScaleFeatureFusion")
        print("   ℹ️  Note: FocalLoss available as 'FocalLossWithLogits' (not imported by default)")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
else:
    print("   ❌ File not found!")
print()

# 2. Check if improved_augmentation.py exists
print("2. Checking for improved_augmentation.py...")
if os.path.exists("improved_augmentation.py"):
    print("   ✅ File found!")
    try:
        from improved_augmentation import (
            get_advanced_video_transforms,
            get_advanced_audio_transforms,
            TemporalConsistencyAugmenter,
            mix_up_augmentation
        )
        print("   ✅ All augmentation functions imported successfully!")
        print("      - get_advanced_video_transforms")
        print("      - get_advanced_audio_transforms")
        print("      - TemporalConsistencyAugmenter")
        print("      - mix_up_augmentation")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
else:
    print("   ❌ File not found!")
print()

# 3. Check if multi_modal_model.py has the integration
print("3. Checking multi_modal_model.py integration...")
try:
    with open("multi_modal_model.py", "r", encoding="utf-8") as f:
        content = f.read()
        
    checks = {
        "Import statement": "from advanced_model_components import",
        "ADVANCED_COMPONENTS_AVAILABLE flag": "ADVANCED_COMPONENTS_AVAILABLE = True",
        "use_advanced_components flag": "self.use_advanced_components = True",
        "SelfAttentionPooling init": "self.visual_self_attention = SelfAttentionPooling",
        "TemporalConsistencyDetector init": "self.temporal_consistency_detector = TemporalConsistencyDetector",
        "EnhancedCrossModalFusion init": "self.enhanced_cross_modal_fusion = EnhancedCrossModalFusion",
        "PeriodicalFeatureExtractor init": "self.periodical_extractor = PeriodicalFeatureExtractor",
        "MultiScaleFeatureFusion init": "self.multiscale_fusion = MultiScaleFeatureFusion",
        "Forward pass integration": "if hasattr(self, 'use_advanced_components') and self.use_advanced_components:",
    }
    
    all_found = True
    for check_name, check_string in checks.items():
        if check_string in content:
            print(f"   ✅ {check_name}")
        else:
            print(f"   ❌ {check_name}")
            all_found = False
    
    if all_found:
        print("\n   ✅ All integration checks passed!")
    else:
        print("\n   ⚠️ Some integration checks failed!")
        
except Exception as e:
    print(f"   ❌ Error reading file: {e}")
print()

# 4. Check if dataset_loader.py has the integration
print("4. Checking dataset_loader.py integration...")
try:
    with open("dataset_loader.py", "r", encoding="utf-8") as f:
        content = f.read()
        
    checks = {
        "Import statement": "from improved_augmentation import",
        "get_advanced_video_transforms usage": "get_advanced_video_transforms(phase='train')",
        "get_advanced_audio_transforms usage": "get_advanced_audio_transforms(phase='train')",
        "mix_up_augmentation usage": "mix_up_augmentation",
    }
    
    all_found = True
    for check_name, check_string in checks.items():
        if check_string in content:
            print(f"   ✅ {check_name}")
        else:
            print(f"   ❌ {check_name}")
            all_found = False
    
    if all_found:
        print("\n   ✅ All integration checks passed!")
    else:
        print("\n   ⚠️ Some integration checks failed!")
        
except Exception as e:
    print(f"   ❌ Error reading file: {e}")
print()

# 5. Test model initialization
print("5. Testing model initialization...")
try:
    import torch
    from multi_modal_model import MultiModalDeepfakeModel
    
    print("   Creating model instance...")
    model = MultiModalDeepfakeModel(
        num_classes=2,
        video_feature_dim=1024,
        audio_feature_dim=1024,
        debug=True
    )
    
    # Check if advanced components are initialized
    if hasattr(model, 'use_advanced_components'):
        if model.use_advanced_components:
            print("   ✅ Model created with advanced components enabled!")
            print(f"   ✅ Advanced component attributes found:")
            
            advanced_attrs = [
                'visual_self_attention',
                'audio_self_attention',
                'temporal_consistency_detector',
                'enhanced_cross_modal_fusion',
                'periodical_extractor',
                'multiscale_fusion'
            ]
            
            for attr in advanced_attrs:
                if hasattr(model, attr):
                    print(f"      ✅ {attr}")
                else:
                    print(f"      ❌ {attr} (missing)")
        else:
            print("   ⚠️ Model created but advanced components are disabled")
    else:
        print("   ⚠️ Model created without advanced components integration")
        
except Exception as e:
    print(f"   ❌ Error creating model: {e}")
    import traceback
    traceback.print_exc()
print()

# Summary
print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print()
print("Next steps:")
print("1. If all checks passed (✅), you can proceed with training")
print("2. If some checks failed (❌), review the error messages above")
print("3. Run the training script: .\\train_enhanced_model.ps1")
print()
print("Expected log messages during training:")
print("  - '✅ Successfully imported advanced model components'")
print("  - '✅ Integrating advanced model components...'")
print("  - '✅ Advanced components initialized successfully!'")
print("  - '✅ Using MixUp/CutMix augmentation from improved_augmentation.py'")
print("=" * 80)
