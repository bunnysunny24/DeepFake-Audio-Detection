
# ===================
# GLOBAL FAULTHANDLER FOR NATIVE CRASH LOGGING
# ===================

import os
import sys
import faulthandler
import logging
try:
    # Ensure logs directory exists (fallback to current dir if not found)
    logs_dir = os.path.join(os.path.dirname(__file__), '../LAV_DF/dev/logs')
    os.makedirs(logs_dir, exist_ok=True)
    crash_log_path = os.path.join(logs_dir, 'native_crash.log')
    crash_log_file = open(crash_log_path, 'a')
    faulthandler.enable(file=crash_log_file, all_threads=True)
    faulthandler.enable(all_threads=True)  # Also enable for stderr
    # Setup error logging
    error_log_path = os.path.join(logs_dir, 'error.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(error_log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"[FAULTHANDLER] Native crash logging enabled: {crash_log_path}")
except Exception as fh_err:
    logging.error(f"[FAULTHANDLER] Failed to enable native crash logging: {fh_err}")

"""
🔥 ADVANCED DEEPFAKE DETECTION TRAINING SCRIPT 🔥

# Global debug flag for verbose logging
DEBUG_MODE = False

# Quick helper: if user requests symbol listing, do it before importing heavy modules
import sys
if '--print_model_symbols' in sys.argv:
    import os, re
    from collections import Counter
    model_path = os.path.join(os.path.dirname(__file__), 'multi_modal_model.py')
    if os.path.exists(model_path):
        try:
            with open(model_path, 'r', encoding='utf-8') as mf:
                content = mf.read()
            matches = re.findall(r'^\s*(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)', content, flags=re.MULTILINE)
            names = [name for _kind, name in matches]
            counts = Counter(names)

            print('\nCount Name')
            print('----- ----')
            for name, cnt in counts.most_common():
                print(f"{cnt:5d} {name}")
            print('')
        except Exception as e:
            print(f"[ERROR] Failed to read model file: {e}")
    else:
        print(f"[ERROR] Model file not found: {model_path}")
    sys.exit(0)

✅ MULTIMODAL ARCHITECTURE (31 TRAINING COMPONENTS):
   - 27 Always-Active Components (Deployment-Ready):
     • Core Detection (10): Facial landmarks, micro-expressions, head pose, eye dynamics, 
       lip-audio sync, oculomotor, lighting, texture, facial AU, landmark trajectory
     • Mobile Sensors (6): Optical flow, camera metadata, rolling shutter, A-V sync, depth, fusion
     • Physiological (4): rPPG heartbeat, blood flow, breathing, skin color
     • Audio (3): Voice analysis, MFCC, voice stress (jitter/shimmer/HNR)
     • Visual Artifacts (4): GAN fingerprint, frequency domain
   
   - 4 Contrastive Learning Components (Training-Only):
     • Feature difference analyzer, audio difference analyzer
     • Contrastive fusion, similarity scorer
     • Used during training to compare fake vs original pairs
     • Disabled in deployment (model uses learned weights on single videos)
   
   - 21 Disabled Components (Preserved in Code):
     • File Forensics (5): ELA, metadata, compression analysis
     • Heavy/Slow (8): Autoencoder, phoneme-viseme, voice biometrics, siamese, emotion
     • Advanced (8): Self-attention pooling, temporal consistency, multi-scale fusion

✅ PRODUCTION ROBUSTNESS:
   - Social Media Compression: Instagram, TikTok, WhatsApp, YouTube (multi-round)
   - Resolution Degradation: 4 quality levels (high/mid/low/very_low)
   - Adaptive Lighting: Low-light, overexposed, shadows, color temperature shifts
   - Demographic Fairness: Balanced sampling across skin tones, age, gender
   - Domain Adaptation: Adversarial training for distribution shift handling

✅ COMPONENT DIVERSITY (AUXILIARY LOSSES):
   - Per-component auxiliary classifiers (physiological, facial, audio, visual, forensic)
   - Diversity penalty to prevent feature correlation
   - Silent module detection (flags components with <1% contribution)
   - Component contribution tracking with exponential moving average
   - Prevents overfitting in 40+ component model

✅ QUANTIZATION-AWARE TRAINING (QAT):
   - Prepares model for INT8 deployment (4x smaller, 2-4x faster)
   - FakeQuantize modules simulate quantization during training
   - Post-training conversion to INT8 with <2% accuracy drop
   - Export to PyTorch (.pth) and ONNX for production deployment

✅ CLASS IMBALANCE FIXES:
   - Focal Loss: Focuses on hard examples (α, γ tunable)
   - Class-balanced weights with multiple modes
   - Macro F1 as primary metric for balanced evaluation
   - Per-class metrics tracking (Real vs Fake performance)

✅ OVERFITTING PREVENTION:
   - Dropout regularization (default 0.2-0.3)
   - L2 weight decay (default 1e-4)
   - Early stopping with Macro F1 monitoring
   - Gradient clipping for stability
   - Component diversity enforcement

Usage Examples:
  # Full training with all features
  python train_multimodal.py --loss_type focal --dropout_rate 0.2 --enable_qat --enhanced_augmentation
  
  # Production deployment training (30 epochs with QAT from epoch 15)
  python train_multimodal.py --num_epochs 30 --enable_qat --qat_start_epoch 15 --qat_backend fbgemm
"""

"""
Lightweight imports first; heavy/optional imports (visualization, sklearn, wandb, cv2, pandas)
are loaded lazily inside functions so the module can be imported for quick checks
like `--print_model_symbols` without failing when optional deps are missing.
"""

# ============================================================================
# CRITICAL: Set matplotlib backend BEFORE any other imports to prevent crashes
# The 'Agg' backend is headless and thread-safe, preventing 0xC0000005 errors
# ============================================================================
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# Fix Windows console encoding for emoji support
# ============================================================================
import sys
import os
if sys.platform == 'win32':
    # Force UTF-8 encoding for Windows console
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Older Python versions don't support reconfigure
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import torch
import GPUtil
import torch.nn as nn
import torch.nn.functional as F  # Added for Focal Loss
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import warnings
import numpy as np
import json
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse
import shutil
import torch.multiprocessing as mp
import traceback
import atexit
import gc
import signal
from pathlib import Path

# ============================================================================
# CUDA Error Handler to catch and recover from CUDA crashes
# ============================================================================
def _setup_cuda_error_handler():
    """Setup CUDA error handling to prevent silent crashes."""
    if torch.cuda.is_available():
        # Enable CUDA debug mode for better error messages
        os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')  # Set to '1' for debugging
        
        # Set CUDA device to reset on error (helps with memory fragmentation)
        try:
            # Force CUDA initialization early
            torch.cuda.init()
            device_count = torch.cuda.device_count()
            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                print(f"[CUDA] Initialized: {device_name} ({device_count} device(s))")
                
                # NOTE: Do NOT set memory fraction - let PyTorch use all available VRAM
                # torch.cuda.set_per_process_memory_fraction(0.95) - REMOVED
        except Exception as e:
            if globals().get('DEBUG_MODE', False):
                print(f"[DEBUG] [CUDA] Initialization warning: {e}")

# NOTE: Do NOT run CUDA initialization at import time on Windows/with spawn.
# Call _setup_cuda_error_handler() from the main guard to avoid running
# CUDA init during DataLoader worker spawn (which re-imports this module).

# Graceful shutdown flag and cleanup
_GLOBAL_SHUTDOWN_FLAG = {'requested': False}


def request_shutdown():
    _GLOBAL_SHUTDOWN_FLAG['requested'] = True


def is_shutdown_requested():
    return _GLOBAL_SHUTDOWN_FLAG.get('requested', False)


def cleanup_and_exit(trainer=None, save_checkpoint=True, reason="signal"):
    """Attempt to save checkpoints, close logging, release GPU resources and terminate DDP safely.

    This function is safe to call multiple times.
    """
    try:
        print(f"[CLEANUP] Cleanup requested due to {reason}")
        request_shutdown()
        # Save checkpoint if trainer provided
        if trainer is not None and save_checkpoint:
            try:
                # Prefer trainer.save_checkpoint if available
                if hasattr(trainer, 'save_checkpoint'):
                    trainer.save_checkpoint(tag=f'exit_{reason}')
                    print("[CLEANUP] Checkpoint saved via trainer.save_checkpoint()")
                else:
                    # Try to save model state_dict
                    state = {
                        'model_state_dict': getattr(trainer.model, 'state_dict', lambda: None)(),
                        'optimizer_state_dict': getattr(trainer.optimizer, 'state_dict', lambda: None)(),
                        'epoch': getattr(trainer, 'epoch', -1)
                    }
                    ckpt_path = os.path.join(getattr(trainer, 'checkpoint_dir', '.'), f'checkpoint_exit_{reason}.pth')
                    torch.save(state, ckpt_path)
                    print(f"[CLEANUP] Checkpoint saved to {ckpt_path}")
            except Exception as e:
                print(f"[CLEANUP] Failed to save checkpoint: {e}")

        # Attempt to close wandb safely if present
        try:
            import wandb
            try:
                wandb.finish(timeout=30)
                print("[CLEANUP] wandb.finish() called")
            except Exception:
                try:
                    wandb.join()
                except Exception:
                    pass
        except Exception:
            pass

        # Destroy process group if initialized
        try:
            if dist.is_available() and dist.is_initialized():
                try:
                    dist.barrier(timeout=30)
                except Exception:
                    pass
                try:
                    dist.destroy_process_group()
                    print("[CLEANUP] Distributed process group destroyed")
                except Exception as e:
                    print(f"[CLEANUP] Error destroying process group: {e}")
        except Exception:
            pass

        # Clear CUDA cache only if OOM occurred
        try:
            if torch.cuda.is_available():
                # Only clear cache during cleanup to free memory for next process
                torch.cuda.empty_cache()
                print("[CLEANUP] torch.cuda.empty_cache() called")
        except Exception:
            pass

        # Force garbage collection
        try:
            gc.collect()
        except Exception:
            pass

    except Exception as top_e:
        logging.error(f"[CLEANUP] Top-level cleanup error: {top_e}")


def _signal_handler(signum, frame):
    # Soft-request shutdown and let training loop exit gracefully
    print(f"[SIGNAL] Caught signal {signum}. Requesting graceful shutdown...")
    request_shutdown()


# Register signal handlers for SIGINT/SIGTERM
try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    # Some platforms or contexts may not allow signal handling
    pass

# Optional/lazy imports - try to import now but if they fail set fallbacks to None
try:
    from multi_modal_model import MultiModalDeepfakeModel
except Exception as _e:
    MultiModalDeepfakeModel = None
    _MULTIMODAL_IMPORT_ERROR = _e

try:
    # dataset loading utilities are heavy; if missing, raise when used
    from dataset_loader import get_data_loaders, get_transforms, get_transforms_enhanced
except Exception as _e:
    get_data_loaders = None
    get_transforms = None
    get_transforms_enhanced = None
    _DATA_LOADER_IMPORT_ERROR = _e

try:
    from skin_analyzer import SkinColorAnalyzer
except Exception:
    SkinColorAnalyzer = None

# Quick delegation: if user wants the paired fake/original training path, run the
# dedicated pairwise trainer we added (`train_pairwise_smoke.py`) and exit. This
# re-uses the hardened `MultiModalDeepfakeDataset` with `load_originals_always=True`.
if '--use_paired' in sys.argv:
    try:
        # Import and run the pairwise smoke trainer with sensible defaults.
        # Note: train_pairwise_smoke.py module must exist in the same directory
        # from train_pairwise_smoke import main as _paired_main
        # Remove our flag so argparse in the module doesn't see it twice
        sys.argv = [a for a in sys.argv if a != '--use_paired']
        # Run with defaults; the module accepts CLI args if present
        # _paired_main()
        if globals().get('DEBUG_MODE', False):
            print("[DEBUG] --use_paired flag detected but train_pairwise_smoke.py not found. Continuing with normal training.")
        # sys.exit(0)
    except Exception as _e:
        print(f"[ERROR] Failed to launch pairwise trainer: {_e}")
        # fallthrough to normal training

# 🧼 PROCESS SAFETY: We set start method in the main guard to avoid
# being invoked in worker subprocesses during spawn.


def _format_shape_info(x):
    try:
        if isinstance(x, torch.Tensor):
            return f"Tensor shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
        elif isinstance(x, (list, tuple)):
            return f"List/Tuple len={len(x)}"
        elif hasattr(x, 'shape'):
            return f"Has shape attribute: {getattr(x, 'shape', None)}"
        else:
            return f"Type={type(x)}"
    except Exception as e:
        return f"<error formatting shape: {e}>"


def log_batch_shapes(batch, prefix="batch"):
    """Log shapes/types for items in a batch or results dict. Guarded by DEBUG_MODE.

    Use `globals().get('DEBUG_MODE', False)` to avoid NameError in different contexts.
    """
    if not globals().get('DEBUG_MODE', False):
        return
    try:
        print(f"[SHAPES] {prefix} contents:")
        if isinstance(batch, dict):
            for k, v in batch.items():
                try:
                    info = _format_shape_info(v)
                    # For small tensors show min/max as well to detect degenerate values
                    if isinstance(v, torch.Tensor) and v.numel() > 0 and v.numel() <= 1000:
                        try:
                            info += f", min={v.min().item():.6g}, max={v.max().item():.6g}, mean={v.mean().item():.6g}"
                        except Exception:
                            pass
                    print(f"  - {k}: {info}")
                except Exception as e:
                    print(f"  - {k}: <error retrieving shape: {e}>")
        else:
            print(f"  - {prefix}: {_format_shape_info(batch)}")
    except Exception as e:
        print(f"[SHAPES] Error logging shapes for {prefix}: {e}")


# 🔧 FIX NCCL GPU MAPPING WARNING: Set CUDA device IMMEDIATELY for distributed training
if "LOCAL_RANK" in os.environ:
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        print(f"[Rank {local_rank}] 🚀 Starting distributed initialization (rank {local_rank}/{world_size})")
        
        # Verify CUDA is available before setting device
        if not torch.cuda.is_available():
            print(f"[Rank {local_rank}] ❌ CUDA not available!")
            sys.exit(1)
            
        # Verify we have enough GPUs
        num_gpus = torch.cuda.device_count()
        if local_rank >= num_gpus:
            print(f"[Rank {local_rank}] ❌ Local rank {local_rank} >= available GPUs {num_gpus}")
            sys.exit(1)
        
        # Set CUDA device first
        torch.cuda.set_device(local_rank)
        device_props = torch.cuda.get_device_properties(local_rank)
        print(f"[Rank {local_rank}] ✅ Using CUDA device: cuda:{local_rank} ({device_props.name})")
        
        # Test CUDA functionality before proceeding
        try:
            test_tensor = torch.ones(1, device=local_rank)
            _ = test_tensor * 2  # Simple operation to verify CUDA works
            del test_tensor
            torch.cuda.empty_cache()
            print(f"[Rank {local_rank}] ✅ CUDA functionality verified")
        except Exception as cuda_test_error:
            print(f"[Rank {local_rank}] ❌ CUDA test failed: {cuda_test_error}")
            sys.exit(1)
        
        # Set comprehensive NCCL environment variables for stability
        nccl_env_vars = {
            'NCCL_SOCKET_IFNAME': '^docker0,lo',  # Avoid network interface issues
            'NCCL_IB_DISABLE': '1',  # Disable InfiniBand if not needed
            'NCCL_P2P_DISABLE': '1',  # Disable P2P if causing issues
            'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',  # Better error handling
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',  # Consistent device ordering
            'NCCL_TREE_THRESHOLD': '0',  # Force ring algorithm for small clusters
            'NCCL_LAUNCH_MODE': 'PARALLEL',  # Parallel launch mode
            'NCCL_TOPO_FILE': '',  # Disable topology detection that causes warnings
            'TORCH_NCCL_USE_COMM_NONBLOCKING': '1',  # Use non-blocking communications
            'NCCL_IGNORE_DISABLED_P2P': '1',  # Ignore P2P warnings
            'NCCL_IGNORE_CPU_AFFINITY': '1',  # Ignore CPU affinity warnings
            'NCCL_DEBUG': 'WARN',  # Only show warnings and errors
            'NCCL_TIMEOUT': '3600',  # 1 hour timeout
            'TORCH_NCCL_BLOCKING_WAIT': '1',  # Use blocking wait
            'NCCL_NET_GDR_LEVEL': '0',  # Disable GPU Direct RDMA
            'NCCL_MIN_NCHANNELS': '1',  # Minimum channels
            'NCCL_MAX_NCHANNELS': '1',  # Maximum channels (force single channel)
        }
        
        for key, value in nccl_env_vars.items():
            os.environ[key] = value
            
        print(f"[Rank {local_rank}] ⚙️ NCCL environment variables configured")
        
    except Exception as e:
        print(f"[Rank {local_rank if 'local_rank' in locals() else 'unknown'}] ❌ Early CUDA setup failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
elif "CUDA_VISIBLE_DEVICES" in os.environ and torch.cuda.is_available():
    # Single GPU or DataParallel mode
    torch.cuda.set_device(0)
    print(f"✅ Using CUDA device: cuda:0 (non-distributed)")
else:
    print("⚠️ No LOCAL_RANK found - running in single GPU/CPU mode")

# 🧼 GPU Memory Management and Process Safety
def cleanup_gpu_memory():
    """Clean up GPU memory and handle CUDA context properly."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    except Exception as e:
        if globals().get('DEBUG_MODE', False):
            print(f"[DEBUG] GPU cleanup warning: {e}")

def cleanup_processes():
    """Clean up processes and GPU memory."""
    try:
        cleanup_gpu_memory()
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        if globals().get('DEBUG_MODE', False):
            print(f"[DEBUG] Process cleanup warning: {e}")

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print(f"🧼 Received signal {signum}, cleaning up...")
    cleanup_processes()
    sys.exit(0)

# Register cleanup functions
atexit.register(cleanup_processes)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def suppress_warnings():
    """Suppress all warnings and unnecessary logs."""
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    """Get the current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def visualize_attention_maps(frames, attention_maps, save_dir, sample_idx, epoch):
    """Visualize attention maps overlaid on input frames."""
    os.makedirs(save_dir, exist_ok=True)
    
    frames = frames.cpu().numpy()
    attention_maps = attention_maps.cpu().numpy()
    
    num_frames = min(5, frames.shape[0])
    
    for t in range(num_frames):
        frame = frames[t].transpose(1, 2, 0)
        frame = (frame * 255).astype(np.uint8)
        
        # Get attention map for the fake class (index 1)
        attention = attention_maps[t, 1]
        
        # Normalize attention map to 0-255
        attention = (attention * 255).astype(np.uint8)
        
        # Apply colormap for visualization
        import cv2
        heatmap = cv2.applyColorMap(attention, cv2.COLORMAP_JET)
        
        # Overlay attention map on frame
        alpha = 0.4
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        # Save visualization
        save_path = os.path.join(save_dir, f'sample_{sample_idx}_frame_{t}_epoch_{epoch}.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def plot_metrics(train_values, val_values, metric_name, epoch, save_dir="plots"):
    """Plot training and validation metrics on the same graph and save to file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_values) + 1), train_values, label=f"Train {metric_name}", color="blue", marker="o")
        plt.plot(range(1, len(val_values) + 1), val_values, label=f"Validation {metric_name}", color="red", marker="o")
        plt.title(f"{metric_name.capitalize()} for Epoch {epoch}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_path = os.path.join(save_dir, f"{metric_name}_epoch_{epoch}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[DEBUG] Plot saved successfully: {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"❌ Error in plot_metrics for {metric_name}: {e}")
        import traceback
        traceback.print_exc()
        # Ensure plot is closed even if there's an error
        plt.close('all')
        return None


def plot_confusion_matrix(y_true, y_pred, epoch, save_dir="plots", split='val'):
    """Plot confusion matrix and save to file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except Exception:
            sns = None

        os.makedirs(save_dir, exist_ok=True)

        # Try using sklearn's confusion_matrix if available, else compute simple numpy-based CM
        try:
            from sklearn.metrics import confusion_matrix as _conf
            cm = _conf(y_true, y_pred, labels=[0, 1])
        except Exception:
            # simple 2x2 confusion
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[int(t), int(p)] += 1

        # Ensure confusion matrix has values (prevent seaborn zero-size array error)
        if cm.size == 0 or cm.sum() == 0:
            print(f"[DEBUG] Skipping confusion matrix plot - no data (cm sum={cm.sum()})")
            return None

        plt.figure(figsize=(8, 6))
        if sns is not None:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, vmin=0, vmax=max(1, cm.max()))
        else:
            plt.imshow(cm, cmap='Blues')
            for (i, j), val in np.ndenumerate(cm):
                plt.text(j, i, int(val), ha='center', va='center')

        split_label = split.capitalize()
        plt.title(f"Confusion Matrix ({split_label}) - Epoch {epoch}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks([0, 1], ["Real", "Fake"])
        plt.yticks([0, 1], ["Real", "Fake"])

        cm_path = os.path.join(save_dir, f"confusion_matrix_{split}_epoch_{epoch}.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[DEBUG] {split_label} confusion matrix saved successfully: {cm_path}")
        return cm_path
        
    except Exception as e:
        print(f"❌ Error in plot_confusion_matrix: {e}")
        import traceback
        traceback.print_exc()
        # Ensure plot is closed even if there's an error
        plt.close('all')
        return None


def calculate_metrics(y_true, y_pred, y_probs, epoch, return_dict=False):
    """Calculate comprehensive metrics including per-class and macro metrics."""
    # Convert lists to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    # Try to import sklearn metrics lazily; provide lightweight fallbacks if missing
    try:
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
        _sklearn_available = True
    except Exception:
        _sklearn_available = False

    if _sklearn_available:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        confusion = confusion_matrix(y_true, y_pred)

        # Per-class metrics (Real=0, Fake=1)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Macro F1 (average of both classes)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        try:
            # Ensure y_probs are valid probabilities in [0,1]. If they look like logits, apply sigmoid.
            y_probs = np.array(y_probs, dtype=float)
            if y_probs.size == 0:
                auc_score = 0.0
            else:
                minp = float(np.min(y_probs))
                maxp = float(np.max(y_probs))
                # If values outside [0,1], assume logits and apply sigmoid
                if minp < 0.0 or maxp > 1.0:
                    try:
                        y_probs = 1.0 / (1.0 + np.exp(-y_probs))
                        print(f"[METRICS] Converted logits to probabilities for AUC (min={minp:.3f}, max={maxp:.3f}).")
                    except Exception:
                        # fallback: clip
                        y_probs = np.clip(y_probs, 0.0, 1.0)
                # Clip tiny numerical issues
                y_probs = np.clip(y_probs, 1e-7, 1.0 - 1e-7)
                auc_score = float(roc_auc_score(y_true, y_probs))
        except Exception:
            auc_score = 0.0
    else:
        # Lightweight fallbacks (binary case)
        accuracy = float(np.mean(y_true == y_pred)) if y_true.size > 0 else 0.0

        # binary precision/recall/f1
        tp = int(np.sum((y_true == 1) & (y_pred == 1))) if y_true.size > 0 else 0
        fp = int(np.sum((y_true == 0) & (y_pred == 1))) if y_true.size > 0 else 0
        fn = int(np.sum((y_true == 1) & (y_pred == 0))) if y_true.size > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # simple confusion matrix
        confusion = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true.tolist(), y_pred.tolist()):
            confusion[int(t), int(p)] += 1

        precision_per_class = [precision, precision]
        recall_per_class = [recall, recall]
        f1_per_class = [f1, f1]
        macro_f1 = float(np.mean(f1_per_class))

        try:
            # Approximate AUC using rank method if possible
            from sklearn.metrics import roc_auc_score as _roc
            auc_score = _roc(y_true, y_probs)
        except Exception:
            auc_score = 0.0

    print(f"Epoch {epoch} Metrics:")
    print(f"- Accuracy    : {accuracy:.4f}")
    print(f"- Precision   : {precision:.4f}")
    print(f"- Recall      : {recall:.4f}")
    print(f"- F1 Score    : {f1:.4f}")
    print(f"- Macro F1    : {macro_f1:.4f} (key)")  # Key metric for imbalanced data
    print(f"- AUC Score   : {auc_score:.4f}")
    
    # Per-class breakdown
    if len(precision_per_class) >= 2:
        print(f"- Real (0): P={precision_per_class[0]:.4f}, R={recall_per_class[0]:.4f}, F1={f1_per_class[0]:.4f}")
        print(f"- Fake (1): P={precision_per_class[1]:.4f}, R={recall_per_class[1]:.4f}, F1={f1_per_class[1]:.4f}")
    
    print(f"- Confusion Matrix:\n{confusion}")
    
    if return_dict:
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_f1': macro_f1,
            'auc': auc_score,
            'confusion_matrix': confusion.tolist(),
            'precision_per_class': precision_per_class.tolist() if len(precision_per_class) > 0 else [],
            'recall_per_class': recall_per_class.tolist() if len(recall_per_class) > 0 else [],
            'f1_per_class': f1_per_class.tolist() if len(f1_per_class) > 0 else []
        }
    else:
        return precision, recall, f1, auc_score, accuracy, macro_f1


# ============================================================================
# EMA (Exponential Moving Average) for better generalization
# ============================================================================
class ModelEMA:
    """Exponential Moving Average of model parameters for better generalization.
    
    Maintains a shadow copy of model parameters that is updated as an
    exponential moving average of the training parameters. The EMA model
    typically generalizes better than the raw trained model.
    
    Tracks both trainable parameters AND BatchNorm running statistics
    to ensure consistent behavior during evaluation.
    """

    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.device = device
        # Create shadow parameters (detached copies) for trainable params
        self.shadow = {}
        self.backup = {}
        # Include ALL parameters (even frozen backbone ones) so that
        # progressive unfreezing works seamlessly with EMA.
        # Frozen params won't be updated in update() until requires_grad=True.
        for name, param in model.named_parameters():
            shadow_param = param.data.clone().detach()
            if device is not None:
                shadow_param = shadow_param.to(device)
            self.shadow[name] = shadow_param
        
        # Also track BatchNorm running statistics (running_mean, running_var)
        self.shadow_buffers = {}
        self.backup_buffers = {}
        for name, buf in model.named_buffers():
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                shadow_buf = buf.data.clone().detach()
                if device is not None:
                    shadow_buf = shadow_buf.to(device)
                self.shadow_buffers[name] = shadow_buf

    @torch.no_grad()
    def update(self, model):
        """Update shadow parameters with EMA of current model parameters and BN stats."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_avg.clone()
        
        # Update BN running stats with EMA as well
        for name, buf in model.named_buffers():
            if name in self.shadow_buffers:
                if 'num_batches_tracked' in name:
                    self.shadow_buffers[name] = buf.data.clone()
                else:
                    new_avg = self.decay * self.shadow_buffers[name] + (1.0 - self.decay) * buf.data
                    self.shadow_buffers[name] = new_avg.clone()

    def apply_shadow(self, model):
        """Replace model parameters with shadow (EMA) parameters for evaluation."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        
        # Also apply BN running stats
        self.backup_buffers = {}
        for name, buf in model.named_buffers():
            if name in self.shadow_buffers:
                self.backup_buffers[name] = buf.data.clone()
                buf.data.copy_(self.shadow_buffers[name])

    def restore(self, model):
        """Restore original model parameters after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
        
        # Restore BN running stats
        for name, buf in model.named_buffers():
            if name in self.backup_buffers:
                buf.data.copy_(self.backup_buffers[name])
        self.backup_buffers = {}


# ============================================================================
# Mixup augmentation for improved generalization
# ============================================================================
def mixup_batch(batch, labels, alpha=0.2):
    """Apply Mixup augmentation to a batch of data.
    
    Linearly interpolates between random pairs of samples and their labels.
    Helps regularization and improves generalization.
    
    Args:
        batch: dict of tensors (video_frames, audio, landmarks, etc.)
        labels: tensor of labels [B]
        alpha: Beta distribution parameter (0 = disabled)
    
    Returns:
        mixed_batch, labels_a, labels_b, lam
    """
    if alpha <= 0:
        return batch, labels, labels, 1.0
    
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 for stability
    
    batch_size = labels.size(0)
    index = torch.randperm(batch_size, device=labels.device)
    
    mixed_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == batch_size and value.is_floating_point():
            mixed_batch[key] = lam * value + (1 - lam) * value[index]
        else:
            mixed_batch[key] = value  # Keep non-tensor / integer fields unchanged
    
    labels_a = labels
    labels_b = labels[index]
    
    return mixed_batch, labels_a, labels_b, lam


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses learning on hard examples and reduces weight of easy examples.
    
    NOTE: alpha here is a global scaling factor. Class-specific weighting is
    handled by class_weights parameter which is passed to CrossEntropyLoss.
    
    For best results with imbalanced datasets:
    - Set alpha=1.0 (neutral global scale)
    - Use class_weights to balance Real vs Fake
    - Use gamma=2.0 to focus on hard examples
    """
    
    def __init__(self, alpha=1.0, gamma=2, class_weights=None, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        # CRITICAL: Default alpha to 1.0 to avoid accidentally scaling down the loss
        self.alpha = alpha if alpha is not None else 1.0
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.debug_counter = 0  # For periodic debugging
        
    def forward(self, inputs, targets):
        # Add numerical stability - clamp logits to prevent extreme values
        inputs = torch.clamp(inputs, min=-50, max=50)
        
        # Debug: Check inputs and targets (disabled to reduce spam)
        # if self.debug_counter % 100 == 0:
        #     print(f"[FOCAL LOSS DEBUG] Input logits shape: {inputs.shape}")
        #     print(f"[FOCAL LOSS DEBUG] Input logits sample: {inputs[:3]}")
        #     print(f"[FOCAL LOSS DEBUG] Targets shape: {targets.shape}")
        #     print(f"[FOCAL LOSS DEBUG] Targets sample: {targets[:8]}")
        #     print(f"[FOCAL LOSS DEBUG] Class weights: {self.class_weights}")
        
        # First, calculate CE loss WITHOUT weights to diagnose the issue
        ce_loss_no_weight = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Then calculate WITH weights, ensuring proper device but KEEP float32 dtype
        if self.class_weights is not None:
            # CRITICAL: Keep weights in float32, don't convert to float16!
            # PyTorch will handle the dtype conversion internally for cross_entropy
            weights = self.class_weights.to(inputs.device, dtype=torch.float32)
            ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction='none', label_smoothing=self.label_smoothing)
        else:
            ce_loss = ce_loss_no_weight
        
        # Debug CE loss (disabled to reduce spam)
        # if self.debug_counter % 100 == 0:
        #     print(f"[FOCAL LOSS DEBUG] CE loss (NO weights) sample: {ce_loss_no_weight[:8]}")
        #     print(f"[FOCAL LOSS DEBUG] CE loss (WITH weights) sample: {ce_loss[:8]}")
        #     print(f"[FOCAL LOSS DEBUG] CE loss (NO weights) range: [{ce_loss_no_weight.min().item():.6f}, {ce_loss_no_weight.max().item():.6f}]")
        
        pt = torch.exp(-ce_loss)
        
        # Clamp pt to prevent (1-pt) from being exactly zero
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # Debug logging every 100 batches (disabled to reduce spam)
        # if self.debug_counter % 100 == 0:
        #     print(f"[FOCAL LOSS DEBUG] CE loss (weighted) range: [{ce_loss.min().item():.6f}, {ce_loss.max().item():.6f}]")
        #     print(f"[FOCAL LOSS DEBUG] pt range: [{pt.min().item():.6f}, {pt.max().item():.6f}]")
        #     print(f"[FOCAL LOSS DEBUG] (1-pt)^gamma range: [{((1-pt)**self.gamma).min().item():.6e}, {((1-pt)**self.gamma).max().item():.6e}]")
        #     print(f"[FOCAL LOSS DEBUG] Focal loss range: [{focal_loss.min().item():.6e}, {focal_loss.max().item():.6e}]")
        self.debug_counter += 1
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def save_visualizations(inputs, outputs, results, epoch, sample_idx, viz_dir, model=None):
    """Save visualizations of model predictions and attention maps."""
    os.makedirs(viz_dir, exist_ok=True)
    # Local (optional) visualization imports to avoid hard dependency at module import
    try:
        import cv2
    except Exception:
        cv2 = None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    
    try:
        # Get input video frames
        video_frames = inputs['video_frames']
        batch_size, num_frames = video_frames.shape[:2]
        
        # Limit to a small number of frames for visualization
        max_viz_frames = min(5, num_frames)
        
        # Extract a sample video
        sample_frames = video_frames[sample_idx, :max_viz_frames].cpu()
        
        # Save original frames (skip if cv2 missing)
        frames_dir = os.path.join(viz_dir, f'sample_{sample_idx}_epoch_{epoch}')
        os.makedirs(frames_dir, exist_ok=True)
        if cv2 is not None:
            for t in range(max_viz_frames):
                frame = sample_frames[t].permute(1, 2, 0).numpy()
                frame = (frame * 255).astype(np.uint8)
                try:
                    cv2.imwrite(
                        os.path.join(frames_dir, f'frame_{t}.jpg'),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    )
                except Exception as _e:
                    print(f"Warning: failed to write frame image: {_e}")
        else:
            print("[VIS] cv2 not available - skipping frame image writes")
        
        # Save model explanation if available
        if 'explanation' in results and results['explanation'] is not None:
            explanation = results['explanation']
            
            # Save highlighted regions if available
            if 'highlighted_regions' in explanation and explanation['highlighted_regions']:
                regions = explanation['highlighted_regions']

                for region in regions:
                    if len(region) >= 3:
                        batch_idx, frame_idx, score = region
                        if batch_idx == sample_idx and frame_idx < max_viz_frames:
                            if cv2 is None:
                                continue
                            frame = sample_frames[frame_idx].permute(1, 2, 0).numpy()
                            frame = (frame * 255).astype(np.uint8)
                            # Add highlight overlay for suspicious regions
                            highlight = np.zeros_like(frame)
                            highlight[:, :, 0] = 255  # Red channel
                            # Apply highlight with transparency
                            alpha = min(0.7, score * 5)  # Scale by score
                            try:
                                highlighted_frame = cv2.addWeighted(
                                    frame, 1 - alpha, highlight, alpha, 0
                                )
                                cv2.imwrite(
                                    os.path.join(frames_dir, f'frame_{frame_idx}_highlighted.jpg'),
                                    cv2.cvtColor(highlighted_frame, cv2.COLOR_RGB2BGR)
                                )
                            except Exception as _e:
                                print(f"Warning: failed to write highlighted frame: {_e}")
            
            # Save explanation to text file
            with open(os.path.join(frames_dir, 'explanation.txt'), 'w') as f:
                f.write(f"Prediction: {'Fake' if outputs.argmax(1)[sample_idx].item() == 1 else 'Real'}\n")
                f.write(f"Confidence: {torch.softmax(outputs, dim=1)[sample_idx][outputs.argmax(1)[sample_idx]].item():.4f}\n\n")
                
                if 'issues_found' in explanation:
                    f.write("Issues found:\n")
                    for issue in explanation['issues_found']:
                        f.write(f"- {issue}\n")
                
                if 'detection_scores' in explanation:
                    f.write("\nDetection scores:\n")
                    for key, value in explanation['detection_scores'].items():
                        f.write(f"- {key}: {value:.4f}\n")
                
                if 'confidence' in explanation:
                    f.write(f"\nOverall confidence: {explanation['confidence']:.4f}\n")
        
        # Generate and save attention maps if model has this capability
        # Accept optional `model` parameter (pass trainer.model when calling).
        # If model is not provided, skip attention-map generation gracefully.
        try:
            model_obj = locals().get('model', None)
            if model_obj is not None and hasattr(model_obj, 'get_attention_maps'):
                with torch.no_grad():
                    attention_maps = model_obj.get_attention_maps(inputs)
                    if attention_maps is not None:
                        visualize_attention_maps(
                            sample_frames,
                            attention_maps[sample_idx],
                            frames_dir,
                            sample_idx,
                            epoch
                        )
        except Exception as e:
            print(f"Error generating attention maps: {e}")
                
        # NEW: Visualize facial landmarks if available
        if 'facial_landmarks' in inputs and inputs['facial_landmarks'] is not None:
            try:
                landmarks = inputs['facial_landmarks'][sample_idx].cpu().numpy()
                
                for t in range(min(max_viz_frames, landmarks.shape[0])):
                    # Load the frame
                    frame_path = os.path.join(frames_dir, f'frame_{t}.jpg')
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        
                        # Draw landmarks
                        lm = landmarks[t]
                        for i in range(0, len(lm), 2):
                            if i+1 < len(lm) and lm[i] > 0 and lm[i+1] > 0:
                                x, y = int(lm[i]), int(lm[i+1])
                                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                        
                        # Save the frame with landmarks
                        cv2.imwrite(
                            os.path.join(frames_dir, f'frame_{t}_landmarks.jpg'),
                            frame
                        )
            except Exception as e:
                print(f"Error visualizing facial landmarks: {e}")
                
        # NEW: Visualize head pose if available
        if 'head_pose' in inputs and inputs['head_pose'] is not None:
            try:
                head_pose = inputs['head_pose'][sample_idx].cpu().numpy()
                
                # Create a plot of head pose over time
                plt.figure(figsize=(10, 5))
                time_axis = np.arange(len(head_pose))
                plt.plot(time_axis, head_pose[:, 0], label='Pitch', color='red')
                plt.plot(time_axis, head_pose[:, 1], label='Yaw', color='green')
                plt.plot(time_axis, head_pose[:, 2], label='Roll', color='blue')
                plt.title(f'Head Pose Over Time - Sample {sample_idx}')
                plt.xlabel('Frame')
                plt.ylabel('Angle (normalized)')
                plt.legend()
                plt.grid(True)
                
                # Save the plot
                plt.savefig(os.path.join(frames_dir, 'head_pose.png'))
                plt.close()
            except Exception as e:
                print(f"Error visualizing head pose: {e}")
                
        # NEW: Visualize eye blink pattern if available
        if 'eye_blink_features' in inputs and inputs['eye_blink_features'] is not None:
            try:
                blink_pattern = inputs['eye_blink_features'][sample_idx].cpu().numpy()
                
                # Create a plot of blink pattern over time
                plt.figure(figsize=(10, 3))
                time_axis = np.arange(len(blink_pattern))
                plt.plot(time_axis, blink_pattern, label='Blink Score', color='purple')
                plt.title(f'Eye Blink Pattern - Sample {sample_idx}')
                plt.xlabel('Frame')
                plt.ylabel('Blink Score')
                plt.ylim([0, 1])
                plt.grid(True)
                
                # Save the plot
                plt.savefig(os.path.join(frames_dir, 'eye_blink_pattern.png'))
                plt.close()
            except Exception as e:
                print(f"Error visualizing eye blink pattern: {e}")
                
        # NEW: Visualize pulse signal if available
        if 'pulse_signal' in inputs and inputs['pulse_signal'] is not None:
            try:
                pulse = inputs['pulse_signal'][sample_idx].cpu().numpy()
                
                # Create a plot of pulse signal over time
                plt.figure(figsize=(10, 3))
                time_axis = np.arange(len(pulse))
                plt.plot(time_axis, pulse, label='Pulse Signal', color='red')
                plt.title(f'Pulse Signal - Sample {sample_idx}')
                plt.xlabel('Frame')
                plt.ylabel('Signal Amplitude')
                plt.grid(True)
                
                # Save the plot
                plt.savefig(os.path.join(frames_dir, 'pulse_signal.png'))
                plt.close()
            except Exception as e:
                print(f"Error visualizing pulse signal: {e}")
                
        # NEW: Visualize frequency domain features if available
        if 'frequency_features' in inputs and inputs['frequency_features'] is not None:
            try:
                freq_features = inputs['frequency_features'][sample_idx].cpu().numpy()
                
                # Create heatmap of frequency features
                import seaborn as sns
                plt.figure(figsize=(6, 6))
                sns.heatmap(freq_features[0], cmap='viridis')
                plt.title(f'Frequency Domain Features - Sample {sample_idx}')
                plt.axis('off')
                
                # Save the plot
                plt.savefig(os.path.join(frames_dir, 'frequency_features.png'))
                plt.close()
            except Exception as e:
                print(f"Error visualizing frequency features: {e}")
                
    except Exception as e:
        print(f"Error saving visualizations: {e}")


def oversample_minority_class(train_loader, minority_class=0, oversample_ratio=2.0):
    """
    Oversample minority class (Real videos) to address class imbalance.
    
    Args:
        train_loader: Original training data loader
        minority_class: Class to oversample (0=Real, 1=Fake)
        oversample_ratio: How much to oversample minority class
    """
    print(f"🔄 Oversampling minority class {minority_class} with ratio {oversample_ratio}")
    
    # This would typically be implemented in the dataset creation
    # For now, we'll just print the intent - actual implementation 
    # would require modifying the data loader creation
    print("⚠️ Oversampling would be implemented in dataset creation phase")
    return train_loader


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        try:
            # Ensure all async CUDA work is finished so memory reads are accurate
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            return allocated, reserved
        except Exception:
            return 0.0, 0.0
    return 0, 0


def _read_trace_toggle_file(path=None):
    """Read trace toggle file. Returns None (no-file), True (enable), False (disable).

    The file should contain a single char: '1' to enable tracing, '0' to disable.
    """
    try:
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'trace_toggle.txt')
        if not os.path.exists(path):
            return None
        with open(path, 'r') as fh:
            txt = fh.read().strip()
        if txt == '1':
            return True
        if txt == '0':
            return False
        return None
    except Exception:
        return None

def check_memory_and_reduce_batch_size(batch, max_retries=3, trainer=None):
    """Check memory usage and reduce batch size if needed."""
    current_batch_size = len(batch['label']) if 'label' in batch else 1
    
    for retry in range(max_retries):
        try:
            allocated, reserved = get_gpu_memory_usage()
            if allocated > 35.0:  # If using more than 35GB, reduce batch size
                new_batch_size = max(1, current_batch_size // 2)
                if globals().get('DEBUG_MODE', False):
                    print(f"[MEMORY] Reducing batch size from {current_batch_size} to {new_batch_size} (allocated: {allocated:.2f}GB)")
                # Track memory warnings
                if trainer is not None:
                    trainer.memory_warnings += 1
                # Consistently truncate all batch elements
                reduced_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and value.size(0) == current_batch_size:
                        reduced_batch[key] = value[:new_batch_size]
                    elif isinstance(value, list) and len(value) == current_batch_size:
                        reduced_batch[key] = value[:new_batch_size]
                    else:
                        reduced_batch[key] = value
                # Verify batch consistency after reduction
                actual_batch_size = len(reduced_batch['label']) if 'label' in reduced_batch else new_batch_size
                if actual_batch_size != new_batch_size and globals().get('DEBUG_MODE', False):
                    print(f"[WARNING] Batch size inconsistency after reduction: expected {new_batch_size}, got {actual_batch_size}")
                batch = reduced_batch
                current_batch_size = new_batch_size
                clear_gpu_cache()
            else:
                break
        except Exception as e:
            if globals().get('DEBUG_MODE', False):
                print(f"[MEMORY] Error checking memory usage: {e}")
            break
    return batch

def move_batch_to_device(batch, device, trainer=None):
    """Safely move batch items to device, handling non-tensor items properly."""
    # Check and optimize memory before moving to device
    batch = check_memory_and_reduce_batch_size(batch, trainer=trainer)
    
    device_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            device_batch[key] = value.to(device, non_blocking=True)
        elif value is None:
            device_batch[key] = None
        elif isinstance(value, list):
            # Keep lists as they are
            device_batch[key] = value
        else:
            # Try to convert other types to tensor if possible
            try:
                device_batch[key] = torch.tensor(value).to(device)
            except:
                device_batch[key] = value
    
    return device_batch


class DeepfakeTrainer:
    def check_and_adjust_gpu_memory(self):
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.id == 0:  # Only check GPU 0
                    dedicated = gpu.memoryUsed  # in MB
                    shared = gpu.memoryUtil * gpu.memoryTotal - dedicated  # approx shared
                    if dedicated > 7900 or shared > 15700:
                        print(f"[WARNING] GPU memory high: dedicated={dedicated/1024:.2f}GB, shared~={shared/1024:.2f}GB. Reducing batch size.")
                        if hasattr(self.config, 'batch_size') and self.config.batch_size > 1:
                            self.config.batch_size = max(1, self.config.batch_size - 1)
                            print(f"[ACTION] Batch size reduced to {self.config.batch_size}")
                        else:
                            print("[INFO] Batch size already at minimum.")
        except Exception as e:
            print(f"[GPU CHECK ERROR] {e}")

    """Class to manage the training, validation, and testing of the multimodal deepfake detection model."""

    def __init__(self, config):
        """Initialize the trainer with the given configuration."""
        self.config = config
        self.distributed = config.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1
        self.local_rank = config.local_rank
        self.is_main_process = not self.distributed or self.local_rank == 0
        self.debug = getattr(config, 'debug', False)  # Debug mode flag
        # Debug snapshot behavior: save model/batch when anomaly detected
        self.debug_save_on_anomaly = getattr(config, 'debug_save_on_anomaly', False)
        
        # Set device properly for distributed training
        if self.distributed:
            # In distributed mode, set CUDA device immediately to avoid warnings
            print(f"[Rank {self.local_rank}] ⚙️ Setting CUDA device in constructor...")
            torch.cuda.set_device(self.local_rank)
            
            # Ensure device is properly initialized by PyTorch
            current_device = torch.cuda.current_device()
            if current_device != self.local_rank:
                print(f"[Rank {self.local_rank}] ⚠️ Device mismatch: expected {self.local_rank}, got {current_device}")
                torch.cuda.set_device(self.local_rank)  # Try again
            
            self.device = torch.device(f'cuda:{self.local_rank}')
            print(f"[Rank {self.local_rank}] ✅ Set CUDA device: {self.device}")
        else:
            self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.amp_enabled = config.amp_enabled and self.device.type == 'cuda'
        # Allow disabling AMP globally for debug runs
        if getattr(config, 'debug_disable_amp', False):
            print("[DEBUG] AMP disabled via config.debug_disable_amp")
            self.amp_enabled = False
        
        # Set up directories
        self.setup_directories()
        
        # Initialize QAT attributes BEFORE print_training_improvements()
        self.qat_enabled = config.enable_qat
        self.qat_start_epoch = config.qat_start_epoch if config.enable_qat else 999
        self.qat_active = False
        
        # Print configuration improvements
        self.print_training_improvements()
        # (Removed misplaced epoch-level diagnostics that referenced epoch local variables.)
        # Set up run directories (fix indentation)
        run_dirs = [d for d in os.listdir(self.config.output_dir) if d.startswith("run_")]
        if run_dirs:
            self.run_dir = max(run_dirs, key=lambda d: os.path.getctime(os.path.join(self.config.output_dir, d)))
            self.run_dir = os.path.join(self.config.output_dir, self.run_dir)
            self.log_dir = os.path.join(self.run_dir, "logs")
            self.viz_dir = os.path.join(self.run_dir, "visualizations")
            self.plot_dir = os.path.join(self.run_dir, "plots")
            print(f"[Rank {self.local_rank}] Using shared run directory: {self.run_dir}")
        else:
            print(f"[Rank {self.local_rank}] Warning: No run directory found, using fallback")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(self.config.output_dir, f"run_{timestamp}")
            self.log_dir = os.path.join(self.run_dir, "logs")
            self.viz_dir = os.path.join(self.run_dir, "visualizations")
            self.plot_dir = os.path.join(self.run_dir, "plots")
            
            # Another barrier to ensure all processes have the correct paths
            if dist.is_initialized():
                dist.barrier()
        
        # Initialize wandb if enabled
        if config.use_wandb and self.is_main_process:
            # For distributed training, disable wandb by default to avoid conflicts
            if self.distributed and not config.use_wandb:
                print(f"[Rank {self.local_rank}] Note: W&B disabled for distributed training (use --use_wandb to force enable)")
                self.config.use_wandb = False
            else:
                self.setup_wandb()
        else:
            self.config.use_wandb = False
        
        # Save configuration
        self.save_config()
        
        # Set seed for reproducibility
        set_seed(config.seed)

        # If user requested faster (less deterministic) mode, enable cudnn benchmark
        # This allows PyTorch to autotune cuDNN kernels for better throughput.
        if getattr(config, 'fast_mode', False):
            try:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                print("[PERF] fast_mode enabled: cudnn.benchmark=True, deterministic=False")
            except Exception as e:
                print(f"[PERF] Failed to enable fast_mode cudnn settings: {e}")
        
        # Initialize data loaders
        self.setup_data()
        
        # Initialize model, optimizer, and scheduler
        self.setup_model()
        
        # Initialize metrics tracking
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'train_f1_scores': [],
            'val_f1_scores': [],
            'train_macro_f1_scores': [],  # Key metric for imbalanced data
            'val_macro_f1_scores': [],    # Key metric for imbalanced data
            'train_auc_scores': [],
            'val_auc_scores': []
        }
        
        # Set up logging to file if specified
        self.log_file = None
        if self.config.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.config.log_file)), exist_ok=True)
            self.log_file = open(self.config.log_file, 'w', encoding='utf-8')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"=== DeepFake Detection Training Log - {timestamp} ===\n")
            self.log_file.write(f"Configuration: {vars(self.config)}\n\n")
            self.log_file.flush()
        
        # Initialize numerical stability tracking
        self.nan_count = 0
        self.total_batches = 0
        
        # Initialize memory management tracking
        self.memory_warnings = 0
        self.oom_errors = 0
        
        # Initialize best model tracking
        self.best_val_accuracy = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0
        
        # ====== QUANTIZATION-AWARE TRAINING (QAT) SETUP ======
        # QAT attributes already initialized earlier (before print_training_improvements)
        self.model_fp32 = None  # Store original FP32 model before QAT
        
        if self.qat_enabled and self.is_main_process:
            print("\n" + "="*80)
            print("🔧 QUANTIZATION-AWARE TRAINING (QAT) ENABLED")
            print("="*80)
            print(f"   QAT will start at epoch: {self.qat_start_epoch}")
            print(f"   QAT backend: {config.qat_backend}")
            print(f"   QAT learning rate scale: {config.qat_lr_scale}x")
            print(f"   Benefits: 4x smaller model, 2-4x faster inference")
            print("="*80 + "\n")
        
        # Initialize scaler for mixed precision
        self.scaler = GradScaler() if self.amp_enabled else None
        # Gradient accumulation steps (default 1)
        self.grad_accum_steps = max(1, int(getattr(self.config, 'grad_accum_steps', 1)))
        
        # Initialize run checkpoint directory and handle resume checkpoint
        self._init_run_checkpoint_dir(config)
        
        # One-line config dump for quick debugging
        if self.is_main_process:
            try:
                print(
                    f"[DEBUG_CONFIG] grad_accum_steps={self.grad_accum_steps} "
                    f"debug_disable_amp={getattr(self.config, 'debug_disable_amp', False)} "
                    f"debug_autograd_detect={getattr(self.config, 'debug_autograd_detect', False)} "
                    f"debug_strict_clip={getattr(self.config, 'debug_strict_clip', False)} "
                    f"debug_log_grad_stats={getattr(self.config, 'debug_log_grad_stats', False)} "
                    f"debug_log_aux_losses={getattr(self.config, 'debug_log_aux_losses', False)} "
                    f"debug_save_on_anomaly={getattr(self.config, 'debug_save_on_anomaly', False)}"
                )
            except Exception:
                pass

    def _save_debug_snapshot(self, tag, index, batch, outputs, loss):
        """Save model/optimizer/batch snapshot for post-mortem debugging.
        tag: short string (e.g., 'anomaly' or 'pairwise')
        index: batch index or step
        """
        try:
            import time
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            snap_dir = os.path.join(self.run_dir if hasattr(self, 'run_dir') and self.run_dir else self.config.output_dir, 'debug_snapshots')
            os.makedirs(snap_dir, exist_ok=True)
            snap_path = os.path.join(snap_dir, f"snapshot_{tag}_idx{index}_{ts}.pth")
            to_save = {
                'timestamp': ts,
                'tag': tag,
                'index': int(index) if isinstance(index, (int, float)) else str(index),
                'loss': float(loss.detach().cpu().item()) if hasattr(loss, 'detach') else float(loss) if isinstance(loss, (int, float)) else None,
            }
            # Save model and optimizer state (best-effort)
            try:
                to_save['model_state_dict'] = {k: v.cpu() for k, v in self.model.state_dict().items()}
            except Exception:
                to_save['model_state_dict'] = None
            try:
                to_save['optimizer_state_dict'] = {k: v for k, v in self.optimizer.state_dict().items()}
            except Exception:
                to_save['optimizer_state_dict'] = None

            # Save batch (may contain non-tensor objects) - try to save tensors only
            batch_tensors = {}
            try:
                if isinstance(batch, dict):
                    for k, v in batch.items():
                        try:
                            if isinstance(v, torch.Tensor):
                                batch_tensors[k] = v.detach().cpu()
                        except Exception:
                            pass
                elif isinstance(batch, torch.Tensor):
                    batch_tensors['batch'] = batch.detach().cpu()
            except Exception:
                batch_tensors = {}
            to_save['batch_tensors'] = batch_tensors

            # Save outputs if tensor
            try:
                if isinstance(outputs, torch.Tensor):
                    to_save['outputs'] = outputs.detach().cpu()
                else:
                    to_save['outputs'] = None
            except Exception:
                to_save['outputs'] = None

            # Save grad norms
            try:
                grad_norms = {}
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        grad_norms[name] = float(p.grad.detach().abs().max().cpu().item())
                to_save['grad_max_abs'] = grad_norms
            except Exception:
                to_save['grad_max_abs'] = None

            torch.save(to_save, snap_path)
            print(f"[DEBUG] Saved anomaly snapshot: {snap_path}")
        except Exception as e:
            print(f"[DEBUG] Failed to save snapshot: {e}")
    
    def _init_run_checkpoint_dir(self, config):
        """Initialize run checkpoint directory and handle resume checkpoint."""
        # Create a specific run folder in the checkpoint directory
        # In distributed training, only main process creates the directory
        if self.distributed:
            # Use the same timestamp across all processes by broadcasting from rank 0
            if self.local_rank == 0:
                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                self.timestamp = None
            
            # Synchronize timestamp across all processes (simplified approach)
            import time
            if self.local_rank == 0:
                # Main process creates timestamp file
                timestamp_file = os.path.join(self.config.checkpoint_dir, "current_run_timestamp.txt")
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                with open(timestamp_file, 'w') as f:
                    f.write(self.timestamp)
            
            # Wait a bit and then all processes read the same timestamp
            time.sleep(1)
            timestamp_file = os.path.join(self.config.checkpoint_dir, "current_run_timestamp.txt")
            if os.path.exists(timestamp_file):
                with open(timestamp_file, 'r') as f:
                    self.timestamp = f.read().strip()
            else:
                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"run_{self.timestamp}")
        
        # Only main process creates the directory in distributed training
        if not self.distributed or self.is_main_process:
            os.makedirs(self.run_checkpoint_dir, exist_ok=True)
        
        # Initialize starting epoch and batch
        self.start_epoch = 0
        self.start_batch = 0
        
        # Check for resume checkpoint
        if hasattr(config, 'resume_checkpoint') and config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training state."""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint file not found at {checkpoint_path}")
            return
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Load checkpoint on appropriate device
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model weights
            if self.distributed:
                self.model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Load optimizer state (skip if resume_weights_only)
            weights_only = getattr(self.config, 'resume_weights_only', False)
            if weights_only:
                print(f"⚠️  resume_weights_only=True: Skipping optimizer/scheduler state, using fresh hyperparameters")
            else:
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load scheduler state if available
                if self.scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Update metrics and best values if available
            if 'accuracy' in checkpoint:
                self.best_val_accuracy = checkpoint['accuracy']
            
            if 'f1_score' in checkpoint:
                self.best_val_f1 = checkpoint['f1_score']
            
            # Handle epoch resumption (skip if weights_only - start fresh)
            if weights_only:
                # When only loading weights, do not overwrite any existing "best" metrics
                # if the checkpoint contains them — preserve for early-stopping and saving.
                self.start_epoch = 0
                self.start_batch = 0
                if 'accuracy' in checkpoint:
                    try:
                        self.best_val_accuracy = float(checkpoint.get('accuracy', self.best_val_accuracy))
                    except Exception:
                        pass
                if 'f1_score' in checkpoint:
                    try:
                        self.best_val_f1 = float(checkpoint.get('f1_score', self.best_val_f1))
                    except Exception:
                        pass
                # Also preserve the epoch number if available (useful for bookkeeping)
                try:
                    self.best_epoch = int(checkpoint.get('epoch', self.best_epoch))
                except Exception:
                    pass
                print(f"[CHECKPOINT] resume_weights_only=True: Loaded weights and preserved best metrics (acc={self.best_val_accuracy:.4f}, f1={self.best_val_f1:.4f}, epoch={self.best_epoch})")
            else:
                # Quick compatibility check: ensure checkpoint final layer matches model output shape
                try:
                    model_state = checkpoint.get('model_state_dict', checkpoint)
                    # Look for classifier weight parameter names commonly 'fc.weight' or 'classifier.weight' or final linear
                    final_w = None
                    for key in ['classifier.weight', 'fc.weight', 'head.weight', 'linear.weight', 'out.weight']:
                        if key in model_state:
                            final_w = model_state[key]
                            break
                    if final_w is not None:
                        ckpt_out_dim = final_w.shape[0]
                        # try infer model output dim from model's last linear
                        try:
                            # find model parameter that looks like final linear weight
                            model_final = None
                            for name, p in self.model.named_parameters():
                                if name.endswith('weight') and p.dim() == 2:
                                    model_final = p
                            if model_final is not None:
                                model_out_dim = list(model_final.shape)[0]
                                if model_out_dim != ckpt_out_dim:
                                    print(f"[CHECKPOINT] Warning: checkpoint output dim ({ckpt_out_dim}) != model output dim ({model_out_dim}). Consider using --resume_weights_only or adjusting model num_classes.")
                        except Exception:
                            pass
                except Exception:
                    # Non-fatal: compatibility check failed
                    print(f"[CHECKPOINT] Compatibility check failed (continuing): {traceback.format_exc() if 'traceback' in globals() else 'see logs'}")

                # Decide whether this is an intermediate (within-epoch) checkpoint or a regular epoch checkpoint
                if isinstance(checkpoint, dict) and 'batch' in checkpoint:
                    # This is an intermediate checkpoint - resume from within the epoch
                    self.start_epoch = checkpoint.get('epoch', 1) - 1  # Convert to 0-indexed
                    self.start_batch = checkpoint.get('batch', 0) + 1  # Resume from next batch
                    print(f"🔄 Intermediate checkpoint detected")
                    print(f"📍 Will resume from epoch {self.start_epoch + 1}, batch {self.start_batch}")
                else:
                    # This is a regular epoch checkpoint - start next epoch
                    self.start_epoch = checkpoint.get('epoch', 0)
                    self.start_batch = 0
                    print(f"🔄 Epoch checkpoint detected")
                    print(f"📍 Will resume from epoch {self.start_epoch + 1}, batch 0")
            
            # Load metrics history if available (skip if weights_only)
            if 'metrics' in checkpoint and not weights_only:
                self.metrics = checkpoint['metrics']
            
            print(f"✅ Checkpoint loaded successfully!")
            print(f"📊 Previous best metrics - Accuracy: {self.best_val_accuracy:.4f}, F1: {self.best_val_f1:.4f}")
        
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
        
    def setup_directories(self):
        """Set up directories for saving models, logs, and visualizations."""
        # Only main process creates directories in distributed training
        if self.distributed and not self.is_main_process:
            return
            
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.config.output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create checkpoint directory (using the specified path)
        self.model_dir = self.config.checkpoint_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create subdirectories
        self.log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.viz_dir = os.path.join(self.run_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        
        print(f"Run directory created at: {self.run_dir}")
        print(f"Checkpoint directory created at: {self.model_dir}")
    
    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        try:
            # Import wandb lazily to avoid hard dependency at module import
            global wandb
            try:
                import wandb as _wandb
                wandb = _wandb
            except Exception as _e:
                print(f"[Rank {self.local_rank}] Warning: wandb import failed: {_e} - disabling wandb")
                self.config.use_wandb = False
                return

            # Set wandb to offline mode for distributed training to avoid conflicts
            if self.distributed:
                os.environ['WANDB_MODE'] = 'offline'
                os.environ['WANDB_SILENT'] = 'true'

            project_name = self.config.wandb_project or "deepfake-detection"
            run_name = self.config.wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Define detailed tags for better tracking
            tags = [
                f"backbone_visual:{self.config.backbone_visual}",
                f"backbone_audio:{self.config.backbone_audio}",
                f"fusion:{self.config.fusion_type}"
            ]

            # Add enhanced feature tags
            if hasattr(self.config, 'detect_faces') and self.config.detect_faces:
                tags.append("facial_analysis")
            if hasattr(self.config, 'enhanced_preprocessing') and self.config.enhanced_preprocessing:
                tags.append("enhanced_preprocessing")
            if hasattr(self.config, 'enhanced_augmentation') and self.config.enhanced_augmentation:
                tags.append("enhanced_augmentation")

            # Add distributed training tag
            if self.distributed:
                try:
                    tags.append(f"distributed_{dist.get_world_size()}gpus")
                except Exception:
                    pass

            wandb.init(
                project=project_name,
                name=run_name,
                config=vars(self.config),
                tags=tags,
                mode='offline' if self.distributed else 'online',  # Offline mode for distributed
                settings=wandb.Settings(start_method="thread")  # Use thread start method
            )

            print(f"[Rank {self.local_rank}] Weights & Biases initialized: {project_name}/{run_name}")
            if self.distributed:
                print(f"[Rank {self.local_rank}] Note: W&B running in offline mode for distributed training")
        except Exception as e:
            print(f"[Rank {self.local_rank}] Error initializing Weights & Biases: {e}")
            self.config.use_wandb = False
    
    def save_config(self):
        """Save configuration to JSON file."""
        # Only main process saves config in distributed training
        if self.distributed and not self.is_main_process:
            return
            
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            # Convert config to dict and handle non-serializable values
            config_dict = vars(self.config).copy()
            for key, value in config_dict.items():
                if not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    config_dict[key] = str(value)
            
            json.dump(config_dict, f, indent=4)
        
        print(f"Configuration saved to: {config_path}")
    
    def setup_data(self):
        """Initialize data loaders for training, validation, and testing."""
        print("Setting up data loaders...")
        
        # Initialize distributed training BEFORE setting up data loaders
        if self.distributed:
            try:
                # 🔧 ROBUST DISTRIBUTED INITIALIZATION
                print(f"[Rank {self.local_rank}] 🚀 Starting distributed process group initialization...")
                
                # Step 1: Verify CUDA device is properly set
                torch.cuda.set_device(self.local_rank)
                current_device = torch.cuda.current_device()
                if current_device != self.local_rank:
                    print(f"[Rank {self.local_rank}] ❌ Device mismatch: expected {self.local_rank}, got {current_device}")
                    raise RuntimeError(f"Device setting failed: expected {self.local_rank}, got {current_device}")
                
                # Step 2: Force CUDA context creation and test basic operations
                try:
                    with torch.cuda.device(self.local_rank):
                        # Force CUDA context creation on the correct device
                        test_tensor = torch.zeros(10, device=self.local_rank)
                        test_result = test_tensor.sum()  # Force computation
                        print(f"[Rank {self.local_rank}] ✅ CUDA context initialized (test result: {test_result.item()})")
                        del test_tensor, test_result
                        torch.cuda.empty_cache()
                except Exception as cuda_context_error:
                    print(f"[Rank {self.local_rank}] ❌ CUDA context creation failed: {cuda_context_error}")
                    raise
                
                self.device = torch.device(f'cuda:{self.local_rank}')
                print(f"[Rank {self.local_rank}] ✅ CUDA device confirmed: {self.device}")
                
                # Step 3: Initialize process group with comprehensive error handling
                if not dist.is_initialized():
                    # Additional NCCL safety settings
                    additional_nccl_vars = {
                        'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',  # Use new non-deprecated variable
                        'TORCH_NCCL_DESYNC_DEBUG': '1',  # Use new non-deprecated variable
                        'NCCL_CUMEM_ENABLE': '0',  # Disable CUDA memory pooling
                        'NCCL_BUFFSIZE': '2097152',  # 2MB buffer
                        'NCCL_NTHREADS': '1',  # Single thread per rank
                        'NCCL_CHECKS_DISABLE': '0',  # Enable all checks
                        'NCCL_CHECK_POINTERS': '1',  # Check pointers
                    }
                    
                    for key, value in additional_nccl_vars.items():
                        os.environ[key] = value
                    
                    print(f"[Rank {self.local_rank}] ⚙️ Additional NCCL safety settings applied")
                    
                    # Initialize with multiple fallback strategies
                    init_strategies = [
                        {'timeout': 1800, 'backend': 'nccl'},
                        {'timeout': 3600, 'backend': 'nccl'},  # Longer timeout
                    ]
                    
                    init_success = False
                    for strategy_idx, strategy in enumerate(init_strategies):
                        try:
                            print(f"[Rank {self.local_rank}] � Trying initialization strategy {strategy_idx + 1}/{len(init_strategies)} (timeout: {strategy['timeout']}s)...")
                            
                            # Synchronize before initialization attempt
                            torch.cuda.synchronize(self.local_rank)
                            
                            with torch.cuda.device(self.local_rank):
                                start_time = time.time()
                                dist.init_process_group(
                                    backend=strategy['backend'],
                                    init_method='env://',
                                    timeout=timedelta(seconds=strategy['timeout'])
                                )
                                init_time = time.time() - start_time
                                
                            print(f"[Rank {self.local_rank}] ✅ Process group initialized successfully in {init_time:.2f}s")
                            init_success = True
                            break
                            
                        except Exception as init_error:
                            print(f"[Rank {self.local_rank}] ❌ Strategy {strategy_idx + 1} failed: {init_error}")
                            if strategy_idx < len(init_strategies) - 1:
                                print(f"[Rank {self.local_rank}] 🔄 Trying next strategy...")
                                time.sleep(2)  # Wait before retry
                            else:
                                print(f"[Rank {self.local_rank}] 💥 All initialization strategies failed")
                                raise init_error
                    
                    if not init_success:
                        raise RuntimeError("Failed to initialize process group with any strategy")
                
                # Step 4: Verify process group functionality
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    print(f"[Rank {self.local_rank}] ✅ Process group verified: rank {rank}/{world_size}")
                    
                    # Test basic communication
                    try:
                        test_tensor = torch.ones(1, device=self.local_rank) * self.local_rank
                        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
                        expected_sum = sum(range(world_size))
                        if abs(test_tensor.item() - expected_sum) < 1e-6:
                            print(f"[Rank {self.local_rank}] ✅ Communication test passed (sum: {test_tensor.item()})")
                        else:
                            print(f"[Rank {self.local_rank}] ⚠️ Communication test failed: got {test_tensor.item()}, expected {expected_sum}")
                        del test_tensor
                    except Exception as comm_test_error:
                        print(f"[Rank {self.local_rank}] ⚠️ Communication test failed: {comm_test_error}")
                        # Don't fail here, just warn
                    
                    # Synchronization barrier with timeout
                    try:
                        print(f"[Rank {self.local_rank}] 🔄 Waiting at synchronization barrier...")
                        barrier_start = time.time()
                        dist.barrier()
                        barrier_time = time.time() - barrier_start
                        print(f"[Rank {self.local_rank}] ✅ Passed synchronization barrier in {barrier_time:.2f}s")
                    except Exception as barrier_error:
                        print(f"[Rank {self.local_rank}] ⚠️ Barrier synchronization failed: {barrier_error}")
                        # Continue without barrier - not always critical
                else:
                    raise RuntimeError("Process group initialization reported success but dist.is_initialized() returns False")
                
            except Exception as distributed_error:
                print(f"[Rank {self.local_rank}] 🚨 Distributed setup failed completely: {distributed_error}")
                print(f"[Rank {self.local_rank}] � Error details:")
                traceback.print_exc()
                
                # Attempt graceful fallback to single-GPU mode
                print(f"[Rank {self.local_rank}] 🔄 Attempting graceful fallback to single-GPU mode...")
                try:
                    self.distributed = False
                    self.is_main_process = True  # Make this process the main one
                    self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
                    torch.cuda.set_device(self.local_rank if torch.cuda.is_available() else 0)
                    print(f"[Rank {self.local_rank}] ✅ Fallback successful - running in single-GPU mode on {self.device}")
                except Exception as fallback_error:
                    print(f"[Rank {self.local_rank}] ❌ Fallback to single-GPU mode also failed: {fallback_error}")
                    raise distributed_error  # Re-raise the original error
        
        # 🧼 OPTIMIZED: Check num_workers setting based on benchmarks
        if self.config.num_workers > 0:
            print(f"⚠️  EXPERIMENTAL: Using num_workers={self.config.num_workers}")
            print("📊 Benchmarks show num_workers=0 is fastest for this multimodal dataset")
            print("🧼 If training is slow, try --num_workers=0")
        else:
            print(f"✅ OPTIMAL: Using num_workers=0 (benchmarked as fastest for multimodal data)")
            print("🚀 Single-threaded loading avoids I/O contention with complex preprocessing")
        
        # Get data loaders with appropriate options
        result = get_data_loaders(
            json_path=self.config.json_path,
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            test_split=self.config.test_split,
            shuffle=True,
            num_workers=self.config.num_workers,
            max_samples=self.config.max_samples,
            detect_faces=self.config.detect_faces,
            compute_spectrograms=self.config.compute_spectrograms,
            temporal_features=self.config.temporal_features,
            enhanced_preprocessing=getattr(self.config, 'enhanced_preprocessing', True),
            enhanced_augmentation=getattr(self.config, 'enhanced_augmentation', False),
            multiprocessing_context="spawn",  # 🧼 SAFETY: Prevent GPU context corruption in multiprocessing
            class_weights_mode=getattr(self.config, 'class_weights_mode', 'balanced'),  # ✅ FIX: Pass class_weights_mode
            oversample_minority=getattr(self.config, 'oversample_minority', False),
            oversample_factor=getattr(self.config, 'oversample_factor', 1.0),
            max_frames=getattr(self.config, 'max_frames', None)
        )
        
        # Unpack result - handle optional mixup_fn
        if len(result) == 5:
            self.train_loader, self.val_loader, self.test_loader, self.class_weights, self.mixup_fn = result
            print("✅ Using MixUp/CutMix augmentation")
        else:
            self.train_loader, self.val_loader, self.test_loader, self.class_weights = result
            self.mixup_fn = None
        
        # CRITICAL DEBUG: Check class weights immediately after assignment from dataset loader
        print(f"[CRITICAL DEBUG] Class weights from dataset_loader: {self.class_weights}")
        print(f"[CRITICAL DEBUG] Dtype from dataset_loader: {self.class_weights.dtype if self.class_weights is not None else 'None'}")
        # Verify oversampling / sampler status
        try:
            oversample_flag = getattr(self.config, 'oversample_minority', False)
            if oversample_flag:
                print(f"[DATA LOADER] oversample_minority=True (requested)")
            # Inspect sampler on train_loader
            try:
                sampler = getattr(self.train_loader, 'sampler', None)
                if sampler is not None:
                    print(f"[DATA LOADER] train_loader.sampler: {sampler.__class__.__name__}")
                    # If WeightedRandomSampler, try to log basic weight stats
                    if sampler.__class__.__name__ == 'WeightedRandomSampler' and hasattr(sampler, 'weights'):
                        import numpy as _np
                        w = _np.array(list(sampler.weights))
                        print(f"[DATA LOADER] WeightedRandomSampler weights: min={w.min():.4g}, max={w.max():.4g}, mean={w.mean():.4g}")
            except Exception as _e:
                print(f"[DATA LOADER] Could not inspect sampler: {_e}")
        except Exception:
            pass
        
        print(f"✅ Data loaders created: {len(self.train_loader)} train batches, "
              f"{len(self.val_loader)} validation batches, {len(self.test_loader)} test batches")
    
    def setup_model(self):
        """Initialize model, loss function, optimizer, and scheduler."""
        print(f"Initializing model on {self.device}...")

        # Move class weights to device - CRITICAL: Preserve float32 dtype!
        if self.class_weights is not None:
            print(f"[DEBUG] Class weights BEFORE device transfer: {self.class_weights}")
            print(f"[DEBUG] Class weights dtype BEFORE: {self.class_weights.dtype}")
            # Ensure weights stay in float32, not float16
            self.class_weights = self.class_weights.to(device=self.device, dtype=torch.float32)
            print(f"[DEBUG] Class weights AFTER device transfer: {self.class_weights}")
            print(f"[DEBUG] Class weights dtype AFTER: {self.class_weights.dtype}")
        else:
            self.class_weights = None

        # Initialize model
        # Allow enabling model-level tensor tracing via environment variable TRACE_TENSORS=1
        enable_trace = os.environ.get('TRACE_TENSORS', '0') == '1'
        if enable_trace:
            print("[TRACE] TRACE_TENSORS=1 detected — model will be constructed with debug/tracing enabled")
        
        self.model = MultiModalDeepfakeModel(
            num_classes=self.config.num_classes,
            video_feature_dim=self.config.video_feature_dim,
            audio_feature_dim=self.config.audio_feature_dim,
            transformer_dim=self.config.transformer_dim,
            num_transformer_layers=self.config.num_transformer_layers,
            enable_face_mesh=self.config.enable_face_mesh,
            enable_explainability=self.config.enable_explainability,
            fusion_type=self.config.fusion_type,
            backbone_visual=self.config.backbone_visual,
            backbone_audio=self.config.backbone_audio,
            use_spectrogram=self.config.use_spectrogram,
            detect_deepfake_type=self.config.detect_deepfake_type,
            num_deepfake_types=self.config.num_deepfake_types,
            debug=(self.config.debug or enable_trace),
            enable_skin_color_analysis=getattr(self.config, 'enable_skin_color_analysis', False),
            enable_advanced_physiological=getattr(self.config, 'enable_advanced_physiological', False),
            deployment_mode=False  # Use training mode to enable contrastive learning
            # Note: Dropout regularization is handled within the model architecture
            # Weight decay is handled by the optimizer (L2 regularization)
        )

        # Load pretrained weights if specified
        if self.config.pretrained_path is not None and os.path.exists(self.config.pretrained_path):
            print(f"Loading pretrained weights from: {self.config.pretrained_path}")
            state_dict = torch.load(self.config.pretrained_path, map_location=self.device)
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            # Handle partial loading with mismatched keys
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained weights")

        # Move model to device
        self.model.to(self.device)

        # ============================================================================
        # MODEL PARAMETER SUMMARY
        # ============================================================================
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Calculate model size in MB (float32 = 4 bytes per parameter)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        print("\n" + "="*70)
        print("📊 MODEL PARAMETER SUMMARY")
        print("="*70)
        print(f"Total Parameters:       {total_params:,}")
        print(f"Trainable Parameters:   {trainable_params:,}")
        print(f"Non-trainable Params:   {non_trainable_params:,}")
        print(f"Model Size (FP32):      {model_size_mb:.2f} MB")
        print(f"Expected Checkpoint:    ~{model_size_mb * 2:.2f} MB (with optimizer state)")
        
        # Component breakdown
        print("\n📦 Component Breakdown:")
        component_params = {}
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_params[name] = params
                pct = (params / total_params) * 100
                print(f"  {name:30s}: {params:>12,} ({pct:5.1f}%)")
        
        # Show top 5 largest components
        if component_params:
            sorted_components = sorted(component_params.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\n🔝 Top 5 Largest Components:")
            for name, params in sorted_components:
                pct = (params / total_params) * 100
                print(f"  {name:30s}: {params:>12,} ({pct:5.1f}%)")
        
        print("="*70 + "\n")
        
        # BIAS INITIALIZATION REMOVED: Was setting bias to ±0.30 which overpowered feature learning
        # Now using default PyTorch initialization (zero bias) and letting class_weights handle imbalance

        # Initialize in distributed mode if specified
        if self.distributed:
            # Ensure the model is on the correct device before wrapping with DDP
            print(f"[Rank {self.local_rank}] Moving model to device: {self.device}")
            
            # Explicitly set the device before DDP wrapping to avoid GPU mapping warnings
            torch.cuda.set_device(self.local_rank)
            self.model = self.model.to(self.device)
            # Wrap model with DDP, explicitly specifying device_ids to avoid warnings
            try:
                # Some PyTorch versions may not accept `static_graph` kwarg; try the full call first
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                    static_graph=False
                )
            except TypeError:
                # Fallback for older PyTorch versions without the `static_graph` parameter
                try:
                    self.model = DDP(
                        self.model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=True
                    )
                except Exception as e:
                    print(f"[Rank {self.local_rank}] ❌ Failed to wrap model with DDP: {e}")
                    print(f"[Rank {self.local_rank}] Falling back to single-GPU/DataParallel mode")
                    # Leave model on device (already moved above)
            except Exception as e:
                print(f"[Rank {self.local_rank}] ❌ Failed to wrap model with DDP: {e}")
                print(f"[Rank {self.local_rank}] Falling back to single-GPU/DataParallel mode")
            
            # Note: Static graph disabled for dynamic parameter usage in multimodal model when supported
            print(f"[Rank {self.local_rank}] Model device setup complete")
        # Enable DataParallel for multi-GPU if available and not using distributed
        elif torch.cuda.device_count() > 1:
            print(f"[INFO] Using DataParallel on {torch.cuda.device_count()} GPUs.")
            self.model = torch.nn.DataParallel(self.model)

        # Initialize loss function with class weights for imbalanced data
        loss_type = getattr(self.config, 'loss_type', 'ce')  # Default to cross-entropy
        
        if loss_type == 'focal':
            # Use Focal Loss for better handling of class imbalance
            focal_alpha = getattr(self.config, 'focal_alpha', 1.0)
            focal_gamma = getattr(self.config, 'focal_gamma', 2.0)
            
            # CRITICAL FIX: Ensure class_weights are in float32 before passing to FocalLoss
            if self.class_weights is not None and self.config.use_weighted_loss:
                # Clone and ensure float32 dtype to prevent float16 corruption
                class_weights_for_loss = self.class_weights.clone().to(dtype=torch.float32)
                print(f"[CRITICAL FIX] Class weights for FocalLoss: {class_weights_for_loss} (dtype: {class_weights_for_loss.dtype})")
            else:
                class_weights_for_loss = None
            
            self.criterion = FocalLoss(
                alpha=focal_alpha, 
                gamma=focal_gamma, 
                class_weights=class_weights_for_loss,
                label_smoothing=getattr(self.config, 'label_smoothing', 0.0)
            )
            print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma}, label_smoothing={getattr(self.config, 'label_smoothing', 0.0)}) with weights: {class_weights_for_loss}")
        
        elif self.class_weights is not None and self.config.use_weighted_loss:
            # Use class-balanced Cross Entropy Loss
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            print(f"Using weighted CrossEntropyLoss with weights: {self.class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")

        # Initialize optimizer
        # Separate parameters: no weight decay for BatchNorm and bias parameters
        # Also use discriminative learning rates: backbone gets 10x lower LR
        # CRITICAL: Include ALL params (even frozen ones) so progressive unfreezing works.
        # PyTorch only updates params that have gradients — frozen params are safe to include.
        backbone_decay = []
        backbone_no_decay = []
        head_decay = []
        head_no_decay = []
        
        backbone_prefixes = ('visual_model.', 'audio_model.')
        
        for name, param in self.model.named_parameters():
            is_backbone = any(name.startswith(p) for p in backbone_prefixes)
            is_no_decay = ('bn' in name or 'norm' in name or 'bias' in name or 'BatchNorm' in name)
            
            if is_backbone:
                if is_no_decay:
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)
            else:
                if not param.requires_grad:
                    continue  # Skip frozen non-backbone params (shouldn't exist but be safe)
                if is_no_decay:
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)
        
        backbone_lr = self.config.learning_rate * 0.1  # 10x lower for pretrained backbone
        param_groups = [
            {'params': backbone_decay, 'weight_decay': self.config.weight_decay, 'lr': backbone_lr},
            {'params': backbone_no_decay, 'weight_decay': 0.0, 'lr': backbone_lr},
            {'params': head_decay, 'weight_decay': self.config.weight_decay},
            {'params': head_no_decay, 'weight_decay': 0.0},
        ]
        
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(param_groups, lr=self.config.learning_rate)
        elif self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(param_groups, lr=self.config.learning_rate)
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(param_groups, lr=self.config.learning_rate, momentum=0.9)
        else:
            self.optimizer = optim.AdamW(param_groups, lr=self.config.learning_rate)
        
        print(f"Optimizer: {self.config.optimizer} | backbone: {len(backbone_decay)+len(backbone_no_decay)} params (lr={backbone_lr:.1e}), head: {len(head_decay)+len(head_no_decay)} params (lr={self.config.learning_rate:.1e})")

        # Initialize learning rate scheduler
        if self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.scheduler_step_size, gamma=self.config.scheduler_gamma
            )
        elif self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler == 'cosine_with_restarts':
            # Cosine Annealing with Warm Restarts (SGDR)
            # T_0: Number of epochs for the first restart (cycle length)
            # T_mult: Factor to increase cycle length after each restart
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=10,  # First cycle: 10 epochs
                T_mult=2,  # Each subsequent cycle is 2x longer (10, 20, 40...)
                eta_min=1e-7  # Minimum learning rate
            )
        elif self.config.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=self.config.scheduler_gamma, patience=self.config.scheduler_patience, verbose=True
            )
        else:
            self.scheduler = None

        # Initialize learning rate warmup
        self.warmup_scheduler = None
        if self.config.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.config.warmup_epochs * len(self.train_loader)
            )

        print("Model, optimizer, and scheduler initialized")
        
        # Initialize EMA (Exponential Moving Average) for better generalization
        ema_decay = getattr(self.config, 'ema_decay', 0.0)
        if ema_decay > 0:
            self.ema = ModelEMA(self.model, decay=ema_decay, device=self.device)
            print(f"[EMA] Initialized with decay={ema_decay}")
        else:
            self.ema = None
        
        # AMP error counter for auto-disable
        self.amp_error_count = 0
        self.amp_error_threshold = getattr(self.config, 'debug_amp_error_threshold', 3)
        
        # Debug: Print GPU usage information
        self.print_gpu_usage_info()

    def disable_amp(self):
        """Disable AMP at runtime after repeated scaler errors."""
        if self.amp_enabled:
            print(f"[AMP] Disabling AMP after {self.amp_error_count} errors")
            self.amp_enabled = False
            self.scaler = None
            # Ensure GradScaler reference removed
            try:
                delattr(self, 'scaler')
            except Exception:
                pass
            # Reset any AMP-related flags in config to avoid re-enabling
            try:
                self.config.amp_enabled = False
            except Exception:
                pass
    
    def print_gpu_usage_info(self):
        """Print detailed information about GPU usage."""
        print("\n" + "="*60)
        print("🔥 GPU USAGE INFORMATION")
        print("="*60)
        
        if torch.cuda.is_available():
            total_gpus = torch.cuda.device_count()
            print(f"📊 Total GPUs Available: {total_gpus}")
            
            # Show which GPUs are visible
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
                print(f"🎯 CUDA_VISIBLE_DEVICES: {visible_devices}")
                gpu_list = visible_devices.split(',')
                print(f"📝 GPUs in use: {len(gpu_list)} GPUs (indices: {gpu_list})")
            else:
                print(f"📝 All {total_gpus} GPUs are visible (no CUDA_VISIBLE_DEVICES set)")
            
            # Show current device
            current_device = torch.cuda.current_device()
            print(f"🎮 Current CUDA Device: {current_device}")
            
            # Show GPU memory for each visible device
            print(f"\n💾 GPU Memory Status:")
            for i in range(total_gpus):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                    gpu_name = torch.cuda.get_device_properties(i).name
                    print(f"   GPU {i} ({gpu_name}): {memory_allocated:.2f}GB/{memory_total:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                except Exception as e:
                    print(f"   GPU {i}: Error getting memory info - {e}")
            
            # Show model parallel status
            print(f"\n🚀 Model Parallel Status:")
            if self.distributed:
                print(f"   ✅ Using DistributedDataParallel (DDP)")
                print(f"   📍 Local Rank: {self.local_rank}")
                print(f"   🌟 Main Process: {self.is_main_process}")
            elif hasattr(self.model, 'module'):
                # DataParallel wraps model in .module
                print(f"   ✅ Using DataParallel")
                if hasattr(self.model, 'device_ids'):
                    print(f"   📍 Device IDs: {self.model.device_ids}")
                else:
                    print(f"   📍 Using all available GPUs")
            else:
                print(f"   ❌ Using Single GPU only")
                print(f"   💡 Consider using --distributed for better multi-GPU performance")
            
        else:
            print("❌ CUDA not available - using CPU")
        
        print("="*60)
    
    def save_intermediate_checkpoint(self, epoch, batch_idx):
        """Save intermediate checkpoint during training."""
        # Only main process saves checkpoints in distributed training
        if not self.is_main_process:
            return
            
        if not self.config.save_intermediate:
            print(f"[DEBUG] Intermediate saving disabled, skipping batch {batch_idx}")
            return
            
        if batch_idx % self.config.save_intermediate_interval != 0:
            print(f"[DEBUG] Batch {batch_idx} not at save interval ({self.config.save_intermediate_interval}), skipping")
            return
        
        print(f"[CHECKPOINT] Saving intermediate checkpoint at epoch {epoch+1}, batch {batch_idx}")
        
        try:
            intermediate_dir = os.path.join(self.run_checkpoint_dir, "intermediate")
            os.makedirs(intermediate_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(
                intermediate_dir, 
                f"checkpoint_epoch_{epoch+1}_batch_{batch_idx}.pth"
            )
            
            # Handle distributed vs non-distributed model state dict
            if self.distributed:
                model_state_dict = self.model.module.state_dict()
            elif hasattr(self.model, 'module'):  # DataParallel
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
            
            checkpoint = {
                'epoch': epoch + 1,
                'batch': batch_idx,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'config': vars(self.config)  # Save config for resume compatibility
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Intermediate checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Error saving intermediate checkpoint: {e}")
            import traceback
            traceback.print_exc()
    
    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        
        # Apply model stabilization measures
        if hasattr(self.model, 'stabilize_model'):
            self.model.stabilize_model()
        
        # Gradually unfreeze visual layers
        if hasattr(self.model, 'unfreeze_visual_layers'):
            self.model.unfreeze_visual_layers(epoch)
        
        epoch_loss = 0
        valid_batch_count = 0  # Track valid batches for accurate loss averaging
        batch_losses = []  # Track individual batch losses for statistics
        y_true, y_pred, y_probs = [], [], []
        
        train_progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]", 
                           disable=not self.is_main_process)
        
        # Add debugging info for distributed training
        if self.is_main_process:
            print(f"[RANK {self.local_rank}] Starting training epoch {epoch+1} with {len(self.train_loader)} batches")
        
        # Main training loop
        train_progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]", disable=not self.is_main_process)
        for batch_idx, batch in enumerate(train_progress):
            # Runtime GPU memory check and batch size adjustment
            if batch_idx % 2 == 0:
                self.check_and_adjust_gpu_memory()
            # Ensure labels are integer class indices (0/1) and correct dtype
            try:
                if 'label' in batch:
                    lbl = batch['label']
                    # If one-hot or probabilities, convert to indices
                    if isinstance(lbl, torch.Tensor):
                        if lbl.dim() > 1 and lbl.shape[1] > 1:
                            # assume one-hot or logits -> take argmax
                            lbl = torch.argmax(lbl, dim=1)
                        # cast to long
                        lbl = lbl.to(dtype=torch.long)
                    else:
                        # try to convert list to tensor
                        try:
                            lbl = torch.tensor(lbl, dtype=torch.long)
                        except Exception:
                            pass
                    batch['label'] = lbl
            except Exception as e:
                print(f"[DATA] Warning: failed to normalize label dtype: {e}")
            # Runtime trace toggle: check a file to enable/disable model forward-hooks without restart
            try:
                desired = _read_trace_toggle_file()
                if desired is not None and hasattr(self, 'model'):
                    # If desired == False -> disable tracing
                    if desired is False and getattr(self.model, 'trace_tensors', False):
                        try:
                            if hasattr(self.model, 'remove_trace_hooks'):
                                self.model.remove_trace_hooks()
                            self.model.trace_tensors = False
                            if self.is_main_process:
                                print('[TRACE TOGGLE] Tracing disabled via trace_toggle.txt')
                        except Exception as e:
                            if self.is_main_process:
                                print(f'[TRACE TOGGLE] Failed to remove trace hooks: {e}')
                    # If desired == True -> enable tracing
                    if desired is True and not getattr(self.model, 'trace_tensors', False):
                        try:
                            if hasattr(self.model, 'register_trace_hooks'):
                                self.model.trace_tensors = True
                                self.model.register_trace_hooks()
                            if self.is_main_process:
                                print('[TRACE TOGGLE] Tracing enabled via trace_toggle.txt')
                        except Exception as e:
                            if self.is_main_process:
                                print(f'[TRACE TOGGLE] Failed to register trace hooks: {e}')
            except Exception:
                pass
            # Skip batches if resuming from a checkpoint
            if hasattr(self, 'start_batch') and epoch == getattr(self, 'start_epoch', 0) and batch_idx < self.start_batch:
                if self.is_main_process and batch_idx % 10 == 0:
                    print(f"[RESUME] Skipping batch {batch_idx} (resuming from batch {self.start_batch})")
                continue
            
            # Debug: Print batch loading progress
            if self.is_main_process and batch_idx == 0:
                print(f"[RANK {self.local_rank}] Processing first batch...")
            elif batch_idx % 10 == 0 and self.is_main_process:
                print(f"[RANK {self.local_rank}] Processing batch {batch_idx}/{len(self.train_loader)}")
                
            # Reset start_batch after first epoch to avoid skipping in subsequent epochs
            if epoch == getattr(self, 'start_epoch', 0) and batch_idx == getattr(self, 'start_batch', 0):
                if self.is_main_process:
                    print(f"🔄 Resumed training from epoch {epoch+1}, batch {batch_idx}")
                # Clear the start_batch to avoid affecting subsequent epochs
                if hasattr(self, 'start_batch'):
                    delattr(self, 'start_batch')
            try:
                # Add timeout protection for first batch
                if batch_idx == 0:
                    print(f"[RANK {self.local_rank}] Loading first batch, this may take time...")
                    start_time = time.time()
                
                # Periodic memory cleanup (once per epoch-fraction, not per-batch)
                if batch_idx % 100 == 0 and batch_idx > 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Start batch timer for throughput calculation
                batch_timer_start = time.time()

                # Move batch to device with timeout protection
                if batch_idx % 500 == 0:
                    print(f"[RANK {self.local_rank}] Moving batch {batch_idx} to device...")
                batch = move_batch_to_device(batch, self.device, trainer=self)
                # Detailed shape logging (guarded)
                log_batch_shapes(batch, prefix=f"after_move_to_device train batch {batch_idx}")
                # Detect empty or invalid tensors produced by preprocessing (e.g., failed MTCNN)
                try:
                    invalid_batch = False
                    for k, v in list(batch.items()):
                        if isinstance(v, torch.Tensor):
                            # Empty tensor or zero batch dimension
                            if v.numel() == 0 or (hasattr(v, 'shape') and len(v.shape) > 0 and v.shape[0] == 0):
                                print(f"[WARNING] Empty tensor detected for key '{k}' in batch {batch_idx}; skipping batch")
                                invalid_batch = True
                                break
                        elif isinstance(v, list):
                            if len(v) == 0:
                                print(f"[WARNING] Empty list detected for key '{k}' in batch {batch_idx}; skipping batch")
                                invalid_batch = True
                                break
                    if invalid_batch:
                        # Attempt to recover by clearing gradients and moving on
                        try:
                            self.optimizer.zero_grad()
                        except Exception:
                            pass
                        continue
                except Exception:
                    # Non-fatal: if this check fails, continue with processing
                    pass
                # GPU memory usage after moving batch to device (helpful when OS tools show N/A)
                try:
                    if torch.cuda.is_available() and self.is_main_process and batch_idx % 500 == 0:
                        allocated_gb, reserved_gb = get_gpu_memory_usage()
                        print(f"[GPU] Batch {batch_idx}: allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB")
                except Exception:
                    pass

                # Filter out placeholder samples marked with label == -1 (dataset fallbacks)
                try:
                    labels_tmp = batch.get('label', None)
                    if isinstance(labels_tmp, torch.Tensor):
                        labels_cpu = labels_tmp.detach()
                        if (labels_cpu == -1).any():
                            mask = (labels_cpu != -1)
                            keep_idx = mask.nonzero(as_tuple=False).squeeze(1)
                            num_removed = int((~mask).sum().item())
                            if keep_idx.numel() == 0:
                                print(f"[INFO] All samples in batch {batch_idx} are placeholders. Skipping batch.")
                                continue
                            # Rebuild tensors/lists in batch to keep only valid indices
                            for k, v in list(batch.items()):
                                if isinstance(v, torch.Tensor) and v.shape[0] == labels_cpu.shape[0]:
                                    batch[k] = v[keep_idx]
                                elif isinstance(v, list) and len(v) == labels_cpu.shape[0]:
                                    idxs = keep_idx.cpu().tolist()
                                    batch[k] = [v[i] for i in idxs]
                            print(f"[INFO] Filtered {num_removed} placeholder samples from batch {batch_idx} (kept {keep_idx.numel()})")
                except Exception as filter_e:
                    print(f"[WARNING] Could not filter placeholder samples at batch {batch_idx}: {filter_e}")
                
                if batch_idx == 0:
                    load_time = time.time() - start_time
                    print(f"[RANK {self.local_rank}] First batch loaded in {load_time:.2f} seconds")
                # --- SkinColorAnalyzer integration ---
                if getattr(self.config, 'enable_skin_color_analysis', False) and 'video_frames' in batch:
                    if not hasattr(self, '_skin_color_analyzer'):
                        self._skin_color_analyzer = SkinColorAnalyzer().to(self.device)
                    with torch.no_grad():
                        frames = batch['video_frames']
                        if frames.dim() == 5 and frames.shape[2] == 3:
                            frames_small = torch.nn.functional.interpolate(
                                frames.view(-1, 3, frames.shape[3], frames.shape[4]),
                                size=(56, 56), mode='bilinear', align_corners=False
                            ).view(frames.shape[0], frames.shape[1], 3, 56, 56)
                        else:
                            frames_small = frames
                        skin_features = self._skin_color_analyzer(frames_small)
                        batch['skin_color_features'] = skin_features
                
                # Get labels
                labels = batch['label']
                
                # ✅ DEPLOYMENT MODE: No paired labels, no original videos
                # Model processes single videos only, outputs single batch predictions
                original_labels = labels
                
                # 🔀 Mixup augmentation (mixes pairs of samples for regularization)
                mixup_alpha = getattr(self.config, 'mixup_alpha', 0.0)
                use_mixup = mixup_alpha > 0 and self.model.training
                if use_mixup:
                    batch, labels_a, labels_b, lam = mixup_batch(batch, labels, alpha=mixup_alpha)
                    labels = batch.get('label', labels)  # labels in batch may be integer, keep original
                else:
                    labels_a, labels_b, lam = labels, labels, 1.0
                
                # Debug: Print initial batch information (disabled to reduce spam)
                # if batch_idx < 5:  # Only for first few batches to avoid spam
                #     print(f"[DEBUG] Batch {batch_idx} - Batch tensor size: {original_labels.shape[0]}")
                #     print(f"[DEBUG] Batch {batch_idx} - Expected labels size: {labels.shape[0]}")
                #     for key, value in batch.items():
                #         if isinstance(value, torch.Tensor):
                #             print(f"[DEBUG] Batch {batch_idx} - {key} shape: {value.shape}")
                #         elif isinstance(value, list):
                #             print(f"[DEBUG] Batch {batch_idx} - {key} length: {len(value)}")
                
                # Validate batch consistency before forward pass (use original_labels for batch size)
                expected_batch_size = original_labels.shape[0]
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] != expected_batch_size:
                        print(f"[WARNING] Batch size mismatch in {key}: expected {expected_batch_size}, got {value.shape[0]}")
                        # Fix the mismatch by truncating or padding
                        if value.shape[0] > expected_batch_size:
                            batch[key] = value[:expected_batch_size]
                        elif value.shape[0] < expected_batch_size:
                            # Skip this batch if we can't fix it easily
                            print(f"[ERROR] Cannot fix batch size mismatch for {key}, skipping batch")
                            continue

                # Forward pass with mixed precision
                # Zero gradients at accumulation boundaries to support gradient accumulation
                if (batch_idx % self.grad_accum_steps) == 0:
                    self.optimizer.zero_grad()

                # Count successful vs failed samples for debugging
                success_count = 0
                total_samples = len(labels)

                def _check_and_fix_batch_size(outputs, labels):
                    """Check and fix batch size mismatches between outputs and labels."""
                    out_bs = outputs.shape[0] if hasattr(outputs, 'shape') and len(outputs.shape) > 0 else 1
                    lbl_bs = labels.shape[0] if hasattr(labels, 'shape') and len(labels.shape) > 0 else 1
                    
                    # For contrastive learning, outputs and labels should both be doubled (e.g., 8 vs 8)
                    # This is NOT a mismatch - it's expected behavior
                    if out_bs == lbl_bs:
                        return outputs, labels, True
                    
                    print(f"[ERROR] Output batch size {out_bs} does not match label batch size {lbl_bs} at batch {batch_idx}")
                    print(f"[DEBUG] outputs.shape: {getattr(outputs, 'shape', None)} labels.shape: {getattr(labels, 'shape', None)}")
                    
                    # Try to fix the mismatch
                    if out_bs > lbl_bs:
                        # Truncate outputs to match labels
                        print(f"[FIX] Truncating outputs from {out_bs} to {lbl_bs}")
                        outputs = outputs[:lbl_bs]
                        return outputs, labels, True
                    elif lbl_bs > out_bs:
                        # Truncate labels to match outputs 
                        print(f"[FIX] Truncating labels from {lbl_bs} to {out_bs}")
                        labels = labels[:out_bs]
                        return outputs, labels, True
                    return outputs, labels, True

                if self.amp_enabled:
                    with autocast():
                        # Debug: Check input shapes (first epoch, first batch only)
                        if batch_idx == 0 and epoch == 0:
                            print(f"[DEBUG] Before model forward - labels shape: {labels.shape}")
                            if 'video_frames' in batch:
                                print(f"[DEBUG] Before model forward - video_frames shape: {batch['video_frames'].shape}")
                            if 'audio' in batch:
                                print(f"[DEBUG] Before model forward - audio shape: {batch['audio'].shape}")
                        
                        # Timed forward pass (synchronize to get accurate GPU timing)
                        # Wrapped in try-except to catch CUDA errors gracefully
                        t_forward_start = time.time()
                        try:
                            outputs, results = self.model(batch)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        except RuntimeError as cuda_err:
                            # Handle CUDA errors (out of memory, device errors, etc.)
                            if 'CUDA' in str(cuda_err) or 'out of memory' in str(cuda_err).lower():
                                print(f"[CUDA ERROR] Batch {batch_idx}: {cuda_err}")
                                print("[CUDA ERROR] Attempting recovery...")
                                torch.cuda.empty_cache()
                                gc.collect()
                                self.optimizer.zero_grad()
                                continue  # Skip this batch
                            else:
                                raise  # Re-raise non-CUDA errors
                        t_forward_end = time.time()
                        if self.is_main_process and (batch_idx % 50 == 0):
                            print(f"[TIMING] Model forward batch {batch_idx}: {t_forward_end - t_forward_start:.3f}s")
                        
                        # Debug: Check output shapes (first batch only)
                        if batch_idx == 0 and epoch == 0:
                            print(f"[DEBUG] outputs shape: {outputs.shape}, labels shape: {labels.shape}")
                        
                        # Check for NaN/Inf in outputs
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            print(f"[ERROR] NaN or Inf detected in model outputs at batch {batch_idx}")
                            if globals().get('DEBUG_MODE', False):
                                print(f"[DEBUG] Output shape: {outputs.shape}")
                                print(f"[DEBUG] Output stats - min: {outputs.min()}, max: {outputs.max()}, mean: {outputs.mean()}")
                            self.optimizer.zero_grad()
                            self.nan_count += 1
                            continue
                        outputs = torch.clamp(outputs, min=-10, max=10)  # Prevent extreme confidence
                        if batch_idx < 5 and globals().get('DEBUG_MODE', False):
                            print(f"[DEBUG] After clamp - min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}")
                            print(f"[DEBUG] Labels: {labels[:8]}")
                        outputs, labels, batch_valid = _check_and_fix_batch_size(outputs, labels)
                        if not batch_valid:
                            print(f"[ERROR] Cannot fix batch size mismatch, skipping batch {batch_idx}")
                            self.optimizer.zero_grad()
                            continue
                        
                        # Debug: Log values before loss computation (disabled to reduce spam)
                        # if batch_idx < 3:
                        #     print(f"[DEBUG] Batch {batch_idx} - outputs range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                        #     print(f"[DEBUG] Batch {batch_idx} - labels: {labels[:4].tolist()}")
                        
                        # 🆕 HYBRID CONTRASTIVE LEARNING LOSS
                        # Main combined loss (with Mixup support)
                        if use_mixup:
                            loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                        else:
                            loss = self.criterion(outputs, labels)
                        
                        # Audio-only loss (learns difference patterns if training with pairs)
                        if 'audio_only_logits' in results and results['audio_only_logits'] is not None:
                            audio_only_logits = results['audio_only_logits']
                            if audio_only_logits.shape[0] == labels.shape[0]:
                                audio_only_loss = self.criterion(audio_only_logits, labels)
                                loss = loss + 0.1 * audio_only_loss  # Reduced from 0.3 to let main loss dominate
                                if batch_idx < 3:
                                    print(f"[MODALITY] Audio-only loss: {audio_only_loss.item():.4f}")
                        
                        # Video-only loss (learns difference patterns if training with pairs)
                        if 'video_only_logits' in results and results['video_only_logits'] is not None:
                            video_only_logits = results['video_only_logits']
                            if video_only_logits.shape[0] == labels.shape[0]:
                                video_only_loss = self.criterion(video_only_logits, labels)
                                loss = loss + 0.1 * video_only_loss  # Reduced from 0.3 to let main loss dominate
                                if batch_idx < 3:
                                    print(f"[MODALITY] Video-only loss: {video_only_loss.item():.4f}")
                        
                        # Contrastive consistency loss: Encourage agreement between modalities on same sample
                        # This helps prevent "detect everything as fake" by enforcing cross-modal consistency
                        if 'audio_only_logits' in results and 'video_only_logits' in results:
                            audio_only_logits = results['audio_only_logits']
                            video_only_logits = results['video_only_logits']
                            
                            # Only compute consistency loss if all modalities have the same batch size
                            if (audio_only_logits is not None and video_only_logits is not None and
                                audio_only_logits.shape[0] == outputs.shape[0] and 
                                video_only_logits.shape[0] == outputs.shape[0]):
                                
                                audio_probs = torch.softmax(audio_only_logits, dim=1)
                                video_probs = torch.softmax(video_only_logits, dim=1)
                                combined_probs = torch.softmax(outputs, dim=1)
                                
                                # KL divergence loss: Make audio/video predictions agree with combined
                                kl_audio = F.kl_div(audio_probs.log(), combined_probs, reduction='batchmean')
                                kl_video = F.kl_div(video_probs.log(), combined_probs, reduction='batchmean')
                                consistency_loss = 0.05 * (kl_audio + kl_video)  # Reduced from 0.1
                                
                                loss = loss + consistency_loss
                                
                                if batch_idx < 3:
                                    print(f"[CONTRASTIVE] Consistency loss: {consistency_loss.item():.4f}")
                        
                        # ====== AUXILIARY LOSSES FOR COMPONENT DIVERSITY ======
                        auxiliary_loss = torch.tensor(0.0, device=outputs.device)
                        diversity_penalty = torch.tensor(0.0, device=outputs.device)
                        
                        if results.get('auxiliary_outputs') is not None:
                            aux_loss, aux_details = self.model.compute_auxiliary_loss(
                                results['auxiliary_outputs'], 
                                labels
                            )
                            auxiliary_loss = aux_loss
                            
                            # Compute diversity penalty
                            diversity_penalty = self.model.compute_diversity_penalty(
                                results.get('component_contributions', {})
                            )
                            
                            # Add to total loss
                            loss = loss + auxiliary_loss + diversity_penalty
                            
                            if self.is_main_process and (batch_idx % 200 == 0 or getattr(self.config, 'debug_log_aux_losses', False)):
                                print(f"[AUX LOSS] Aux: {auxiliary_loss.item():.6f}, Diversity: {diversity_penalty.item():.6f}")
                        
                        # Log loss value periodically
                        if self.is_main_process and (batch_idx % 50 == 0):
                            print(f"[LOSS] Batch {batch_idx}: {loss.item():.6f}")
                        
                        # Check for NaN/Inf in loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"[ERROR] NaN or Inf loss detected at batch {batch_idx}")
                            print(f"[DEBUG] Loss value: {loss.item()}")
                            print(f"[DEBUG] Outputs stats - min: {outputs.min().item()}, max: {outputs.max().item()}, mean: {outputs.mean().item()}")
                            print(f"[DEBUG] Labels: {labels.tolist()}")
                            self.optimizer.zero_grad()
                            self.nan_count += 1
                            continue
                        # Add regularization for deepfake type if enabled
                        if self.config.detect_deepfake_type and 'deepfake_type' in results and results['deepfake_type'] is not None:
                            if 'deepfake_type' in batch and batch['deepfake_type'] is not None:
                                # Ensure deepfake_type target is a tensor
                                deepfake_type_target = batch['deepfake_type']
                                if isinstance(deepfake_type_target, list):
                                    deepfake_type_target = torch.tensor(deepfake_type_target, device=outputs.device, dtype=torch.long)
                                elif not isinstance(deepfake_type_target, torch.Tensor):
                                    deepfake_type_target = torch.tensor([deepfake_type_target], device=outputs.device, dtype=torch.long)
                                
                                # Ensure batch size consistency
                                if deepfake_type_target.shape[0] != results['deepfake_type'].shape[0]:
                                    min_batch = min(deepfake_type_target.shape[0], results['deepfake_type'].shape[0])
                                    deepfake_type_target = deepfake_type_target[:min_batch]
                                    deepfake_type_pred = results['deepfake_type'][:min_batch]
                                else:
                                    deepfake_type_pred = results['deepfake_type']
                                
                                try:
                                    deepfake_type_loss = nn.CrossEntropyLoss()(deepfake_type_pred, deepfake_type_target)
                                    if not torch.isnan(deepfake_type_loss):
                                        loss += self.config.deepfake_type_weight * deepfake_type_loss
                                except Exception as e:
                                    if self.debug:
                                        print(f"[WARNING] Error computing deepfake type loss: {e}")
                                    # Skip deepfake type loss if there's an error
                    
                    # Backward pass with scaler (AMP enabled)
                    try:
                        # For gradient accumulation, scale the loss down before backward
                        if self.grad_accum_steps > 1:
                            loss_for_backward = loss / float(self.grad_accum_steps)
                        else:
                            loss_for_backward = loss

                        self.scaler.scale(loss_for_backward).backward()
                        
                        # Check for NaN in gradients using model method
                        if hasattr(self.model, 'check_for_nan_gradients') and self.model.check_for_nan_gradients():
                            print(f"[ERROR] Skipping backward pass due to NaN gradients at batch {batch_idx}")
                            self.optimizer.zero_grad()
                            # Apply model stabilization
                            if hasattr(self.model, 'stabilize_model'):
                                self.model.stabilize_model()
                            # Reset scaler state
                            self.scaler.update()
                            # Skip this batch but continue training
                            continue
                        
                        # Apply model's gradient clipping with AMP
                        unscale_called = False
                        
                        # Step only at accumulation boundary
                        step_now = ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx == len(self.train_loader) - 1)
                        
                        if step_now:
                            # Unscale gradients once before clipping
                            self.scaler.unscale_(self.optimizer)
                            unscale_called = True
                            
                            # Apply gradient clipping
                            if hasattr(self.model, 'clip_gradients'):
                                self.model.clip_gradients(max_norm=self.config.gradient_clip)
                            elif self.config.gradient_clip > 0:
                                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            
                            # EMA update after optimizer step
                            if self.ema is not None:
                                self.ema.update(self.model)
                            
                            # Step warmup scheduler during warmup phase
                            if self.warmup_scheduler is not None and epoch < self.config.warmup_epochs:
                                self.warmup_scheduler.step()

                            # Update metrics using unscaled/original loss value
                            try:
                                raw_loss_value = float((loss * float(self.grad_accum_steps)).detach().cpu().item()) if self.grad_accum_steps > 1 else float(loss.detach().cpu().item())
                            except Exception:
                                raw_loss_value = float(loss.detach().cpu().item())
                            epoch_loss += raw_loss_value
                            valid_batch_count += 1
                            batch_losses.append(raw_loss_value)
                            
                            # Collect predictions for metrics (moved outside else branch)
                            probs = torch.softmax(outputs, dim=1)
                            _, preds = torch.max(outputs, 1)
                            y_true.extend(labels.cpu().numpy())
                            y_pred.extend(preds.cpu().numpy())
                            try:
                                y_probs.extend(probs[:, 1].detach().cpu().numpy())
                            except Exception:
                                y_probs.extend(probs.max(dim=1)[0].detach().cpu().numpy())
                        else:
                            # For gradient accumulation batches, calculate loss for logging
                            try:
                                raw_loss_value = float(loss.detach().cpu().item())
                            except Exception:
                                raw_loss_value = 0.0
                            probs = torch.softmax(outputs, dim=1)
                            _, preds = torch.max(outputs, 1)
                            y_true.extend(labels.cpu().numpy())
                            y_pred.extend(preds.cpu().numpy())
                            try:
                                y_probs.extend(probs[:, 1].detach().cpu().numpy())
                            except Exception:
                                y_probs.extend(probs.max(dim=1)[0].detach().cpu().numpy())
                        try:
                            if torch.cuda.is_available() and self.is_main_process and (batch_idx % 100 == 0):
                                allocated_gb, reserved_gb = get_gpu_memory_usage()
                                print(f"[GPU] batch {batch_idx}: allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB")
                        except Exception:
                            pass

                        # --- Per-batch metrics: throughput, memory, loss, grad-norm, prediction distribution ---
                        try:
                            batch_time = time.time() - batch_timer_start if 'batch_timer_start' in locals() else None
                            samples = int(original_labels.shape[0]) if 'original_labels' in locals() else (int(labels.shape[0]) if 'labels' in locals() else 0)
                            throughput = (samples / batch_time) if batch_time and batch_time > 0 else None

                            # Ensure we have memory numbers
                            try:
                                alloc_gb, reserved_gb = (allocated_gb, reserved_gb) if 'allocated_gb' in locals() else get_gpu_memory_usage()
                            except Exception:
                                alloc_gb, reserved_gb = get_gpu_memory_usage()

                            # Gradient norm (L2) across parameters (best-effort)
                            grad_norm = None
                            try:
                                total_norm_sq = 0.0
                                for p in self.model.parameters():
                                    if p.grad is not None:
                                        param_norm = p.grad.data.norm(2)
                                        total_norm_sq += float(param_norm.item()) ** 2
                                grad_norm = float(total_norm_sq ** 0.5)
                            except Exception:
                                grad_norm = None

                            # Prediction distribution (probs for 'fake' class if available)
                            pred_mean = pred_std = pred_pct_fake = None
                            try:
                                # Compute probs if not already done
                                if 'probs' not in locals() or probs is None:
                                    probs = torch.softmax(outputs, dim=1)
                                pred_probs = probs[:, 1].detach().cpu()
                                pred_mean = float(pred_probs.mean().item())
                                pred_std = float(pred_probs.std().item())
                                pred_pct_fake = float((pred_probs >= 0.5).float().mean().item())
                            except Exception:
                                pass

                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            time_str = f"{batch_time:.3f}s" if batch_time is not None else "N/A"
                            thr_str = f"{throughput:.1f} samp/s" if throughput is not None else "N/A"
                            grad_str = f"{grad_norm:.4f}" if grad_norm is not None else "N/A"
                            pred_mean_str = f"{pred_mean:.4f}" if pred_mean is not None else "N/A"
                            pred_std_str = f"{pred_std:.4f}" if pred_std is not None else "N/A"
                            pred_pct_str = f"{pred_pct_fake:.3f}" if pred_pct_fake is not None else "N/A"

                            log_line = (
                                f"[METRICS] {timestamp} E{epoch+1} B{batch_idx} "
                                f"time={time_str} thr={thr_str} "
                                f"alloc={alloc_gb:.2f}GB res={reserved_gb:.2f}GB "
                                f"loss={raw_loss_value:.6f} grad_norm={grad_str} "
                                f"pred_mean={pred_mean_str} pred_std={pred_std_str} pct_fake={pred_pct_str}\n"
                            )
                            # Print and persist to log file if available
                            if self.is_main_process:
                                try:
                                    print(log_line.strip())
                                except Exception:
                                    pass
                            if self.log_file:
                                try:
                                    self.log_file.write(log_line)
                                    self.log_file.flush()
                                except Exception:
                                    pass
                        except Exception as metrics_err:
                            print(f"[METRICS] Error computing metrics for batch {batch_idx}: {metrics_err}")
                        
                    except Exception as scaler_error:
                        print(f"[ERROR] AMP scaler error at batch {batch_idx}: {scaler_error}")
                        # Increment AMP error counter and optionally disable AMP if repeated
                        try:
                            self.amp_error_count = getattr(self, 'amp_error_count', 0) + 1
                        except Exception:
                            self.amp_error_count = 1
                        print(f"[AMP] scaler errors so far: {self.amp_error_count}")
                        # Reset optimizer state for safety
                        try:
                            self.optimizer.zero_grad()
                        except Exception:
                            pass
                        # If errors exceed threshold, disable AMP entirely and continue
                        try:
                            if hasattr(self, 'amp_error_threshold') and self.amp_error_count >= self.amp_error_threshold:
                                print(f"[AMP] error threshold reached ({self.amp_error_count} >= {self.amp_error_threshold}), disabling AMP")
                                try:
                                    self.disable_amp()
                                except Exception:
                                    pass
                                # Do not attempt to recreate scaler if AMP disabled
                                continue
                        except Exception:
                            pass
                        # Otherwise attempt to recreate scaler with fresh state
                        try:
                            if getattr(self, 'amp_enabled', True):
                                self.scaler = GradScaler(enabled=self.amp_enabled, init_scale=2**8)
                                print(f"[INFO] Recreated AMP scaler with fresh state")
                        except Exception:
                            pass
                        continue
                else:
                    # Timed forward pass for non-AMP path
                    t_forward_start = time.time()
                    outputs, results = self.model(batch)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except Exception:
                        pass
                    t_forward_end = time.time()
                    if self.is_main_process and (batch_idx % 50 == 0):
                        print(f"[TIMING] Model forward (no AMP) batch {batch_idx}: {t_forward_end - t_forward_start:.3f}s")
                    # Check for NaN/Inf in outputs
                    # Check for NaN/Inf in outputs
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"[ERROR] NaN or Inf detected in model outputs at batch {batch_idx}")
                        if globals().get('DEBUG_MODE', False):
                            print(f"[DEBUG] Output shape: {outputs.shape}")
                            print(f"[DEBUG] Output stats - min: {outputs.min()}, max: {outputs.max()}, mean: {outputs.mean()}")
                        self.optimizer.zero_grad()
                        self.nan_count += 1
                        continue
                    outputs = torch.clamp(outputs, min=-10, max=10)  # Prevent extreme confidence
                    outputs, labels, batch_valid = _check_and_fix_batch_size(outputs, labels)
                    if not batch_valid:
                        print(f"[ERROR] Cannot fix batch size mismatch, skipping batch {batch_idx}")
                        self.optimizer.zero_grad()
                        continue
                    
                    # Log memory periodically
                    try:
                        if torch.cuda.is_available() and (batch_idx % 100 == 0):
                            torch.cuda.synchronize()
                            alloc_gb, reserved_gb = get_gpu_memory_usage()
                            if batch_idx == 0:
                                print(f"[MEMORY] Batch {batch_idx}: Allocated {alloc_gb:.2f}GB, Reserved {reserved_gb:.2f}GB")
                    except Exception:
                        pass
                    
                    # Main combined loss (with Mixup support)
                    if use_mixup:
                        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                    else:
                        loss = self.criterion(outputs, labels)
                    
                    # Per-modality losses (Non-AMP path — same as AMP path)
                    if 'audio_only_logits' in results and results['audio_only_logits'] is not None:
                        audio_only_logits = results['audio_only_logits']
                        if audio_only_logits.shape[0] == labels.shape[0]:
                            loss = loss + 0.1 * self.criterion(audio_only_logits, labels)
                    if 'video_only_logits' in results and results['video_only_logits'] is not None:
                        video_only_logits = results['video_only_logits']
                        if video_only_logits.shape[0] == labels.shape[0]:
                            loss = loss + 0.1 * self.criterion(video_only_logits, labels)
                    
                    # ====== AUXILIARY LOSSES FOR COMPONENT DIVERSITY (Non-AMP path) ======
                    auxiliary_loss = torch.tensor(0.0, device=outputs.device)
                    diversity_penalty = torch.tensor(0.0, device=outputs.device)
                    
                    if results.get('auxiliary_outputs') is not None:
                        aux_loss, aux_details = self.model.compute_auxiliary_loss(
                            results['auxiliary_outputs'], 
                            labels
                        )
                        auxiliary_loss = aux_loss
                        
                        # Compute diversity penalty
                        diversity_penalty = self.model.compute_diversity_penalty(
                            results.get('component_contributions', {})
                        )
                        
                        # Add to total loss
                        loss = loss + auxiliary_loss + diversity_penalty
                        
                        if self.is_main_process and (batch_idx % 200 == 0 or getattr(self.config, 'debug_log_aux_losses', False)):
                            print(f"[AUX LOSS] Aux: {auxiliary_loss.item():.6f}, Diversity: {diversity_penalty.item():.6f}")
                    
                    # Log loss periodically
                    if self.is_main_process and (batch_idx % 200 == 0):
                        print(f"[LOSS] Batch {batch_idx}: {loss.item():.6f}")
                    
                    # Check for NaN/Inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[ERROR] NaN or Inf loss detected at batch {batch_idx}")
                        print(f"[DEBUG] Loss value: {loss.item()}")
                        print(f"[DEBUG] Outputs stats - min: {outputs.min().item()}, max: {outputs.max().item()}, mean: {outputs.mean().item()}")
                        print(f"[DEBUG] Labels: {labels.tolist()}")
                        self.optimizer.zero_grad()
                        self.nan_count += 1
                        continue
                    
                    # Support gradient accumulation: average loss across accumulation steps
                    if self.grad_accum_steps > 1:
                        loss_for_backward = loss / float(self.grad_accum_steps)
                    else:
                        loss_for_backward = loss

                    # Backward pass (no AMP)
                    try:
                        if getattr(self.config, 'debug_autograd_detect', False):
                            with torch.autograd.detect_anomaly():
                                loss_for_backward.backward()
                        else:
                            loss_for_backward.backward()
                    except Exception as back_err:
                        print(f"[ERROR] Exception during backward (non-AMP) at batch {batch_idx}: {back_err}")
                        import traceback
                        traceback.print_exc()
                        # Save snapshot for post-mortem if enabled
                        if getattr(self, 'debug_save_on_anomaly', False):
                            try:
                                self._save_debug_snapshot('anomaly', batch_idx, batch if 'batch' in locals() else None, outputs if 'outputs' in locals() else None, loss_for_backward)
                            except Exception as e:
                                print(f"[DEBUG] Failed to save snapshot after backward error: {e}")
                        self.optimizer.zero_grad()
                        self.nan_count += 1
                        continue

                    # Step only at accumulation boundary
                    step_now = ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx == len(self.train_loader) - 1)
                    if step_now:
                        # Gradient clipping
                        if self.config.gradient_clip > 0:
                            # Respect debug flag to enforce stricter clipping
                            max_norm = float(self.config.gradient_clip)
                            if getattr(self.config, 'debug_strict_clip', False):
                                max_norm = min(max_norm, 1.0)

                            # Log per-parameter grad stats if requested
                            if getattr(self.config, 'debug_log_grad_stats', False):
                                try:
                                    max_g = 0.0
                                    nan_found = False
                                    for name, p in self.model.named_parameters():
                                        if p.grad is None:
                                            continue
                                        grad_abs_max = float(p.grad.detach().abs().max().cpu().item())
                                        if grad_abs_max > max_g:
                                            max_g = grad_abs_max
                                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                            print(f"[DEBUG] NaN/Inf in grad for {name}")
                                            nan_found = True
                                    print(f"[DEBUG] Pre-clip max grad abs={max_g:.6e}, nan_found={nan_found}")
                                except Exception:
                                    pass

                            if hasattr(self.model, 'clip_gradients'):
                                self.model.clip_gradients(max_norm=max_norm)
                            else:
                                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                            if getattr(self.config, 'debug_log_grad_stats', False):
                                try:
                                    max_g_post = 0.0
                                    for name, p in self.model.named_parameters():
                                        if p.grad is None:
                                            continue
                                        grad_abs_max = float(p.grad.detach().abs().max().cpu().item())
                                        if grad_abs_max > max_g_post:
                                            max_g_post = grad_abs_max
                                    print(f"[DEBUG] Post-clip max grad abs={max_g_post:.6e}")
                                except Exception:
                                    pass

                        # Optimizer step and zero grads
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        # EMA update after optimizer step
                        if self.ema is not None:
                            self.ema.update(self.model)
                        
                        # Step warmup scheduler during warmup phase
                        if self.warmup_scheduler is not None and epoch < self.config.warmup_epochs:
                            self.warmup_scheduler.step()

                        # Update metrics using original (unscaled) loss value
                        try:
                            raw_loss_value = float((loss * float(self.grad_accum_steps)).detach().cpu().item()) if self.grad_accum_steps > 1 else float(loss.detach().cpu().item())
                        except Exception:
                            raw_loss_value = float(loss.detach().cpu().item())
                        epoch_loss += raw_loss_value
                        valid_batch_count += 1
                        batch_losses.append(raw_loss_value)
                        
                        # Collect predictions for metrics (moved outside else branch)
                        probs = torch.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)
                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(preds.cpu().numpy())
                        try:
                            y_probs.extend(probs[:, 1].detach().cpu().numpy())
                        except Exception:
                            y_probs.extend(probs.max(dim=1)[0].detach().cpu().numpy())
                    else:
                        # For gradient accumulation batches, calculate loss for logging
                        try:
                            raw_loss_value = float(loss.detach().cpu().item())
                        except Exception:
                            raw_loss_value = 0.0
                        batch_losses.append(raw_loss_value)

                        # Store predictions
                        probs = torch.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)
                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(preds.cpu().numpy())
                        try:
                            y_probs.extend(probs[:, 1].detach().cpu().numpy())
                        except Exception:
                            y_probs.extend(probs.max(dim=1)[0].detach().cpu().numpy())
                        # --- Per-batch metrics (non-AMP path) ---
                        try:
                            batch_time = time.time() - batch_timer_start if 'batch_timer_start' in locals() else None
                            samples = int(original_labels.shape[0]) if 'original_labels' in locals() else (int(labels.shape[0]) if 'labels' in locals() else 0)
                            throughput = (samples / batch_time) if batch_time and batch_time > 0 else None
                            try:
                                alloc_gb, reserved_gb = get_gpu_memory_usage()
                            except Exception:
                                alloc_gb, reserved_gb = 0.0, 0.0

                            # Gradient norm (L2) across parameters
                            grad_norm = None
                            try:
                                total_norm_sq = 0.0
                                for p in self.model.parameters():
                                    if p.grad is not None:
                                        param_norm = p.grad.data.norm(2)
                                        total_norm_sq += float(param_norm.item()) ** 2
                                grad_norm = float(total_norm_sq ** 0.5)
                            except Exception:
                                grad_norm = None

                            pred_mean = pred_std = pred_pct_fake = None
                            try:
                                # Compute probs if not already done
                                if 'probs' not in locals() or probs is None:
                                    probs = torch.softmax(outputs, dim=1)
                                pred_probs = probs[:, 1].detach().cpu()
                                pred_mean = float(pred_probs.mean().item())
                                pred_std = float(pred_probs.std().item())
                                pred_pct_fake = float((pred_probs >= 0.5).float().mean().item())
                            except Exception:
                                pass

                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            time_str = f"{batch_time:.3f}s" if batch_time is not None else "N/A"
                            thr_str = f"{throughput:.1f} samp/s" if throughput is not None else "N/A"
                            grad_str = f"{grad_norm:.4f}" if grad_norm is not None else "N/A"
                            pred_mean_str = f"{pred_mean:.4f}" if pred_mean is not None else "N/A"
                            pred_std_str = f"{pred_std:.4f}" if pred_std is not None else "N/A"
                            pred_pct_str = f"{pred_pct_fake:.3f}" if pred_pct_fake is not None else "N/A"

                            log_line = (
                                f"[METRICS] {timestamp} E{epoch+1} B{batch_idx} "
                                f"time={time_str} thr={thr_str} "
                                f"alloc={alloc_gb:.2f}GB res={reserved_gb:.2f}GB "
                                f"loss={raw_loss_value:.6f} grad_norm={grad_str} "
                                f"pred_mean={pred_mean_str} pred_std={pred_std_str} pct_fake={pred_pct_str}\n"
                            )
                            if self.is_main_process:
                                try:
                                    print(log_line.strip())
                                except Exception:
                                    pass
                            if self.log_file:
                                try:
                                    self.log_file.write(log_line)
                                    self.log_file.flush()
                                except Exception:
                                    pass
                        except Exception as metrics_err:
                            print(f"[METRICS] Error computing metrics for batch {batch_idx}: {metrics_err}")
                    
                    # Add regularization for deepfake type if enabled
                    if self.config.detect_deepfake_type and 'deepfake_type' in results and results['deepfake_type'] is not None:
                        if 'deepfake_type' in batch and batch['deepfake_type'] is not None:
                            # Ensure deepfake_type target is a tensor
                            deepfake_type_target = batch['deepfake_type']
                            if isinstance(deepfake_type_target, list):
                                deepfake_type_target = torch.tensor(deepfake_type_target, device=outputs.device, dtype=torch.long)
                            if self.distributed:
                                # Do not re-run full distributed initialization inside the training loop.
                                # If the process group is missing, warn and continue rather than attempting
                                # heavyweight re-initialization here.
                                if not dist.is_initialized():
                                    print(f"[Rank {self.local_rank}] ⚠️ distributed=True but process group not initialized. Skipping re-init inside loop.")
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Clear GPU cache on any error to prevent memory leaks
                clear_gpu_cache()
                self.optimizer.zero_grad()
                continue
# Calculate average loss using valid batch count
        if valid_batch_count > 0:
            avg_loss = epoch_loss / valid_batch_count
            # Log batch loss statistics for debugging
            if batch_losses:
                print(f"\n[Loss Statistics] Min: {min(batch_losses):.6e}, Max: {max(batch_losses):.6e}, Mean: {np.mean(batch_losses):.6e}, Std: {np.std(batch_losses):.6e}")
                print(f"[Loss Statistics] Valid batches: {valid_batch_count}/{len(self.train_loader)}")
        else:
            avg_loss = 0.0
            print("[ERROR] No valid batches! All losses were NaN or skipped!")
        
        precision, recall, f1, auc_score, accuracy, macro_f1 = calculate_metrics(y_true, y_pred, y_probs, epoch+1)
        
        # Check for degenerate solution (model only predicting one class)
        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1:
            print(f"\n⚠️  WARNING: DEGENERATE SOLUTION DETECTED!")
            print(f"⚠️  Model is only predicting class {unique_preds[0]} (always {'Fake' if unique_preds[0] == 1 else 'Real'})")
            print(f"⚠️  This indicates severe class imbalance issues. Consider:")
            print(f"   1. Increasing class weights for minority class")
            print(f"   2. Using Focal Loss with higher gamma (--focal_gamma 3.0)")
            print(f"   3. Oversampling minority class (--oversample_minority)")
            print(f"   4. Reducing learning rate to prevent overfitting to majority class\n")
        
        # Store metrics (including macro F1 for balanced evaluation)
        self.metrics['train_losses'].append(avg_loss)
        self.metrics['train_accuracies'].append(accuracy)
        self.metrics['train_f1_scores'].append(f1)
        self.metrics['train_macro_f1_scores'].append(macro_f1)  # Key metric for imbalanced data
        self.metrics['train_auc_scores'].append(auc_score)
        
        # Plot confusion matrix for training
        if self.is_main_process:
            cm_path = plot_confusion_matrix(y_true, y_pred, epoch+1, self.plot_dir, split='train')
            
            # Log confusion matrix to WandB
            if self.config.use_wandb:
                wandb.log({f"train_confusion_matrix_epoch_{epoch+1}": wandb.Image(cm_path)})
        
        # Return metrics (include macro F1)
        return avg_loss, accuracy, precision, recall, f1, auc_score, macro_f1
    
    def validate_epoch(self, epoch):
        """Validate the model for one epoch."""
        # Always use eval mode during validation
        self.model.eval()
        
        # Apply EMA shadow parameters for validation (better generalization)
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        
        epoch_loss = 0
        y_true, y_pred, y_probs = [], [], []
        y_pred_audio, y_pred_video = [], []  # 🆕 Per-modality predictions
        
        val_progress = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Val]", 
                          disable=not self.is_main_process)
        
        # Track component contributions to analyze feature importance
        all_component_contributions = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_progress):
                try:
                    # Clear GPU cache periodically during validation
                    if batch_idx % 20 == 0:
                        clear_gpu_cache()
                    
                    # Move batch to device
                    batch = move_batch_to_device(batch, self.device, trainer=self)
                    log_batch_shapes(batch, prefix=f"after_move_to_device val batch {batch_idx}")

                    # Filter out placeholder samples marked with label == -1 in validation
                    try:
                        labels_tmp = batch.get('label', None)
                        if isinstance(labels_tmp, torch.Tensor):
                            labels_cpu = labels_tmp.detach()
                            if (labels_cpu == -1).any():
                                mask = (labels_cpu != -1)
                                keep_idx = mask.nonzero(as_tuple=False).squeeze(1)
                                num_removed = int((~mask).sum().item())
                                if keep_idx.numel() == 0:
                                    print(f"[INFO] Validation batch {batch_idx} contains only placeholders. Skipping batch.")
                                    continue
                                for k, v in list(batch.items()):
                                    if isinstance(v, torch.Tensor) and v.shape[0] == labels_cpu.shape[0]:
                                        batch[k] = v[keep_idx]
                                    elif isinstance(v, list) and len(v) == labels_cpu.shape[0]:
                                        idxs = keep_idx.cpu().tolist()
                                        batch[k] = [v[i] for i in idxs]
                                print(f"[INFO] Filtered {num_removed} placeholder samples from validation batch {batch_idx} (kept {keep_idx.numel()})")
                    except Exception as filter_e:
                        print(f"[WARNING] Could not filter placeholder samples in validation at batch {batch_idx}: {filter_e}")

                    # DEBUG: Inspect validation inputs (first batch, first epoch only)
                    if batch_idx == 0 and epoch == 0:
                        for key in ['video_frames', 'audio']:
                            if key in batch and isinstance(batch[key], torch.Tensor):
                                try:
                                    tensor = batch[key]
                                    bs = tensor.shape[0]
                                    flat = tensor.view(bs, -1).float()
                                    per_sample_means = flat.mean(dim=1)
                                    print(f"[VAL DEBUG] {key} per-sample mean std: {per_sample_means.std().item():.6f}")
                                except Exception:
                                    pass

                    # Get labels
                    labels = batch['label']
                    original_labels = labels
                    
                    # ✅ DEPLOYMENT MODE: No paired labels, single predictions only
                    
                    # DEBUG: Check eval mode on first batch
                    if batch_idx == 0 and epoch == 0:
                        print(f"[VAL] Model training mode: {self.model.training}")
                    
                    # Forward pass
                    if self.amp_enabled:
                        with autocast():
                            outputs, results = self.model(batch)
                            loss = self.criterion(outputs, labels)
                    else:
                        outputs, results = self.model(batch)
                        # Log shapes after forward
                        log_batch_shapes(batch, prefix=f"post_forward_input train batch {batch_idx}")
                        if isinstance(outputs, torch.Tensor):
                            if globals().get('DEBUG_MODE', False):
                                print(f"[SHAPES] outputs: shape={tuple(outputs.shape)}, dtype={outputs.dtype}, device={outputs.device}")
                        if isinstance(results, dict):
                            log_batch_shapes(results, prefix=f"post_forward_results train batch {batch_idx}")
                        loss = self.criterion(outputs, labels)
                    
                    # 🆕 COLLECT PER-MODALITY PREDICTIONS FOR METRICS
                    if 'audio_only_logits' in results and results['audio_only_logits'] is not None:
                        audio_only_probs = torch.softmax(results['audio_only_logits'], dim=1)
                        audio_only_preds = (audio_only_probs[:, 1] >= 0.50).long().cpu().numpy()
                        y_pred_audio.extend(audio_only_preds)
                    
                    if 'video_only_logits' in results and results['video_only_logits'] is not None:
                        video_only_probs = torch.softmax(results['video_only_logits'], dim=1)
                        video_only_preds = (video_only_probs[:, 1] >= 0.50).long().cpu().numpy()
                        y_pred_video.extend(video_only_preds)
                    
                    # ADAPTIVE BIAS REMOVED: Was causing the model to learn bias instead of features
                    # The bias was 4.4x stronger than feature contributions, making the model
                    # ignore video/audio data and just predict based on bias term.
                    # Now relying on proper class_weights in the loss function instead.
                    
                    # Update progress bar
                    val_progress.set_postfix(loss=f"{loss.item():.4f}")
                    
                    # Accumulate loss
                    epoch_loss += loss.item()
                    
                    # Accumulate predictions and labels for metrics calculation
                    y_true.extend(labels.cpu().numpy())
                    
                    # Calculate softmax probabilities
                    probs = torch.softmax(outputs, dim=1)
                    
                    # Debug: Print first batch validation predictions (epoch 0 only)
                    if batch_idx == 0 and epoch == 0:
                        threshold_preds = (probs[:, 1] >= 0.50).long()
                        print(f"\n[VALIDATION] Epoch {epoch+1} first batch:")
                        print(f"  Prob[Real] mean: {probs[:, 0].mean().item():.4f}, Prob[Fake] mean: {probs[:, 1].mean().item():.4f}\n")
                    
                    # Use 50% threshold - standard binary classification decision boundary
                    # Model will learn the right balance through proper class_weights in loss function
                    threshold = 0.50
                    predictions = (probs[:, 1] >= threshold).long().cpu().numpy()
                    y_pred.extend(predictions)
                    y_probs.extend(probs[:, 1].detach().cpu().numpy())
                    
                    # Visualize sample predictions periodically
                    if (batch_idx + 1) % self.config.visualization_interval == 0 and self.is_main_process:
                        try:
                            sample_idx = np.random.randint(min(4, outputs.size(0)))
                            save_visualizations(
                                batch, outputs, results,
                                epoch + 1, sample_idx,
                                os.path.join(self.viz_dir, f"val_epoch_{epoch+1}"),
                                model=self.model
                            )
                        except Exception as vis_error:
                            print(f"Error visualizing predictions: {vis_error}")
                    
                    # Track component contributions for feature importance analysis
                    if 'component_contributions' in results and results['component_contributions'] is not None:
                        for key, value in results['component_contributions'].items():
                            if key not in all_component_contributions:
                                all_component_contributions[key] = []
                            
                            # Extract scalar value if it's a tensor
                            if isinstance(value, torch.Tensor):
                                if value.numel() == 1:
                                    all_component_contributions[key].append(value.item())
                                else:
                                    # Average for multi-element tensors
                                    all_component_contributions[key].append(value.mean().item())
                            elif isinstance(value, (int, float)):
                                all_component_contributions[key].append(value)
                    
                except torch.cuda.OutOfMemoryError as oom_error:
                    print(f"[CUDA OOM] Out of memory error in validation batch {batch_idx}: {oom_error}")
                    print(f"[CUDA OOM] Clearing cache and continuing...")
                    
                    # Track OOM errors
                    self.oom_errors += 1
                    
                    clear_gpu_cache()
                    continue
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    clear_gpu_cache()
                    continue
        
        # Calculate average loss and metrics
        avg_loss = epoch_loss / len(self.val_loader)
        
        # SAFETY CHECK: Detect degenerate predictions (all one class)
        y_pred_array = np.array(y_pred)
        y_probs_array = np.array(y_probs)
        unique_preds = np.unique(y_pred_array)
        
        if len(unique_preds) == 1:
            degenerate_class = unique_preds[0]
            print(f"\n{'='*80}")
            print(f"[DEGENERATE SOLUTION DETECTED] All predictions are class {degenerate_class}")
            print(f"[DEGENERATE SOLUTION] Prob[Fake] mean: {y_probs_array.mean():.4f}, std: {y_probs_array.std():.4f}")
            # Removed emergency threshold hack: do not override model predictions.
            # Instead, log and increment counter so we can track persistent collapse.
            try:
                self.degenerate_count = getattr(self, 'degenerate_count', 0) + 1
            except Exception:
                self.degenerate_count = 1
            print(f"[DEGENERATE] Occurrences so far: {self.degenerate_count}")
            print("[ACTION] Not applying emergency-threshold; please inspect data/features or adjust loss/weights.")

            # AUTO-MITIGATION: If collapse persists, switch to stronger measures automatically
            try:
                mitigation_threshold = getattr(self.config, 'degenerate_mitigation_threshold', 1)
                allow_auto = getattr(self.config, 'enable_degenerate_auto_mitigation', True)
            except Exception:
                mitigation_threshold = 1
                allow_auto = True

            if allow_auto and getattr(self, 'degenerate_count', 0) >= mitigation_threshold and not getattr(self, 'degenerate_mitigation_applied', False):
                print("[AUTOMIT] Persistent degenerate validation detected — applying automated mitigation.")
                try:
                    # Prepare class weights for loss (ensure float32 on correct device)
                    if getattr(self, 'class_weights', None) is None:
                        cw = torch.tensor([2.0, 1.0], device=self.device, dtype=torch.float32)
                    else:
                        cw = self.class_weights.clone().to(self.device, dtype=torch.float32)
                        # Bias up the minority class weight conservatively (cap to avoid instability)
                        try:
                            # Multiply weights by 2.0 but clip to a reasonable max
                            cw = torch.clamp(cw * 2.0, max=10.0)
                        except Exception:
                            pass

                    new_gamma = getattr(self.config, 'auto_mitigation_focal_gamma', 3.0)
                    print(f"[AUTOMIT] Switching loss -> FocalLoss(gamma={new_gamma}) with class_weights={cw}")
                    self.criterion = FocalLoss(alpha=1.0, gamma=new_gamma, class_weights=cw)

                    # Reduce learning rate conservatively (10x smaller but not below min_lr)
                    min_lr = getattr(self.config, 'min_lr', 1e-5)
                    # Safely update optimizer learning rates if optimizer exists
                    sample_lr = None
                    if hasattr(self, 'optimizer') and getattr(self, 'optimizer', None) is not None:
                        try:
                            for pg in getattr(self.optimizer, 'param_groups', []):
                                old_lr = pg.get('lr', getattr(self.config, 'learning_rate', 1e-4))
                                new_lr = max(old_lr * 0.1, min_lr)
                                pg['lr'] = new_lr
                            if len(getattr(self.optimizer, 'param_groups', [])) > 0:
                                sample_lr = self.optimizer.param_groups[0].get('lr', None)
                        except Exception as _e:
                            print(f"[AUTOMIT] Warning: failed to update optimizer LR: {_e}")
                    print(f"[AUTOMIT] Reduced optimizer LR; sample lr={sample_lr}")

                    # Mark mitigation applied to avoid repeated toggles
                    self.degenerate_mitigation_applied = True
                    # Log a warning that training should still be inspected manually
                    print("[AUTOMIT] Automated mitigation applied — please inspect dataset, labels, and consider stronger oversampling or restarting training with --oversample_minority")
                except Exception as e:
                    print(f"[AUTOMIT] Failed to apply automated mitigation: {e}")
            print(f"{'='*80}\n")
        
        # Calculate metrics with current predictions
        precision, recall, f1, auc_score, accuracy, macro_f1 = calculate_metrics(y_true, y_pred, y_probs, epoch+1)
        
        # EXPERIMENTAL: Try class-balanced threshold for better detection
        # With 74% Fake, 26% Real, adjust threshold to compensate for imbalance
        if hasattr(self, 'class_weights') and self.class_weights is not None:
            # Use inverse of class distribution as threshold adjustment
            # If training is 74% Fake (label=1), we should be MORE conservative about predicting Fake
            fake_ratio = 0.74  # Could calculate from training data
            balanced_threshold = 1.0 - fake_ratio  # = 0.26 (require only 26% prob to predict Real)
            
            print(f"\n[THRESHOLD TUNING] Default threshold: 0.50")
            print(f"[THRESHOLD TUNING] Class-balanced threshold: {balanced_threshold:.2f}")
            print(f"[THRESHOLD TUNING] Testing balanced threshold...")
            
            # Re-calculate predictions with balanced threshold
            y_probs_array = np.array(y_probs)
            y_pred_balanced = (y_probs_array >= balanced_threshold).astype(int)
            
            # Calculate metrics with balanced threshold
            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, confusion_matrix
            
            precision_b, recall_b, f1_b, _ = precision_recall_fscore_support(
                y_true, y_pred_balanced, average='binary', zero_division=0
            )
            accuracy_b = accuracy_score(y_true, y_pred_balanced)
            
            # Calculate per-class metrics
            per_class_metrics = precision_recall_fscore_support(
                y_true, y_pred_balanced, average=None, zero_division=0
            )
            macro_f1_b = np.mean(per_class_metrics[2])  # Average of F1 scores
            
            try:
                auc_b = roc_auc_score(y_true, y_probs)
            except:
                auc_b = 0.0
            
            cm_balanced = confusion_matrix(y_true, y_pred_balanced)
            
            print(f"[THRESHOLD TUNING] Balanced threshold results:")
            print(f"  Accuracy: {accuracy_b:.4f} (was {accuracy:.4f})")
            print(f"  Macro F1: {macro_f1_b:.4f} (was {macro_f1:.4f})")
            print(f"  Confusion Matrix:\n{cm_balanced}")
            
            # Only print per-class metrics if we have both classes
            if len(per_class_metrics) > 1 and len(per_class_metrics[1]) > 1:
                print(f"  Real (0): Recall = {per_class_metrics[1][0]:.4f}")
                print(f"  Fake (1): Recall = {per_class_metrics[1][1]:.4f}\n")
            else:
                print(f"  Warning: Only one class predicted in validation\n")
            
            # Check if balanced threshold helps
            if macro_f1_b > macro_f1:
                print(f"✅ Balanced threshold improves Macro F1: {macro_f1:.4f} → {macro_f1_b:.4f}")
                print(f"✅ Consider using threshold={balanced_threshold:.2f} for inference\n")
        
        # Check for degenerate solution in validation (model only predicting one class)
        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1:
            print(f"\n⚠️  WARNING: DEGENERATE SOLUTION IN VALIDATION!")
            print(f"⚠️  Model is only predicting class {unique_preds[0]} (always {'Fake' if unique_preds[0] == 1 else 'Real'})")
            print(f"⚠️  The model has learned a trivial solution. Training should be restarted with:")
            print(f"   1. Focal Loss (--loss_type focal --focal_gamma 3.0)")
            print(f"   2. Stronger class weights (--class_weights_mode manual_extreme)")
            print(f"   3. Lower learning rate (--learning_rate 0.00001)\n")
        
        # Store metrics (including macro F1 for balanced evaluation)
        self.metrics['val_losses'].append(avg_loss)
        self.metrics['val_accuracies'].append(accuracy)
        self.metrics['val_f1_scores'].append(f1)
        self.metrics['val_macro_f1_scores'].append(macro_f1)  # Key metric for imbalanced data
        self.metrics['val_auc_scores'].append(auc_score)
        
        # 🆕 CALCULATE PER-MODALITY METRICS
        if len(y_pred_audio) > 0 and len(y_pred_video) > 0:
            # Only compare samples where we have both audio and video predictions
            min_len = min(len(y_true), len(y_pred_audio), len(y_pred_video))
            if min_len > 0:
                y_true_subset = np.array(y_true[:min_len])
                y_pred_audio_subset = np.array(y_pred_audio[:min_len])
                y_pred_video_subset = np.array(y_pred_video[:min_len])
                
                audio_only_acc = np.mean(y_true_subset == y_pred_audio_subset)
                video_only_acc = np.mean(y_true_subset == y_pred_video_subset)
                print(f"\n🎯 PER-MODALITY DETECTION ACCURACY:")
                print(f"   Audio-only: {audio_only_acc:.4f} ({audio_only_acc*100:.2f}%)")
                print(f"   Video-only: {video_only_acc:.4f} ({video_only_acc*100:.2f}%)")
                print(f"   Combined:   {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                # Check which modality is stronger
                if audio_only_acc > video_only_acc:
                    print(f"   → Audio modality is stronger (+{(audio_only_acc - video_only_acc)*100:.1f}%)")
                elif video_only_acc > audio_only_acc:
                    print(f"   → Video modality is stronger (+{(video_only_acc - audio_only_acc)*100:.1f}%)")
                else:
                    print(f"   → Both modalities are balanced")
        
        # Plot confusion matrix
        if self.is_main_process:
            cm_path = plot_confusion_matrix(y_true, y_pred, epoch+1, self.plot_dir, split='val')
            
            # Log confusion matrix to WandB
            if self.config.use_wandb:
                wandb.log({f"confusion_matrix_epoch_{epoch+1}": wandb.Image(cm_path)})
                
            # Log component importance for feature analysis to WandB
            if self.config.use_wandb and all_component_contributions:
                component_importance = {}
                for key, values in all_component_contributions.items():
                    if values:
                        avg_value = np.mean(values)
                        component_importance[f"component_{key}"] = avg_value
                
                wandb.log(component_importance)
                
                # Create feature importance chart
                if component_importance:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                    except Exception:
                        plt = None

                    if plt is None:
                        print("[VIS] matplotlib not available - skipping validation feature importance plot")
                    else:
                        keys = sorted(component_importance.keys())
                        values = [component_importance[k] for k in keys]
                        # Sort by value
                        sorted_indices = np.argsort(values)
                        sorted_keys = [keys[i] for i in sorted_indices]
                        sorted_values = [values[i] for i in sorted_indices]

                        plt.figure(figsize=(12, 6))
                        plt.barh(sorted_keys, sorted_values)
                        plt.title("Feature Importance Scores")
                        plt.xlabel("Average Contribution")
                        plt.tight_layout()

                        feature_importance_path = os.path.join(self.plot_dir, f"feature_importance_epoch_{epoch+1}.png")
                        plt.savefig(feature_importance_path)
                        plt.close()

                        wandb.log({f"feature_importance_epoch_{epoch+1}": wandb.Image(feature_importance_path)})
        
        # Return metrics (include macro F1)
        # Restore original model parameters after EMA evaluation
        if self.ema is not None:
            self.ema.restore(self.model)
        return avg_loss, accuracy, precision, recall, f1, auc_score, macro_f1
    
    def test_model(self):
        """Test the trained model on the test set."""
        print("\nEvaluating model on test set...")
        self.model.eval()
        test_loss = 0
        y_true, y_pred, y_probs = [], [], []
        
        test_progress = tqdm(self.test_loader, desc="Testing")
        
        results_data = {
            'file_path': [],
            'true_label': [],
            'pred_label': [],
            'confidence': [],
            'issues_found': [],
            'confidence_score': []
        }
        
        # Track detailed component results for analysis
        detailed_component_results = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_progress):
                try:
                    # Move batch to device
                    batch = move_batch_to_device(batch, self.device, trainer=self)
                    log_batch_shapes(batch, prefix=f"after_move_to_device test batch {batch_idx}")

                    # Filter out placeholder samples marked with label == -1 in testing
                    try:
                        labels_tmp = batch.get('label', None)
                        if isinstance(labels_tmp, torch.Tensor):
                            labels_cpu = labels_tmp.detach()
                            if (labels_cpu == -1).any():
                                mask = (labels_cpu != -1)
                                keep_idx = mask.nonzero(as_tuple=False).squeeze(1)
                                num_removed = int((~mask).sum().item())
                                if keep_idx.numel() == 0:
                                    print(f"[INFO] Test batch {batch_idx} contains only placeholders. Skipping batch.")
                                    continue
                                for k, v in list(batch.items()):
                                    if isinstance(v, torch.Tensor) and v.shape[0] == labels_cpu.shape[0]:
                                        batch[k] = v[keep_idx]
                                    elif isinstance(v, list) and len(v) == labels_cpu.shape[0]:
                                        idxs = keep_idx.cpu().tolist()
                                        batch[k] = [v[i] for i in idxs]
                                print(f"[INFO] Filtered {num_removed} placeholder samples from test batch {batch_idx} (kept {keep_idx.numel()})")
                    except Exception as filter_e:
                        print(f"[WARNING] Could not filter placeholder samples in testing at batch {batch_idx}: {filter_e}")

                    # Get labels and file paths
                    labels = batch['label']
                    file_paths = batch.get('file_path', ['unknown'] * len(labels))
                    
                    # Forward pass
                    if self.amp_enabled:
                        with autocast():
                            outputs, results = self.model(batch)
                            # Log shapes after forward inside AMP path
                            log_batch_shapes(batch, prefix=f"post_forward_input train batch {batch_idx}")
                            if isinstance(outputs, torch.Tensor) and globals().get('DEBUG_MODE', False):
                                print(f"[SHAPES] outputs: shape={tuple(outputs.shape)}, dtype={outputs.dtype}, device={outputs.device}")
                            if isinstance(results, dict):
                                log_batch_shapes(results, prefix=f"post_forward_results train batch {batch_idx}")
                            loss = self.criterion(outputs, labels)
                    else:
                        outputs, results = self.model(batch)
                        # Log shapes after forward
                        log_batch_shapes(batch, prefix=f"post_forward_input train batch {batch_idx}")
                        if isinstance(outputs, torch.Tensor) and globals().get('DEBUG_MODE', False):
                            print(f"[SHAPES] outputs: shape={tuple(outputs.shape)}, dtype={outputs.dtype}, device={outputs.device}")
                        if isinstance(results, dict):
                            log_batch_shapes(results, prefix=f"post_forward_results train batch {batch_idx}")
                        loss = self.criterion(outputs, labels)
                    
                    # Update progress bar
                    test_progress.set_postfix(loss=f"{loss.item():.4f}")
                    
                    # Accumulate loss
                    test_loss += loss.item()
                    
                    # Accumulate predictions and labels for metrics calculation
                    batch_y_true = labels.cpu().numpy()
                    y_true.extend(batch_y_true)
                    
                    predictions = outputs.argmax(1).cpu().numpy()
                    y_pred.extend(predictions)
                    
                    confidences = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                    y_probs.extend(confidences)
                    
                    # Collect detailed results for analysis
                    for i in range(len(batch_y_true)):
                        issues = []
                        confidence_score = 0.5  # Default confidence
                        
                        # Extract explanation if available
                        if 'explanation' in results and results['explanation'] is not None:
                            explanation = results['explanation']
                            if 'issues_found' in explanation and isinstance(explanation['issues_found'], list):
                                issues = explanation['issues_found'][:3]  # Take first 3 issues
                            if 'confidence' in explanation:
                                confidence_score = explanation['confidence']
                        
                        # Basic result data
                        results_data['file_path'].append(file_paths[i] if i < len(file_paths) else 'unknown')
                        results_data['true_label'].append(int(batch_y_true[i]))
                        results_data['pred_label'].append(int(predictions[i]))
                        results_data['confidence'].append(float(confidences[i]))
                        results_data['issues_found'].append('; '.join(issues))
                        results_data['confidence_score'].append(float(confidence_score))
                        
                        # Extract and store component contributions for detailed analysis
                        if 'component_contributions' in results and results['component_contributions'] is not None:
                            for key, value in results['component_contributions'].items():
                                if key not in detailed_component_results:
                                    detailed_component_results[key] = []
                                
                                # Extract value for this sample
                                if isinstance(value, torch.Tensor):
                                    if value.numel() == 1:
                                        detailed_component_results[key].append(value.item())
                                    elif i < value.size(0):
                                        # First value for this batch item
                                        if value[i].numel() == 1:
                                            detailed_component_results[key].append(value[i].item())
                                        else:
                                            detailed_component_results[key].append(value[i].mean().item())
                                    else:
                                        detailed_component_results[key].append(0.0)
                                elif isinstance(value, (int, float)):
                                    detailed_component_results[key].append(value)
                                elif isinstance(value, dict) and 'naturalness' in value:
                                    if isinstance(value['naturalness'], torch.Tensor):
                                        detailed_component_results[key].append(value['naturalness'].item())
                                    else:
                                        detailed_component_results[key].append(value['naturalness'])
                                else:
                                    detailed_component_results[key].append(0.0)
                    
                    # Visualize sample predictions periodically
                    if (batch_idx + 1) % self.config.visualization_interval == 0 and self.is_main_process:
                        try:
                            sample_idx = np.random.randint(min(4, outputs.size(0)))
                            save_visualizations(
                                batch, outputs, results, 
                                0, sample_idx, 
                                os.path.join(self.viz_dir, "test_results"),
                                model=self.model
                            )
                        except Exception as vis_error:
                            print(f"Error visualizing predictions: {vis_error}")
                    
                except Exception as e:
                    print(f"Error in test batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Calculate average loss and metrics
        avg_loss = test_loss / len(self.test_loader)
        metrics_dict = calculate_metrics(y_true, y_pred, y_probs, 0, return_dict=True)
        
        print("\nTest Results:")
        print(f"- Loss     : {avg_loss:.4f}")
        print(f"- Accuracy : {metrics_dict['accuracy']:.4f}")
        print(f"- Precision: {metrics_dict['precision']:.4f}")
        print(f"- Recall   : {metrics_dict['recall']:.4f}")
        print(f"- F1 Score : {metrics_dict['f1']:.4f}")
        print(f"- AUC Score: {metrics_dict['auc']:.4f}")
        
        # Plot confusion matrix for test set
        if self.is_main_process:
            cm_path = plot_confusion_matrix(y_true, y_pred, 0, self.plot_dir, split='test')
            
            # Log confusion matrix to WandB
            if self.config.use_wandb:
                wandb.log({
                    "test_confusion_matrix": wandb.Image(cm_path),
                    "test_accuracy": metrics_dict['accuracy'],
                    "test_precision": metrics_dict['precision'],
                    "test_recall": metrics_dict['recall'],
                    "test_f1": metrics_dict['f1'],
                    "test_auc": metrics_dict['auc']
                })
        
        # Save detailed results to CSV
        if self.is_main_process:
            # Basic results: try pandas, fallback to csv module if pandas is not available
            try:
                import pandas as pd
            except Exception:
                pd = None

            if pd is not None:
                results_df = pd.DataFrame(results_data)
                results_path = os.path.join(self.log_dir, "test_results.csv")
                results_df.to_csv(results_path, index=False)
                print(f"Test results saved to: {results_path}")
            else:
                # Fallback: write CSV manually
                import csv
                results_path = os.path.join(self.log_dir, "test_results.csv")
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    headers = list(results_data.keys())
                    writer.writerow(headers)
                    rows = zip(*[results_data[h] for h in headers])
                    for row in rows:
                        writer.writerow(row)
                print(f"Test results saved (csv fallback) to: {results_path}")
            
            # Add component results to detailed analysis DataFrame
            for key, values in detailed_component_results.items():
                if len(values) == len(results_data['file_path']):
                    results_df[f"component_{key}"] = values
            
            # Save enhanced results with component details
            enhanced_results_path = os.path.join(self.log_dir, "test_results_detailed.csv")
            results_df.to_csv(enhanced_results_path, index=False)
            print(f"Detailed test results with component analysis saved to: {enhanced_results_path}")
            
            # Create feature importance visualization
            if detailed_component_results:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                except Exception:
                    plt = None

                if plt is None:
                    print("[VIS] matplotlib not available - skipping feature importance plot")
                else:
                # Calculate average importance for each component
                    component_importance = {}
                    for key, values in detailed_component_results.items():
                        if values:
                            component_importance[key] = np.mean(values)

                    # Sort components by importance
                    sorted_components = sorted(component_importance.items(), key=lambda x: x[1], reverse=True)

                    # Create feature importance chart
                    plt.figure(figsize=(12, 8))
                    component_names = [item[0] for item in sorted_components]
                    component_values = [item[1] for item in sorted_components]

                    plt.barh(component_names, component_values)
                    plt.title("Component Importance in Deepfake Detection")
                    plt.xlabel("Average Contribution")
                    plt.tight_layout()

                    feature_importance_path = os.path.join(self.plot_dir, "feature_importance_test.png")
                    plt.savefig(feature_importance_path)
                    plt.close()

                    if self.config.use_wandb:
                        wandb.log({"feature_importance_test": wandb.Image(feature_importance_path)})
        
        return avg_loss, metrics_dict
    
    def train_and_validate(self):
        """Train and validate the model for the specified number of epochs."""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        
        # Start from the resumed epoch if checkpoint was loaded
        start_epoch = getattr(self, 'start_epoch', 0)
        start_batch = getattr(self, 'start_batch', 0)
        
        if start_batch > 0:
            print(f"🔄 Resuming from epoch {start_epoch+1}, batch {start_batch}")
        else:
            print(f"🚀 Starting from epoch {start_epoch+1}")
        
        # Respect external shutdown signal set by signal handlers/cleanup
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                if is_shutdown_requested():
                    print(f"[SHUTDOWN] Shutdown requested before starting epoch {epoch}. Exiting training loop.")
                    break

                # ====== ACTIVATE QUANTIZATION-AWARE TRAINING (QAT) ======
                if self.qat_enabled and epoch == self.qat_start_epoch and not self.qat_active:
                    if self.is_main_process:
                        print("\n" + "="*80)
                        print(f"🔧 ACTIVATING QUANTIZATION-AWARE TRAINING at Epoch {epoch+1}")
                        print("="*80)
                    
                    # Import QAT utilities
                    from quantization_utils import prepare_model_for_qat
                    
                    # Store original FP32 model
                    self.model_fp32 = self.model
                    
                    # Prepare model for QAT
                    self.model = prepare_model_for_qat(
                        self.model,
                        backend=self.config.qat_backend
                    )
                    
                    # Move to device
                    self.model = self.model.to(self.device)
                    
                    # Wrap with DDP if distributed
                    if self.distributed:
                        self.model = DDP(
                            self.model,
                            device_ids=[self.local_rank],
                            output_device=self.local_rank,
                            find_unused_parameters=True
                        )
                    
                    # Reduce learning rate for QAT phase
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self.config.qat_lr_scale
                    
                    self.qat_active = True
                    
                    if self.is_main_process:
                        print(f"✅ QAT activated!")
                        print(f"✅ Learning rate reduced by {self.config.qat_lr_scale}x")
                        print("="*80 + "\n")

                epoch_start_time = time.time()
                
                # ====== AGGRESSIVE MEMORY CLEANUP BETWEEN EPOCHS ======
                # This prevents CUDA memory fragmentation which can cause crashes (0xC0000005)
                if torch.cuda.is_available():
                    # Synchronize all CUDA operations before cleanup
                    torch.cuda.synchronize()
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    # Force garbage collection
                    import gc
                    gc.collect()
                    if self.config.debug and self.is_main_process:
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"[CUDA] Memory cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

                # Training phase
                train_loss, train_acc, train_precision, train_recall, train_f1, train_auc, train_macro_f1 = self.train_epoch(epoch)

                # Validation phase
                val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_macro_f1 = self.validate_epoch(epoch)

                # Update learning rate scheduler if using plateau scheduler
                if self.scheduler is not None:
                    if self.config.scheduler == 'plateau':
                        # ReduceLROnPlateau step with macro F1 (better for imbalanced data)
                        self.scheduler.step(val_macro_f1)
                    else:
                        self.scheduler.step()

                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time

                # Print epoch summary
                if self.is_main_process:
                    epoch_summary = f"\nEpoch {epoch+1}/{self.config.num_epochs} completed in {epoch_time:.2f}s"
                    # Keep the star in console output but use (key) for log file
                    train_metrics_console = f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}, Macro F1: {train_macro_f1:.4f} ⭐, AUC: {train_auc:.4f}"
                    val_metrics_console = f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, Macro F1: {val_macro_f1:.4f} ⭐, AUC: {val_auc:.4f}"
                    
                    # Plain text versions for log file (no emoji)
                    train_metrics = f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}, Macro F1: {train_macro_f1:.4f} (key), AUC: {train_auc:.4f}"
                    val_metrics = f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, Macro F1: {val_macro_f1:.4f} (key), AUC: {val_auc:.4f}"
                    
                    print(train_metrics_console)
                    print(val_metrics_console)
                    
                    print(epoch_summary)
                    
                    # Log to file if specified
                    if self.log_file:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.log_file.write(f"{timestamp} - {epoch_summary}\n")
                        self.log_file.write(f"{timestamp} - {train_metrics}\n")
                        self.log_file.write(f"{timestamp} - {val_metrics}\n")
                        
                        # Log additional metrics in CSV format for easy analysis
                        self.log_file.write(f"EPOCH_CSV,{epoch+1},{train_loss:.6f},{train_acc:.6f},{train_precision:.6f},{train_recall:.6f},{train_f1:.6f},{train_macro_f1:.6f},{train_auc:.6f},")
                        self.log_file.write(f"{val_loss:.6f},{val_acc:.6f},{val_precision:.6f},{val_recall:.6f},{val_f1:.6f},{val_macro_f1:.6f},{val_auc:.6f}\n")
                        
                        self.log_file.flush()
                        
                # Save checkpoint (use macro F1 for balanced evaluation)
                self.save_checkpoint(epoch, val_acc, val_macro_f1)

                # Plot metrics with error handling
                print("[DEBUG] Starting to generate plots...")
                try:
                    for metric_name, train_values, val_values in [
                        ('loss', self.metrics['train_losses'], self.metrics['val_losses']),
                        ('accuracy', self.metrics['train_accuracies'], self.metrics['val_accuracies']),
                        ('f1', self.metrics['train_f1_scores'], self.metrics['val_f1_scores']),
                        ('macro_f1', self.metrics['train_macro_f1_scores'], self.metrics['val_macro_f1_scores']),  # Key for imbalanced data
                        ('auc', self.metrics['train_auc_scores'], self.metrics['val_auc_scores'])
                    ]:
                        print(f"[DEBUG] Plotting {metric_name} - Train: {len(train_values)} values, Val: {len(val_values)} values")
                        plot_path = plot_metrics(train_values, val_values, metric_name, epoch+1, self.plot_dir)
                        print(f"[DEBUG] Successfully saved {metric_name} plot to: {plot_path}")
                        
                        # Log metrics plot to WandB
                        if self.config.use_wandb:
                            wandb.log({f"{metric_name}_plot": wandb.Image(plot_path)})
                    
                    print("✅ All plots generated successfully!")
                    
                except Exception as e:
                    print(f"❌ Error generating plots: {e}")
                    import traceback
                    traceback.print_exc()
                    
                # Log epoch metrics to WandB
                if self.config.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc,
                        'train_precision': train_precision,
                        'val_precision': val_precision,
                        'train_recall': train_recall,
                        'val_recall': val_recall,
                        'train_f1': train_f1,
                        'val_f1': val_f1,
                        'train_macro_f1': train_macro_f1,  # Key metric for imbalanced data
                        'val_macro_f1': val_macro_f1,      # Key metric for imbalanced data
                        'train_auc': train_auc,
                        'val_auc': val_auc,
                        'learning_rate': get_lr(self.optimizer),
                        'epoch_time': epoch_time
                    })
                
                # Check for early stopping (using macro F1 for better balanced evaluation)
                if val_macro_f1 > self.best_val_f1:
                    self.best_val_f1 = val_macro_f1  # Now tracking macro F1
                    self.best_val_accuracy = val_acc
                    self.best_epoch = epoch + 1
                    self.early_stop_counter = 0
                    
                    # Save best model (using macro F1)
                    self.save_best_model(epoch, val_acc, val_macro_f1)
                else:
                    self.early_stop_counter += 1
                    print(f"Early stopping counter: {self.early_stop_counter}/{self.config.early_stopping_patience}")
                
                if self.early_stop_counter >= self.config.early_stopping_patience:
                    early_stop_msg = f"\nEarly stopping triggered after {epoch+1} epochs"
                    best_results_msg = f"Best validation Macro F1: {self.best_val_f1:.4f}, Accuracy: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})"
                    
                    # Messages without emojis for log file
                    print(early_stop_msg)
                    print(best_results_msg)
                    
                    # Log to file if specified
                    if self.log_file:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.log_file.write(f"{timestamp} - {early_stop_msg}\n")
                        self.log_file.write(f"{timestamp} - {best_results_msg}\n")
                        self.log_file.flush()
                    
                    break
        except KeyboardInterrupt:
            print("[SHUTDOWN] KeyboardInterrupt received. Running cleanup and exiting training.")
            cleanup_and_exit(trainer=self, save_checkpoint=True, reason='KeyboardInterrupt')
            return
        
        # Print training summary
        if self.is_main_process:
            training_complete_msg = "\nTraining completed!"
            best_results_msg = f"Best validation Macro F1: {self.best_val_f1:.4f}, Accuracy: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})"
            
            print(training_complete_msg)
            print(best_results_msg)
            
            # Log to file if specified
            if self.log_file:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_file.write(f"{timestamp} - {training_complete_msg}\n")
                self.log_file.write(f"{timestamp} - {best_results_msg}\n")
                self.log_file.flush()
            
            # Load best model for testing
            self.load_best_model()
            
            # ====== POST-TRAINING QAT CONVERSION ======
            if self.qat_enabled and self.qat_active:
                print("\n" + "="*80)
                print("🔄 POST-TRAINING: Converting QAT Model to INT8 Quantized Model")
                print("="*80)
                
                from quantization_utils import (
                    convert_qat_to_quantized,
                    validate_quantized_model,
                    export_quantized_model,
                    get_model_size
                )
                
                # Convert to INT8 quantized model
                model_quantized = convert_qat_to_quantized(self.model)
                
                # Validate quantization accuracy
                print("\n📊 Validating quantization accuracy...")
                acc_fp32, acc_int8, degradation = validate_quantized_model(
                    self.model_fp32 if self.model_fp32 is not None else self.model,
                    model_quantized,
                    self.val_loader,
                    device='cpu'  # Quantized models run on CPU
                )
                
                # Export quantized model
                quantized_path = os.path.join(self.log_dir, "model_int8_quantized.pth")
                export_quantized_model(
                    model_quantized,
                    quantized_path,
                    sample_input=None  # Can add sample input for ONNX export
                )
                
                # Save quantization report
                qat_report = {
                    'fp32_accuracy': float(acc_fp32),
                    'int8_accuracy': float(acc_int8),
                    'accuracy_degradation': float(degradation),
                    'model_size_fp32_mb': get_model_size(self.model),
                    'model_size_int8_mb': get_model_size(model_quantized),
                    'quantization_backend': self.config.qat_backend,
                    'qat_start_epoch': self.qat_start_epoch,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                qat_report_path = os.path.join(self.log_dir, "qat_report.json")
                with open(qat_report_path, 'w') as f:
                    json.dump(qat_report, f, indent=4)
                
                print(f"✅ Quantization report saved: {qat_report_path}")
                print(f"✅ INT8 model exported: {quantized_path}")
                print("="*80 + "\n")
            
            # Test the model
            test_loss, test_metrics = self.test_model()
            
            # Save final results
            final_results = {
                'best_epoch': self.best_epoch,
                'best_val_accuracy': float(self.best_val_accuracy),
                'best_val_f1': float(self.best_val_f1),
                'test_loss': float(test_loss),
                'test_accuracy': float(test_metrics['accuracy']),
                'test_precision': float(test_metrics['precision']),
                'test_recall': float(test_metrics['recall']),
                'test_f1': float(test_metrics['f1']),
                'test_auc': float(test_metrics['auc']),
                'training_time': time.time() - self.training_start_time if hasattr(self, 'training_start_time') else None,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'config': vars(self.config)
            }
            
            # Save as JSON
            final_results_path = os.path.join(self.log_dir, "final_results.json")
            with open(final_results_path, 'w') as f:
                # Handle non-serializable values
                serializable_results = {}
                for key, value in final_results.items():
                    if key == 'config':
                        # Convert config to serializable dict
                        config_dict = {}
                        for k, v in value.items():
                            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                                config_dict[k] = v
                            else:
                                config_dict[k] = str(v)
                        serializable_results[key] = config_dict
                    elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                        serializable_results[key] = value
                    else:
                        serializable_results[key] = str(value)
                
                json.dump(serializable_results, f, indent=4)
            
            print(f"Final results saved to: {final_results_path}")
            
            # Close log file if it was opened
            if self.log_file:
                final_msg = f"Training completed. Final results saved to: {final_results_path}"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_file.write(f"{timestamp} - {final_msg}\n")
                
                # Log test metrics
                test_metrics_msg = f"Test - Loss: {test_loss:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, " \
                                  f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, AUC: {test_metrics['auc']:.4f}"
                self.log_file.write(f"{timestamp} - {test_metrics_msg}\n")
                
                # Close the log file
                self.log_file.close()
                print(f"Training log saved to: {self.config.log_file}")
    
    def save_checkpoint(self, epoch, accuracy, f1_score):
        """Save model checkpoint."""
        # Guard: ensure `run_checkpoint_dir` exists (some code paths may not set it)
        try:
            if not hasattr(self, 'run_checkpoint_dir') or self.run_checkpoint_dir is None:
                timestamp = getattr(self, 'timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
                self.run_checkpoint_dir = os.path.join(getattr(self.config, 'checkpoint_dir', './checkpoints'), f"run_{timestamp}")
                try:
                    os.makedirs(self.run_checkpoint_dir, exist_ok=True)
                except Exception:
                    pass
        except Exception:
            pass

        checkpoint_dir = os.path.join(self.run_checkpoint_dir, "regular")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch_{epoch+1}_acc_{accuracy:.4f}_f1_{f1_score:.4f}.pth"
        )
        
        try:
            # Handle distributed/DataParallel models
            if self.distributed:
                model_state_dict = self.model.module.state_dict()
            elif hasattr(self.model, 'module'):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'config': vars(self.config)  # Save config for reproducibility
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Log file size
            file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"✅ Checkpoint saved: {checkpoint_path}")
            print(f"   File size: {file_size_mb:.2f} MB")
            
            # Force CUDA synchronization after checkpoint I/O to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()
    
    def save_best_model(self, epoch, accuracy, f1_score):
        """Save best model checkpoint."""
        best_model_path = os.path.join(self.run_checkpoint_dir, "best_model.pth")
        
        try:
            # Handle distributed/DataParallel models
            if self.distributed:
                model_state_dict = self.model.module.state_dict()
            elif hasattr(self.model, 'module'):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'config': vars(self.config)  # Save config for reproducibility
            }
            
            torch.save(checkpoint, best_model_path)
            
            # Log file size
            file_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
            print(f"🏆 Best model saved: {best_model_path}")
            print(f"   Epoch: {epoch+1} | Accuracy: {accuracy:.4f} | F1: {f1_score:.4f}")
            print(f"   File size: {file_size_mb:.2f} MB")
            
            # Copy to fixed best model location for easy reference
            fixed_best_path = os.path.join(self.model_dir, "best_model.pth")
            shutil.copy(best_model_path, fixed_best_path)
            print(f"   Copied to: {fixed_best_path}")
            
            # Force CUDA synchronization after large I/O operations
            # This prevents memory fragmentation crashes (0xC0000005)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error saving best model: {e}")
            import traceback
            traceback.print_exc()
    
    def load_best_model(self):
        """Load the best model for testing."""
        best_model_path = os.path.join(self.run_checkpoint_dir, "best_model.pth")
        
        if not os.path.exists(best_model_path):
            print(f"Best model not found at {best_model_path}. Using current model.")
            return
        
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        
        # Load model weights
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        print(f"Best model loaded (Epoch {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.4f}, F1: {checkpoint['f1_score']:.4f})")

    def run_pairwise(self, max_pairs=1000, steps=500, pairwise_batch_size=8, ckpt_interval=100, save_dir=None, amp_enabled=None):
        """
        Run a pairwise training routine that builds batches containing both the fake
        sample and its corresponding original sample. For each minibatch from
        `self.train_loader` we build a combined batch of shape [2*B, ...] where
        the first B samples are fakes and the next B are their originals. Labels
        are set to 1 for fake and 0 for original. Facial features and other
        per-sample tensors are concatenated in the same order so the model sees
        aligned pairs.

        The method supports AMP when `self.amp_enabled` is True (or when
        `amp_enabled=True` is passed). Checkpoints are saved to
        `self.run_checkpoint_dir/pairwise` by default.
        """
        # Resolve AMP preference
        amp = self.amp_enabled if amp_enabled is None else bool(amp_enabled)
        scaler = GradScaler() if amp else None

        # Prepare checkpoint/save directory
        if save_dir is None:
            save_dir = os.path.join(self.run_checkpoint_dir, 'pairwise')
        os.makedirs(save_dir, exist_ok=True)

        # Basic training mode
        self.model.train()

        processed_pairs = 0
        step = 0

        train_iter = iter(self.train_loader)

        print(f"[PAIRWISE] Starting pairwise training: max_pairs={max_pairs}, steps={steps}, batch_size={pairwise_batch_size}")

        while processed_pairs < max_pairs and step < steps and not is_shutdown_requested():
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    print("[PAIRWISE] Train loader empty, stopping pairwise loop")
                    break

            # Filter placeholder samples (label == -1)
            try:
                labels_tmp = batch.get('label', None)
                if isinstance(labels_tmp, torch.Tensor):
                    labels_cpu = labels_tmp.detach()
                    if (labels_cpu == -1).any():
                        mask_keep = (labels_cpu != -1)
                        keep_idx = mask_keep.nonzero(as_tuple=False).squeeze(1)
                        if keep_idx.numel() == 0:
                            continue
                        for k, v in list(batch.items()):
                            if isinstance(v, torch.Tensor) and v.shape[0] == labels_cpu.shape[0]:
                                batch[k] = v[keep_idx]
                            elif isinstance(v, list) and len(v) == labels_cpu.shape[0]:
                                idxs = keep_idx.cpu().tolist()
                                batch[k] = [v[i] for i in idxs]
            except Exception as e:
                print(f"[PAIRWISE] Warning: could not filter placeholders: {e}")

            # Ensure originals exist in batch
            if 'original_video_frames' not in batch or batch.get('original_video_frames') is None:
                # Skip non-paired samples
                continue

            # Move individual tensors to device (do not convert lists yet)
            batch = move_batch_to_device(batch, self.device, trainer=self)

            # Extract tensors (expect shapes: [B, T, C, H, W] and [B, L])
            fake_v = batch.get('video_frames')
            orig_v = batch.get('original_video_frames')
            fake_a = batch.get('audio')
            orig_a = batch.get('original_audio')

            # Some datasets may return originals as lists or tensors; attempt to coerce
            if not isinstance(orig_v, torch.Tensor):
                try:
                    orig_v = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in orig_v], dim=0).to(self.device)
                except Exception:
                    # Skip if we cannot form originals
                    continue

            if orig_a is None and fake_a is not None:
                # If original audio missing, use zeros with same length as fake audio
                orig_a = torch.zeros_like(fake_a)

            # Build combined batch by concatenating fake then original
            try:
                combined_video = torch.cat([fake_v, orig_v], dim=0)
            except Exception as e:
                print(f"[PAIRWISE] Error concatenating video tensors: {e}")
                continue

            if fake_a is not None and orig_a is not None:
                try:
                    combined_audio = torch.cat([fake_a, orig_a], dim=0)
                except Exception:
                    combined_audio = None
            else:
                combined_audio = None

            # Concatenate optional per-sample features if present (face_embeddings, facial_landmarks, etc.)
            combined = {
                'video_frames': combined_video,
                'audio': combined_audio
            }

            for key in ['face_embeddings', 'facial_landmarks', 'metadata_features', 'ela_features', 'pulse_signal', 'head_pose', 'eye_blink_features', 'frequency_features', 'mfcc_features', 'skin_color_variations']:
                if key in batch and batch[key] is not None:
                    fake_val = batch[key]
                    original_key = 'original_' + key
                    
                    if isinstance(fake_val, torch.Tensor):
                        try:
                            # Use REAL original features if available, otherwise duplicate fake features
                            if original_key in batch and batch[original_key] is not None:
                                original_val = batch[original_key]
                                combined[key] = torch.cat([fake_val, original_val], dim=0)
                                
                                # Log verification that features are different (only for debugging)
                                if self.debug and key in ['pulse_signal', 'facial_landmarks', 'mfcc_features']:
                                    fake_mean = fake_val.mean().item()
                                    orig_mean = original_val.mean().item()
                                    diff = abs(fake_mean - orig_mean)
                                    print(f"[CONTRASTIVE] {key}: fake_mean={fake_mean:.4f}, orig_mean={orig_mean:.4f}, diff={diff:.4f}")
                            else:
                                # Fallback: duplicate fake features (for features without original versions)
                                combined[key] = torch.cat([fake_val, fake_val], dim=0)
                                if self.debug:
                                    print(f"[WARNING] No original features for '{key}', using duplicated fake features")
                        except Exception as e:
                            # Final fallback: duplicate fake features
                            combined[key] = torch.cat([fake_val, fake_val], dim=0)
                            if self.debug:
                                print(f"[ERROR] Failed to concatenate {key}: {e}, using duplicated features")

            # Build labels: first B = fake (1), next B = original (0)
            B = fake_v.shape[0]
            labels_combined = torch.cat([torch.ones(B, dtype=torch.long, device=self.device), torch.zeros(B, dtype=torch.long, device=self.device)], dim=0)

            # Prepare fake_mask if provided: make mask for combined batch
            fake_mask_combined = None
            if 'fake_mask' in batch and batch['fake_mask'] is not None:
                try:
                    # batch['fake_mask'] may be list of lists or tensor [B, T]
                    fm = batch['fake_mask']
                    if isinstance(fm, list):
                        fm = [torch.tensor(m, dtype=torch.float32) for m in fm]
                        # Pad to same length
                        max_t = max([m.numel() for m in fm]) if fm else 0
                        fm_tensors = []
                        for m in fm:
                            if m.numel() < max_t:
                                pad = torch.zeros(max_t - m.numel(), dtype=torch.float32, device=self.device)
                                fm_tensors.append(torch.cat([m.to(self.device), pad]))
                            else:
                                fm_tensors.append(m.to(self.device))
                        fm_stack = torch.stack(fm_tensors, dim=0)
                    elif isinstance(fm, torch.Tensor):
                        fm_stack = fm.to(self.device)
                    else:
                        fm_stack = None
                    if fm_stack is not None:
                        zeros = torch.zeros_like(fm_stack)
                        fake_mask_combined = torch.cat([fm_stack, zeros], dim=0)
                        combined['fake_mask'] = fake_mask_combined
                except Exception as e:
                    if getattr(self, 'is_main_process', True):
                        print(f"[PAIRWISE] Warning: failed to prepare fake_mask: {e}")

            # Move combined optional fields to device if not already
            for k, v in list(combined.items()):
                if isinstance(v, torch.Tensor):
                    combined[k] = v.to(self.device)

            # Training step (single step per combined minibatch)
            try:
                self.optimizer.zero_grad()
                # Allow disabling AMP for debug runs
                amp_effective = bool(amp) and not getattr(self.config, 'debug_disable_amp', False)
                if amp_effective:
                    with autocast():
                        outputs, results = self.model(combined)
                        loss = self.criterion(outputs, labels_combined)
                    try:
                        if getattr(self.config, 'debug_autograd_detect', False):
                            with torch.autograd.detect_anomaly():
                                scaler.scale(loss).backward()
                        else:
                            scaler.scale(loss).backward()
                    except Exception as back_err:
                        print(f"[ERROR] Exception during AMP backward (pairwise) at step {step}: {back_err}")
                        import traceback
                        traceback.print_exc()
                        self.optimizer.zero_grad()
                        self.nan_count += 1
                        continue
                    scaler.unscale_(self.optimizer)

                    # Gradient clipping with optional stricter debug clip and logging
                    max_norm = float(self.config.gradient_clip)
                    if getattr(self.config, 'debug_strict_clip', False):
                        max_norm = min(max_norm, 1.0)
                    if getattr(self.config, 'debug_log_grad_stats', False):
                        try:
                            max_g = 0.0
                            nan_found = False
                            for name, p in self.model.named_parameters():
                                if p.grad is None:
                                    continue
                                grad_abs_max = float(p.grad.detach().abs().max().cpu().item())
                                if grad_abs_max > max_g:
                                    max_g = grad_abs_max
                                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                    print(f"[DEBUG] NaN/Inf in grad for {name}")
                                    nan_found = True
                            print(f"[DEBUG] Pre-clip max grad abs={max_g:.6e}, nan_found={nan_found}")
                        except Exception:
                            pass

                    if hasattr(self.model, 'clip_gradients'):
                        self.model.clip_gradients(max_norm=max_norm)
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                    if getattr(self.config, 'debug_log_grad_stats', False):
                        try:
                            max_g_post = 0.0
                            for name, p in self.model.named_parameters():
                                if p.grad is None:
                                    continue
                                grad_abs_max = float(p.grad.detach().abs().max().cpu().item())
                                if grad_abs_max > max_g_post:
                                    max_g_post = grad_abs_max
                            print(f"[DEBUG] Post-clip max grad abs={max_g_post:.6e}")
                        except Exception:
                            pass

                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs, results = self.model(combined)
                    loss = self.criterion(outputs, labels_combined)
                    try:
                        if getattr(self.config, 'debug_autograd_detect', False):
                            with torch.autograd.detect_anomaly():
                                loss.backward()
                        else:
                            loss.backward()
                    except Exception as back_err:
                        print(f"[ERROR] Exception during backward (pairwise non-AMP) at step {step}: {back_err}")
                        import traceback
                        traceback.print_exc()
                        if getattr(self, 'debug_save_on_anomaly', False):
                            try:
                                self._save_debug_snapshot('pairwise_anomaly', step, combined if 'combined' in locals() else None, outputs if 'outputs' in locals() else None, loss if 'loss' in locals() else None)
                            except Exception as e:
                                print(f"[DEBUG] Failed to save pairwise snapshot: {e}")
                        self.optimizer.zero_grad()
                        self.nan_count += 1
                        continue

                    # Gradient clipping (non-AMP) with debug options
                    max_norm = float(self.config.gradient_clip)
                    if getattr(self.config, 'debug_strict_clip', False):
                        max_norm = min(max_norm, 1.0)
                    if getattr(self.config, 'debug_log_grad_stats', False):
                        try:
                            max_g = 0.0
                            nan_found = False
                            for name, p in self.model.named_parameters():
                                if p.grad is None:
                                    continue
                                grad_abs_max = float(p.grad.detach().abs().max().cpu().item())
                                if grad_abs_max > max_g:
                                    max_g = grad_abs_max
                                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                    print(f"[DEBUG] NaN/Inf in grad for {name}")
                                    nan_found = True
                            print(f"[DEBUG] Pre-clip max grad abs={max_g:.6e}, nan_found={nan_found}")
                        except Exception:
                            pass

                    if hasattr(self.model, 'clip_gradients'):
                        self.model.clip_gradients(max_norm=max_norm)
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                    if getattr(self.config, 'debug_log_grad_stats', False):
                        try:
                            max_g_post = 0.0
                            for name, p in self.model.named_parameters():
                                if p.grad is None:
                                    continue
                                grad_abs_max = float(p.grad.detach().abs().max().cpu().item())
                                if grad_abs_max > max_g_post:
                                    max_g_post = grad_abs_max
                            print(f"[DEBUG] Post-clip max grad abs={max_g_post:.6e}")
                        except Exception:
                            pass

                    self.optimizer.step()

                processed_pairs += B
                step += 1

                if self.is_main_process and step % 10 == 0:
                    print(f"[PAIRWISE] Step {step}: processed_pairs={processed_pairs}, loss={loss.item():.6f}")

                if step % ckpt_interval == 0:
                    # Save checkpoint for pairwise run
                    ckpt_path = os.path.join(save_dir, f"pairwise_step_{step}_pairs_{processed_pairs}.pth")
                    try:
                        model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
                        torch.save({
                            'step': step,
                            'processed_pairs': processed_pairs,
                            'model_state_dict': model_state,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                        }, ckpt_path)
                        if self.is_main_process:
                            print(f"[PAIRWISE] Checkpoint saved: {ckpt_path}")
                    except Exception as e:
                        print(f"[PAIRWISE] Failed to save checkpoint: {e}")

            except Exception as e:
                print(f"[PAIRWISE] Error during training step: {e}")
                import traceback
                traceback.print_exc()
                # Clear gradients and continue
                try:
                    self.optimizer.zero_grad()
                except Exception:
                    pass
                continue

        print(f"[PAIRWISE] Finished pairwise training: processed_pairs={processed_pairs}, steps={step}")
    
    def print_training_improvements(self):
        """Print the advanced training configuration details."""
        print("\n" + "="*70)
        print("🔥 ADVANCED DEEPFAKE DETECTION TRAINING")
        print("="*70)
        
        # Multimodal Architecture
        print("✅ MULTIMODAL ARCHITECTURE (40+ COMPONENTS):")
        print(f"   🎭 Facial Analysis: Landmarks, micro-expressions, head pose, eye dynamics")
        print(f"   💓 Physiological: Heartbeat, blood flow, breathing, skin color")
        print(f"   🎤 Audio: Voice biometrics, MFCC, pitch, spectrogram")
        print(f"   🔍 Forensics: ELA, compression artifacts, metadata")
        print(f"   🧠 Advanced: Temporal consistency, attention, multi-scale fusion")
        
        # Production Robustness
        print("\n✅ PRODUCTION ROBUSTNESS:")
        print(f"   📱 Social Media: Instagram, TikTok, WhatsApp compression (multi-round)")
        print(f"   📐 Resolution: 4 quality levels (high/mid/low/very_low)")
        print(f"   💡 Lighting: Low-light, overexposed, shadows, color temperature")
        print(f"   👥 Fairness: Balanced sampling across demographics")
        print(f"   🌐 Domain Adaptation: Adversarial training for distribution shift")
        
        # Component Diversity
        print("\n✅ COMPONENT DIVERSITY (AUXILIARY LOSSES):")
        print(f"   🎯 Per-component auxiliary classifiers (5 key components)")
        print(f"   🔀 Diversity penalty prevents feature correlation")
        print(f"   🔇 Silent module detection (<1% contribution flagged)")
        print(f"   📊 Component contribution tracking with EMA")
        
        # Quantization-Aware Training
        if self.qat_enabled:
            print("\n✅ QUANTIZATION-AWARE TRAINING (QAT):")
            print(f"   🔧 Starts at epoch: {self.qat_start_epoch}")
            print(f"   💾 INT8 deployment: 4x smaller, 2-4x faster inference")
            print(f"   📉 Target accuracy drop: <2%")
            print(f"   🚀 Backend: {self.config.qat_backend}")
        
        # Class Imbalance Fixes
        print("\n✅ CLASS IMBALANCE & LOSS:")
        loss_type = getattr(self.config, 'loss_type', 'ce')
        if loss_type == 'focal':
            alpha = getattr(self.config, 'focal_alpha', 1.0)
            gamma = getattr(self.config, 'focal_gamma', 2.0)
            print(f"   🎯 Focal Loss (α={alpha}, γ={gamma}) - focuses on hard examples")
        else:
            print(f"   📊 Cross-Entropy Loss with class weighting")
        
        if getattr(self.config, 'use_weighted_loss', False):
            print(f"   ⚖️ Class-balanced weights enabled")
        
        print(f"   📏 Macro F1 as primary metric")
        
        # Overfitting Prevention
        print("\n✅ OVERFITTING PREVENTION:")
        dropout = getattr(self.config, 'dropout_rate', 0.0)
        if dropout > 0:
            print(f"   🛡️ Dropout: {dropout:.1%}")
        
        print(f"   📉 L2 Weight decay: {self.config.weight_decay}")
        print(f"   ⏹️ Early stopping: {self.config.early_stopping_patience} epochs")
        print(f"   ✂️ Gradient clipping: {self.config.gradient_clip}")
        print(f"   🔀 Component diversity enforcement")
        
        print("="*70)
        print("Ready to train with production-grade configuration! 🚀")
        print("="*70 + "\n")
    
    def run(self):
        """
        Run the full training pipeline with comprehensive error handling.
        🧼 PROCESS SAFETY: Includes proper cleanup to prevent zombie processes.
        """
        try:
            # Record start time
            self.training_start_time = time.time()
            
            # Print initial memory status
            if torch.cuda.is_available():
                allocated, reserved = get_gpu_memory_usage()
                print(f"[MEMORY] Initial GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            print("🧼 Starting training with GPU memory management enabled")
            
            # Train and validate
            self.train_and_validate()
            
            # Print final memory statistics
            if self.is_main_process:
                print(f"\n[MEMORY] Training completed with {self.oom_errors} out-of-memory errors and {self.memory_warnings} memory warnings")
                if torch.cuda.is_available():
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"[MEMORY] Final GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            print("✅ Training pipeline completed successfully")
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {e}")
            traceback.print_exc()
            # Cleanup on error
            cleanup_gpu_memory()
            raise
            
        finally:
            try:
                # Finalize experiment on WandB
                if self.config.use_wandb and self.is_main_process:
                    wandb.finish()
                
                # Clean up distributed training resources
                if self.distributed and dist.is_initialized():
                    dist.destroy_process_group()
                    
                # Final GPU cleanup
                cleanup_gpu_memory()
                print("🧼 Training cleanup completed")
                
            except Exception as cleanup_error:
                print(f"Warning during cleanup: {cleanup_error}")
        
        # Final GPU cache cleanup
        clear_gpu_cache()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multimodal Deepfake Detection Training")
    
    # Data parameters
    parser.add_argument('--json_path', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for runs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    
    # Model parameters
    parser.add_argument('--backbone_visual', type=str, default='efficientnet', choices=['efficientnet', 'swin'], help='Visual backbone architecture')
    parser.add_argument('--backbone_audio', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert'], help='Audio backbone architecture')
    parser.add_argument('--fusion_type', type=str, default='attention', choices=['attention', 'concat'], help='Fusion type for multimodal features')
    parser.add_argument('--video_feature_dim', type=int, default=1024, help='Dimension of video features')
    parser.add_argument('--audio_feature_dim', type=int, default=1024, help='Dimension of audio features')
    parser.add_argument('--transformer_dim', type=int, default=768, help='Dimension of transformer encoder')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer encoder layers')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes (real/fake)')
    parser.add_argument('--num_deepfake_types', type=int, default=7, help='Number of deepfake types for fine-grained classification')
    parser.add_argument('--enable_face_mesh', action='store_true', help='Enable face mesh analysis')
    parser.add_argument('--enable_explainability', action='store_true', help='Enable model explainability components')
    parser.add_argument('--use_spectrogram', action='store_true', help='Use audio spectrogram features')
    parser.add_argument('--detect_deepfake_type', action='store_true', help='Enable deepfake type detection')
    parser.add_argument('--deepfake_type_weight', type=float, default=0.3, help='Weight for deepfake type classification loss')
    parser.add_argument('--detect_faces', action='store_true', help='Enable face detection')
    parser.add_argument('--compute_spectrograms', action='store_true', help='Compute audio spectrograms')
    parser.add_argument('--temporal_features', action='store_true', help='Compute temporal consistency features')
    parser.add_argument('--enhanced_preprocessing', action='store_true', help='Enable enhanced preprocessing features (physiological, etc.)')
    parser.add_argument('--enhanced_augmentation', action='store_true', help='Enable enhanced data augmentation')
    parser.add_argument('--enable_skin_color_analysis', action='store_true', default=False, help='Enable skin color analysis (memory intensive)')
    parser.add_argument('--enable_advanced_physiological', action='store_true', help='Enable advanced physiological analysis (heartbeat, blood flow, breathing)')
    parser.add_argument('--physiological_fps', type=int, default=30, help='Frame rate for physiological signal analysis')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from')
    parser.add_argument('--resume_weights_only', action='store_true', help='Only load model weights from checkpoint, reset optimizer/scheduler for new hyperparameters')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use (None for full dataset)')
    parser.add_argument('--num_workers', type=int, default=0, help='🧼 SAFETY: Number of data loader workers (default=0, optimal for complex multimodal datasets)')  # Reverted based on benchmark results
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split ratio')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use class-weighted loss function')
    
    # 🔥 Class Imbalance & Bias Fixes
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'], help='Loss function type: ce (CrossEntropy) or focal (FocalLoss for imbalanced data)')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Alpha parameter for Focal Loss (weight for rare class)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss (focusing parameter)')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing for FocalLoss / CrossEntropy (0.0 = no smoothing)')
    parser.add_argument('--oversample_minority', action='store_true', help='Oversample minority class (Real) to balance dataset')
    parser.add_argument('--oversample_factor', type=float, default=1.0, help='Multiplicative factor for oversampling size (e.g. 3.0 increases oversampling)')
    parser.add_argument('--class_weights_mode', type=str, default='balanced', choices=['balanced', 'sqrt_balanced', 'manual', 'manual_extreme', 'none'], help='How to compute class weights')
    # Automated mitigation flags for degenerate validation collapse
    parser.add_argument('--enable_degenerate_auto_mitigation', action='store_true', help='Automatically apply mitigation when validation predicts only one class')
    parser.add_argument('--degenerate_mitigation_threshold', type=int, default=1, help='Number of consecutive degenerate validations before auto-mitigation')
    parser.add_argument('--auto_mitigation_focal_gamma', type=float, default=3.0, help='Focal gamma to use when auto-mitigation triggers')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate when auto-mitigation reduces LR')
    
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine_with_restarts', choices=['step', 'cosine', 'cosine_with_restarts', 'plateau', 'none'], help='Learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Gamma for cosine_with_restarts / StepLR scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--gradient_clip', type=float, default=0.5, help='Gradient clipping value')  # Reduced for stability
    
    # 🔥 Regularization & Overfitting Prevention
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for regularization (0.0-0.8)')
    parser.add_argument('--l2_reg_strength', type=float, default=1e-4, help='L2 regularization strength')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate (0 to disable, 0.999 recommended)')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha (0 to disable, 0.2 recommended)')
    
    # 🔥 Quantization-Aware Training (QAT) for Deployment
    parser.add_argument('--enable_qat', action='store_true', help='Enable Quantization-Aware Training for INT8 deployment')
    parser.add_argument('--qat_backend', type=str, default='fbgemm', choices=['fbgemm', 'qnnpack'], help='Quantization backend: fbgemm (x86) or qnnpack (ARM)')
    parser.add_argument('--qat_start_epoch', type=int, default=15, help='Epoch to start QAT (after initial convergence)')
    parser.add_argument('--qat_lr_scale', type=float, default=0.1, help='Learning rate scale for QAT phase (0.1 = 10x lower)')
    
    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    # Logging and visualization parameters
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='deepfake-detection', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging batch results')
    parser.add_argument('--log_file', type=str, default='', help='Path to save training and validation logs')
    parser.add_argument('--visualization_interval', type=int, default=50, help='Interval for visualizing predictions')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate checkpoints')
    parser.add_argument('--save_intermediate_interval', type=int, default=20, help='Interval for saving intermediate checkpoints')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--amp_enabled', action='store_true', default=False, help='Enable automatic mixed precision (AMP)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model weights')
    # Debugging helpers (used by train_production_mobile.ps1 quick debug runs)
    parser.add_argument('--debug_disable_amp', action='store_true', help='Disable AMP globally for debug runs')
    parser.add_argument('--debug_autograd_detect', action='store_true', help='Wrap backward with torch.autograd.detect_anomaly()')
    parser.add_argument('--debug_strict_clip', action='store_true', help='Enforce stricter gradient clipping (max_norm <= 1.0)')
    parser.add_argument('--debug_log_grad_stats', action='store_true', help='Log per-parameter gradient stats before and after clipping')
    parser.add_argument('--debug_log_aux_losses', action='store_true', help='Log auxiliary/diversity losses each batch for debugging')
    parser.add_argument('--debug_save_on_anomaly', action='store_true', help='Save model/batch snapshot when autograd anomaly or backward exception occurs')
    
    # Performance optimization parameters - 🧼 SAFETY OVERRIDES
    parser.add_argument('--pin_memory', action='store_true', default=False, help='🧼 SAFETY: Pin memory for faster data loading (default=False to prevent memory leaks)')
    parser.add_argument('--persistent_workers', action='store_true', default=False, help='🧼 SAFETY: Keep workers alive between epochs (default=False to prevent hanging processes)')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of samples loaded in advance by each worker (reduced for safety)')
    parser.add_argument('--reduce_frames', type=int, default=8, help='Reduce number of frames per video for faster processing (default: 8, original: 16)')
    parser.add_argument('--disable_skin_analysis', action='store_true', help='Disable memory-intensive skin color analysis for speed')
    parser.add_argument('--disable_advanced_physio', action='store_true', help='Disable advanced physiological analysis for speed')
    parser.add_argument('--fast_mode', action='store_true', help='Enable fast mode with reduced feature extraction')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps to simulate larger batch sizes')
    parser.add_argument('--max_frames', type=int, default=32, help='Number of frames to sample per video (default: 32)')
    
    return parser.parse_args()


def main():
    """
    Main entry point with comprehensive error handling and cleanup.
    🧼 PROCESS SAFETY: Includes proper cleanup to prevent zombie processes.
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Set up basic logging for debugging
    log_level = "INFO" if local_rank == 0 else "ERROR"  # Reduce noise from non-main processes
    
    try:
        print(f"[Rank {local_rank}] 🚀 Main function started (rank {local_rank}/{world_size})")
        
        # Parse arguments
        args = parse_args()
        # Launcher (PowerShell or CLI) now controls training hyperparameters.
        # Previous hard-coded overrides removed so callers can set desired flags.
        
        # Set local_rank from environment variable if running with torchrun
        if args.distributed and 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
            local_rank = args.local_rank
        
        # Suppress warnings
        suppress_warnings()
        
        print(f"[Rank {local_rank}] ⚙️ Configuration parsed and warnings suppressed")
        print(f"[Rank {local_rank}] 🧼 Starting training with process safety measures enabled")
        
        # Add environment debugging for main process
        if local_rank == 0:
            print("🔍 Environment Debug Info:")
            print(f"   - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
            print(f"   - WORLD_SIZE: {world_size}")
            print(f"   - LOCAL_RANK: {local_rank}")
            print(f"   - MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
            print(f"   - MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
            if torch.cuda.is_available():
                print(f"   - Available GPUs: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"     GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        
        # Create trainer with enhanced error handling
        try:
            print(f"[Rank {local_rank}] 🏗️ Creating DeepfakeTrainer...")
            trainer = DeepfakeTrainer(args)
            print(f"[Rank {local_rank}] ✅ DeepfakeTrainer created successfully")
        except Exception as trainer_creation_error:
            print(f"[Rank {local_rank}] ❌ Failed to create DeepfakeTrainer: {trainer_creation_error}")
            traceback.print_exc()
            raise
        
        # Run training with enhanced error handling
        try:
            print(f"[Rank {local_rank}] 🚀 Starting training run...")
            trainer.run()
            print(f"[Rank {local_rank}] ✅ Training completed successfully")
        except Exception as training_error:
            print(f"[Rank {local_rank}] ❌ Training run failed: {training_error}")
            traceback.print_exc()
            raise
        
    except KeyboardInterrupt:
        print(f"[Rank {local_rank}] 🧼 Training interrupted by user (Ctrl+C), cleaning up...")
        cleanup_processes()
        sys.exit(130)  # Standard exit code for Ctrl+C
        
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "cuda" in error_msg or "device" in error_msg or "gpu" in error_msg:
            print(f"[Rank {local_rank}] ❌ CUDA/GPU error: {e}")
            print(f"[Rank {local_rank}] 💡 Troubleshooting tips:")
            print(f"   - Check if GPUs are available and accessible")
            print(f"   - Try reducing batch size with --batch_size")
            print(f"   - Verify CUDA_VISIBLE_DEVICES is set correctly")
            print(f"   - Check for GPU memory issues")
        elif "nccl" in error_msg or "distributed" in error_msg:
            print(f"[Rank {local_rank}] ❌ Distributed training error: {e}")
            print(f"[Rank {local_rank}] 💡 Troubleshooting tips:")
            print(f"   - Try running without --distributed flag")
            print(f"   - Check network connectivity between nodes")
            print(f"   - Verify MASTER_ADDR and MASTER_PORT settings")
        else:
            print(f"[Rank {local_rank}] ❌ Runtime error: {e}")
        
        traceback.print_exc()
        cleanup_processes()
        sys.exit(1)
        
    except Exception as e:
        print(f"[Rank {local_rank}] ❌ Unexpected error: {e}")
        print(f"[Rank {local_rank}] 📋 Full error details:")
        traceback.print_exc()
        cleanup_processes()
        sys.exit(1)
        
    finally:
        # Final cleanup with error handling
        try:
            cleanup_processes()
            print(f"[Rank {local_rank}] 🧼 Process cleanup completed successfully")
        except Exception as cleanup_error:
            print(f"[Rank {local_rank}] ⚠️ Cleanup error (non-fatal): {cleanup_error}")
            # Don't fail on cleanup errors


if __name__ == "__main__":
    try:
        # Ensure multiprocessing start method and CUDA init happen only in main process
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Initialize CUDA error handler now (not at import time) so worker processes
        # won't attempt to initialize CUDA when spawn re-imports this module.
        try:
            _setup_cuda_error_handler()
        except Exception:
            pass

        main()
    except Exception as e:
        import traceback
        print("[GLOBAL ERROR] Unhandled exception in main():", e)
        traceback.print_exc()
        # Also log to error file if possible
        try:
            logs_dir = os.path.join(os.path.dirname(__file__), '../LAV_DF/dev/logs')
            os.makedirs(logs_dir, exist_ok=True)
            error_log_path = os.path.join(logs_dir, 'error.log')
            with open(error_log_path, 'a', encoding='utf-8') as ef:
                ef.write(f"[GLOBAL ERROR] Unhandled exception in main(): {e}\n")
                traceback.print_exc(file=ef)
        except Exception as log_err:
            print(f"[GLOBAL ERROR] Failed to write to error log: {log_err}")
        # Ensure cleanup
        try:
            cleanup_processes()
        except Exception:
            pass
        sys.exit(1)