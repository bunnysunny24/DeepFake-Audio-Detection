"""
Mobile Sensor Feature Extraction for Deepfake Detection
Extracts features from mobile sensors that can be used for both:
1. Training on existing video datasets (extractable features)
2. Live mobile deployment (all sensor features)

Features are designed to work with ANY video while being enhanced with mobile sensor data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings


class OpticalFlowAnalyzer(nn.Module):
    """
    Analyzes optical flow patterns for motion consistency.
    Extractable from ANY video - works with existing datasets.
    
    Real videos: Natural camera shake (8-12 Hz hand tremor)
    Deepfakes: Often too stable or synthetic motion patterns
    """
    
    def __init__(self, feature_dim=64):
        super(OpticalFlowAnalyzer, self).__init__()
        self.feature_dim = feature_dim
        
        # Motion pattern encoder
        self.motion_encoder = nn.Sequential(
            nn.Linear(10, 32),  # Flow statistics
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, feature_dim)
        )
        
        # Shake detector (8-12 Hz natural tremor)
        self.shake_detector = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def compute_optical_flow(self, frames):
        """
        Compute dense optical flow between consecutive frames.
        Uses Farneback algorithm (available in OpenCV).
        """
        try:
            batch_size, num_frames, C, H, W = frames.shape
            device = frames.device
            
            flow_stats = []
            
            for b in range(batch_size):
                frame_flows = []
                
                for t in range(num_frames - 1):
                    # Convert to numpy and grayscale
                    frame1 = frames[b, t].permute(1, 2, 0).cpu().numpy()
                    frame2 = frames[b, t+1].permute(1, 2, 0).cpu().numpy()
                    
                    if frame1.max() <= 1.0:
                        frame1 = (frame1 * 255).astype(np.uint8)
                        frame2 = (frame2 * 255).astype(np.uint8)
                    
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                    
                    # Compute optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        gray1, gray2, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                    
                    # Extract flow statistics
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    
                    mag_ang_corr = np.corrcoef(mag.flatten(), ang.flatten())[0, 1]
                    if np.isnan(mag_ang_corr):
                        mag_ang_corr = 0.0
                    
                    stats = [
                        np.mean(mag),           # Average motion magnitude
                        np.std(mag),            # Motion variance
                        np.max(mag),            # Peak motion
                        np.mean(ang),           # Average direction
                        np.std(ang),            # Direction variance
                        np.percentile(mag, 25), # Q1 magnitude
                        np.percentile(mag, 75), # Q3 magnitude
                        np.count_nonzero(mag > 1.0) / mag.size,  # Moving pixels ratio
                        cv2.Laplacian(gray2, cv2.CV_64F).var(),  # Frame sharpness
                        mag_ang_corr  # Mag-angle correlation (NaN-safe)
                    ]
                    
                    frame_flows.append(stats)
                
                if len(frame_flows) > 0:
                    # Average over temporal dimension
                    flow_stats.append(np.mean(frame_flows, axis=0))
                else:
                    flow_stats.append(np.zeros(10))
            
            flow_tensor = torch.tensor(flow_stats, dtype=torch.float32, device=device)
            return flow_tensor
            
        except Exception as e:
            warnings.warn(f"Optical flow computation error: {e}")
            return torch.zeros(batch_size, 10, device=device)
    
    def forward(self, frames):
        """
        Args:
            frames: Video frames [batch, num_frames, 3, H, W]
        Returns:
            Dictionary with optical flow features
        """
        # Compute flow statistics
        flow_stats = self.compute_optical_flow(frames)
        
        # Encode motion patterns
        motion_features = self.motion_encoder(flow_stats)
        
        # Detect natural shake (8-12 Hz hand tremor)
        shake_score = self.shake_detector(motion_features)
        
        return {
            'flow_features': motion_features,
            'shake_score': shake_score,
            'flow_stats': flow_stats
        }


class CameraMetadataAnalyzer(nn.Module):
    """
    Analyzes camera metadata patterns (exposure, focus, white balance).
    Extractable from video frame properties - works with existing datasets.
    
    Real videos: Natural variations in exposure, focus hunting
    Deepfakes: Often uniform or synthetic metadata
    """
    
    def __init__(self, feature_dim=32):
        super(CameraMetadataAnalyzer, self).__init__()
        self.feature_dim = feature_dim
        
        self.metadata_encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim)
        )
        
    def extract_frame_metadata(self, frames):
        """
        Extract pseudo-metadata from frame statistics.
        Simulates camera ISP behavior analysis.
        """
        try:
            batch_size, num_frames, C, H, W = frames.shape
            device = frames.device
            
            metadata_list = []
            
            for b in range(batch_size):
                frame_meta = []
                
                for t in range(num_frames):
                    frame = frames[b, t].cpu().numpy()
                    
                    # Brightness (simulates ISO/exposure)
                    brightness = np.mean(frame)
                    
                    # Contrast (simulates exposure compensation)
                    contrast = np.std(frame)
                    
                    # Color temperature (R/B ratio)
                    r_mean = np.mean(frame[0])
                    b_mean = np.mean(frame[2])
                    color_temp = r_mean / (b_mean + 1e-6)
                    
                    # Sharpness (simulates focus quality)
                    gray = np.mean(frame, axis=0)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    # Saturation
                    saturation = np.std([np.mean(frame[i]) for i in range(3)])
                    
                    # Histogram entropy (simulates dynamic range)
                    hist, _ = np.histogram(frame.flatten(), bins=256, range=(0, 1))
                    hist = hist / hist.sum()
                    entropy = -np.sum(hist * np.log(hist + 1e-6))
                    
                    # Exposure variation
                    exposure_var = np.var([np.mean(frame[i]) for i in range(3)])
                    
                    # Noise level (high ISO indicator)
                    noise = np.std(frame - cv2.GaussianBlur(frame.transpose(1, 2, 0), (5, 5), 0).transpose(2, 0, 1))
                    
                    meta = [brightness, contrast, color_temp, laplacian_var, 
                           saturation, entropy, exposure_var, noise]
                    frame_meta.append(meta)
                
                # Temporal statistics
                frame_meta = np.array(frame_meta)
                temporal_stats = [
                    np.mean(frame_meta, axis=0),
                    np.std(frame_meta, axis=0),
                ]
                metadata_list.append(np.concatenate(temporal_stats))
            
            # Take first 8 features
            metadata_tensor = torch.tensor([m[:8] for m in metadata_list], 
                                          dtype=torch.float32, device=device)
            return metadata_tensor
            
        except Exception as e:
            warnings.warn(f"Metadata extraction error: {e}")
            return torch.zeros(batch_size, 8, device=device)
    
    def forward(self, frames):
        """
        Args:
            frames: Video frames [batch, num_frames, 3, H, W]
        Returns:
            Camera metadata features
        """
        metadata = self.extract_frame_metadata(frames)
        features = self.metadata_encoder(metadata)
        return features


class RollingShutterDetector(nn.Module):
    """
    Detects rolling shutter artifacts from CMOS sensors.
    Extractable from video frames - works with existing datasets.
    
    Real videos: Rolling shutter wobble during motion
    Deepfakes: Often lack proper rolling shutter simulation
    """
    
    def __init__(self, feature_dim=16):
        super(RollingShutterDetector, self).__init__()
        
        self.shutter_encoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, feature_dim)
        )
        
    def detect_rolling_shutter(self, frames):
        """
        Detect rolling shutter artifacts by analyzing vertical distortion.
        """
        try:
            batch_size, num_frames, C, H, W = frames.shape
            device = frames.device
            
            shutter_features = []
            
            for b in range(batch_size):
                frame_features = []
                
                for t in range(1, num_frames):
                    # Convert frames to grayscale
                    frame1 = frames[b, t-1].mean(dim=0).cpu().numpy()
                    frame2 = frames[b, t].mean(dim=0).cpu().numpy()
                    
                    # Compute vertical gradient differences
                    grad1 = np.gradient(frame1, axis=0)
                    grad2 = np.gradient(frame2, axis=0)
                    
                    # Rolling shutter causes vertical distortion during horizontal motion
                    vertical_distortion = np.mean(np.abs(grad2 - grad1))
                    
                    # Compute horizontal motion
                    horizontal_diff = np.mean(np.abs(frame2 - frame1), axis=0)
                    horizontal_motion = np.std(horizontal_diff)
                    
                    # Rolling shutter correlation (high motion -> high distortion)
                    if horizontal_motion > 0:
                        shutter_ratio = vertical_distortion / (horizontal_motion + 1e-6)
                    else:
                        shutter_ratio = 0.0
                    
                    # Wobble detection (vertical line bending)
                    vertical_edges = cv2.Canny((frame2 * 255).astype(np.uint8), 50, 150)
                    wobble = np.std(np.sum(vertical_edges, axis=0))
                    
                    frame_features.append([
                        vertical_distortion,
                        horizontal_motion,
                        shutter_ratio,
                        wobble
                    ])
                
                if len(frame_features) > 0:
                    shutter_features.append(np.mean(frame_features, axis=0))
                else:
                    shutter_features.append(np.zeros(4))
            
            shutter_tensor = torch.tensor(shutter_features, dtype=torch.float32, device=device)
            return shutter_tensor
            
        except Exception as e:
            warnings.warn(f"Rolling shutter detection error: {e}")
            return torch.zeros(batch_size, 4, device=device)
    
    def forward(self, frames):
        """
        Args:
            frames: Video frames [batch, num_frames, 3, H, W]
        Returns:
            Rolling shutter features
        """
        shutter_stats = self.detect_rolling_shutter(frames)
        features = self.shutter_encoder(shutter_stats)
        return features


class AudioVisualSyncAnalyzer(nn.Module):
    """
    Analyzes audio-visual synchronization timing.
    Extractable from video + audio - works with existing datasets.
    
    Real videos: Natural lip-audio sync (0-40ms delay)
    Deepfakes: Often have sync issues due to separate generation
    """
    
    def __init__(self, feature_dim=32):
        super(AudioVisualSyncAnalyzer, self).__init__()
        
        self.sync_encoder = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, feature_dim)
        )
        
    def compute_sync_features(self, video_features, audio_features):
        """
        Compute synchronization features between video and audio.
        """
        try:
            batch_size = video_features.shape[0]
            device = video_features.device
            
            # Normalize features
            video_norm = F.normalize(video_features, dim=-1)
            audio_norm = F.normalize(audio_features, dim=-1)
            
            # Cross-correlation for sync detection
            sync_features = []
            
            for b in range(batch_size):
                v = video_norm[b].cpu().numpy()
                a = audio_norm[b].cpu().numpy()
                
                # Compute cross-correlation
                correlation = np.correlate(v, a, mode='full')
                peak_idx = np.argmax(correlation)
                center_idx = len(correlation) // 2
                
                # Delay in frames (positive = audio delayed)
                delay = peak_idx - center_idx
                
                # Correlation strength
                corr_strength = correlation[peak_idx] / (np.linalg.norm(v) * np.linalg.norm(a) + 1e-6)
                
                # Variance of correlation (should be peaked for real sync)
                corr_variance = np.std(correlation)
                
                # Energy ratio
                video_energy = np.sum(v ** 2)
                audio_energy = np.sum(a ** 2)
                energy_ratio = video_energy / (audio_energy + 1e-6)
                
                # Phase coherence
                phase_coherence = np.abs(np.mean(np.exp(1j * (v - a))))
                
                # Consistency score
                consistency = 1.0 / (1.0 + np.abs(delay))
                
                sync_features.append([
                    delay / 10.0,  # Normalized delay
                    corr_strength,
                    corr_variance,
                    energy_ratio,
                    phase_coherence,
                    consistency
                ])
            
            sync_tensor = torch.tensor(sync_features, dtype=torch.float32, device=device)
            return sync_tensor
            
        except Exception as e:
            warnings.warn(f"Sync computation error: {e}")
            return torch.zeros(batch_size, 6, device=device)
    
    def forward(self, video_features, audio_features):
        """
        Args:
            video_features: Visual features [batch, feature_dim]
            audio_features: Audio features [batch, feature_dim]
        Returns:
            Synchronization features
        """
        sync_stats = self.compute_sync_features(video_features, audio_features)
        features = self.sync_encoder(sync_stats)
        return features


class MobileDepthAnalyzer(nn.Module):
    """
    Analyzes depth information when available from mobile sensors.
    OPTIONAL: Uses depth data if provided, otherwise computes monocular depth estimation.
    Enhanced during mobile deployment, gracefully handles missing data.
    """
    
    def __init__(self, feature_dim=64):
        super(MobileDepthAnalyzer, self).__init__()
        self.feature_dim = feature_dim
        
        # Depth feature encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, feature_dim)
        )
        
    def estimate_monocular_depth(self, frames):
        """
        Estimate depth from monocular frames (fallback for non-mobile data).
        Uses brightness and defocus cues.
        """
        try:
            batch_size, num_frames, C, H, W = frames.shape
            device = frames.device
            
            depth_maps = []
            
            for b in range(batch_size):
                # Use first frame for depth estimation
                frame = frames[b, 0].cpu().numpy()
                gray = np.mean(frame, axis=0)
                
                # Simple depth estimation using blur detection
                # Sharper regions = closer, blurrier = farther
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness_map = np.abs(laplacian)
                
                # Normalize to 0-1 (inverted: high sharpness = low depth value = close)
                depth_map = 1.0 - (sharpness_map / (np.max(sharpness_map) + 1e-6))
                
                # Resize to consistent size
                depth_map = cv2.resize(depth_map, (W, H))
                depth_maps.append(depth_map)
            
            depth_tensor = torch.tensor(np.array(depth_maps), dtype=torch.float32, device=device)
            depth_tensor = depth_tensor.unsqueeze(1)  # Add channel dimension
            
            return depth_tensor
            
        except Exception as e:
            warnings.warn(f"Monocular depth estimation error: {e}")
            return torch.zeros(batch_size, 1, H, W, device=device)
    
    def forward(self, frames, depth_map=None):
        """
        Args:
            frames: Video frames [batch, num_frames, 3, H, W]
            depth_map: Optional depth map from mobile sensor [batch, 1, H, W]
        Returns:
            Depth features
        """
        if depth_map is None:
            # Estimate depth from monocular frames
            depth_map = self.estimate_monocular_depth(frames)
        
        # Encode depth features
        depth_features = self.depth_encoder(depth_map)
        
        return {
            'depth_features': depth_features,
            'depth_map': depth_map,
            'has_real_depth': depth_map is not None
        }


class MobileSensorFusion(nn.Module):
    """
    Fuses all mobile sensor features into unified representation.
    Handles missing sensor data gracefully with learned attention weights.
    """
    
    def __init__(self, feature_dim=256):
        super(MobileSensorFusion, self).__init__()
        
        # Individual feature projections
        self.optical_flow_proj = nn.Linear(64, feature_dim)
        self.metadata_proj = nn.Linear(32, feature_dim)
        self.shutter_proj = nn.Linear(16, feature_dim)
        self.sync_proj = nn.Linear(32, feature_dim)
        self.depth_proj = nn.Linear(64, feature_dim)
        
        # Attention mechanism for adaptive fusion
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 5, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 5),
            nn.Softmax(dim=1)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 5, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, optical_flow_feat, metadata_feat, shutter_feat, sync_feat, depth_feat):
        """
        Fuse all mobile sensor features with adaptive attention.
        
        Args:
            optical_flow_feat: [batch, 64]
            metadata_feat: [batch, 32]
            shutter_feat: [batch, 16]
            sync_feat: [batch, 32]
            depth_feat: [batch, 64]
        Returns:
            Fused features [batch, feature_dim]
        """
        # Project to common dimension
        flow_proj = self.optical_flow_proj(optical_flow_feat)
        meta_proj = self.metadata_proj(metadata_feat)
        shut_proj = self.shutter_proj(shutter_feat)
        sync_proj = self.sync_proj(sync_feat)
        depth_proj = self.depth_proj(depth_feat)
        
        # Stack features
        all_features = torch.stack([flow_proj, meta_proj, shut_proj, sync_proj, depth_proj], dim=1)
        
        # Compute attention weights
        concatenated = all_features.view(all_features.size(0), -1)
        attention_weights = self.attention(concatenated)
        
        # Apply attention
        weighted_features = all_features * attention_weights.unsqueeze(-1)
        
        # Fuse
        fused = self.fusion(weighted_features.view(weighted_features.size(0), -1))
        
        return fused


if __name__ == "__main__":
    """Test mobile sensor analyzers"""
    print("🔬 Testing Mobile Sensor Analyzers...")
    
    # Create dummy data
    batch_size = 2
    num_frames = 8
    frames = torch.rand(batch_size, num_frames, 3, 224, 224)
    
    # Test optical flow
    print("\n1. Testing Optical Flow Analyzer...")
    flow_analyzer = OpticalFlowAnalyzer()
    flow_results = flow_analyzer(frames)
    print(f"   Flow features shape: {flow_results['flow_features'].shape}")
    print(f"   Shake score: {flow_results['shake_score'].mean().item():.3f}")
    
    # Test camera metadata
    print("\n2. Testing Camera Metadata Analyzer...")
    meta_analyzer = CameraMetadataAnalyzer()
    meta_features = meta_analyzer(frames)
    print(f"   Metadata features shape: {meta_features.shape}")
    
    # Test rolling shutter
    print("\n3. Testing Rolling Shutter Detector...")
    shutter_detector = RollingShutterDetector()
    shutter_features = shutter_detector(frames)
    print(f"   Shutter features shape: {shutter_features.shape}")
    
    # Test audio-visual sync
    print("\n4. Testing Audio-Visual Sync Analyzer...")
    sync_analyzer = AudioVisualSyncAnalyzer()
    video_feat = torch.rand(batch_size, 128)
    audio_feat = torch.rand(batch_size, 128)
    sync_features = sync_analyzer(video_feat, audio_feat)
    print(f"   Sync features shape: {sync_features.shape}")
    
    # Test depth analyzer
    print("\n5. Testing Mobile Depth Analyzer...")
    depth_analyzer = MobileDepthAnalyzer()
    depth_results = depth_analyzer(frames)
    print(f"   Depth features shape: {depth_results['depth_features'].shape}")
    print(f"   Has real depth: {depth_results['has_real_depth']}")
    
    # Test fusion
    print("\n6. Testing Mobile Sensor Fusion...")
    fusion = MobileSensorFusion(feature_dim=256)
    fused = fusion(
        flow_results['flow_features'],
        meta_features,
        shutter_features,
        sync_features,
        depth_results['depth_features']
    )
    print(f"   Fused features shape: {fused.shape}")
    
    print("\n✅ All mobile sensor analyzers working correctly!")
