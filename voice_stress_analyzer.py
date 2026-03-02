"""
Voice Stress and Emotional Analysis for Deepfake Detection
Implements advanced voice stress detection techniques:
1. Jitter Analysis - Cycle-to-cycle pitch variations (voice instability)
2. Shimmer Analysis - Amplitude variations between periods (vocal stress)
3. Harmonic-to-Noise Ratio (HNR) - Voice quality measurement
4. Emotional State Detection - Stress, anxiety, fear detection
5. Voice Formant Analysis - Vocal tract resonance patterns

🎤 VOICE STRESS INDICATORS:
- Jitter > 1% indicates voice instability (common in synthetic voices)
- Shimmer > 3% indicates amplitude stress (deepfakes struggle with this)
- Low HNR (<10 dB) indicates noisy/synthetic voice quality
- Abnormal formant patterns reveal voice synthesis artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa
import warnings
import math


class JitterShimmerAnalyzer(nn.Module):
    """
    Analyzes jitter (pitch variations) and shimmer (amplitude variations).
    These are key indicators of voice stress and synthetic speech.
    """
    
    def __init__(self, sample_rate=16000):
        super(JitterShimmerAnalyzer, self).__init__()
        self.sample_rate = sample_rate
        
        # Neural feature extractor for jitter/shimmer patterns
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 32),  # 10 statistical features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Stress detector
        self.stress_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def compute_jitter(self, audio_waveform):
        """
        Compute jitter (cycle-to-cycle pitch variations).
        Jitter = average absolute difference between consecutive periods / average period
        """
        try:
            audio_np = audio_waveform.cpu().numpy()
            
            # Extract pitch using autocorrelation
            pitches = []
            frame_length = 2048
            hop_length = 512
            
            for i in range(0, len(audio_np) - frame_length, hop_length):
                frame = audio_np[i:i+frame_length]
                
                # Autocorrelation method for pitch detection
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find peaks (excluding zero lag)
                if len(autocorr) > 1:
                    # Normalize
                    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
                    
                    # Find first significant peak
                    min_lag = int(self.sample_rate / 500)  # Max 500 Hz
                    max_lag = int(self.sample_rate / 50)   # Min 50 Hz
                    
                    if max_lag < len(autocorr):
                        search_range = autocorr[min_lag:max_lag]
                        if len(search_range) > 0:
                            peak_idx = np.argmax(search_range) + min_lag
                            pitch = self.sample_rate / peak_idx if peak_idx > 0 else 0
                            pitches.append(pitch)
            
            if len(pitches) < 2:
                return 0.0
            
            # Compute period from pitch (period = 1/frequency)
            periods = [1.0/p if p > 0 else 0 for p in pitches]
            periods = [p for p in periods if p > 0]  # Remove invalid periods
            
            if len(periods) < 2:
                return 0.0
            
            # Jitter calculation
            period_diffs = np.abs(np.diff(periods))
            avg_period = np.mean(periods)
            jitter = np.mean(period_diffs) / avg_period if avg_period > 0 else 0.0
            
            # Convert to percentage
            return float(jitter * 100.0)
            
        except Exception as e:
            if globals().get('DEBUG_MODE', False):
                if not hasattr(self, '_jitter_warned'):
                    print(f"[DEBUG] Jitter computation error: {e}")
                    self._jitter_warned = True
            return 0.0
    
    def compute_shimmer(self, audio_waveform):
        """
        Compute shimmer (amplitude variations between periods).
        Shimmer = average absolute difference between consecutive amplitudes / average amplitude
        """
        try:
            audio_np = audio_waveform.cpu().numpy()
            
            # Extract amplitude envelope
            frame_length = 2048
            hop_length = 512
            amplitudes = []
            
            for i in range(0, len(audio_np) - frame_length, hop_length):
                frame = audio_np[i:i+frame_length]
                # RMS amplitude
                rms = np.sqrt(np.mean(frame**2))
                amplitudes.append(rms)
            
            if len(amplitudes) < 2:
                return 0.0
            
            amplitudes = np.array(amplitudes)
            
            # Shimmer calculation
            amplitude_diffs = np.abs(np.diff(amplitudes))
            avg_amplitude = np.mean(amplitudes)
            shimmer = np.mean(amplitude_diffs) / avg_amplitude if avg_amplitude > 0 else 0.0
            
            # Convert to percentage
            return float(shimmer * 100.0)
            
        except Exception as e:
            if globals().get('DEBUG_MODE', False):
                if not hasattr(self, '_shimmer_warned'):
                    print(f"[DEBUG] Shimmer computation error: {e}")
                    self._shimmer_warned = True
            return 0.0
    
    def compute_hnr(self, audio_waveform):
        """
        Compute Harmonic-to-Noise Ratio (HNR).
        Higher HNR indicates cleaner voice (more harmonic content).
        Deepfakes often have lower HNR due to synthesis artifacts.
        """
        try:
            audio_np = audio_waveform.cpu().numpy()
            
            # Use autocorrelation method
            frame_length = 2048
            hop_length = 512
            hnr_values = []
            
            for i in range(0, len(audio_np) - frame_length, hop_length):
                frame = audio_np[i:i+frame_length]
                
                # Autocorrelation
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                if len(autocorr) > 1 and autocorr[0] > 0:
                    # Find maximum in pitch range
                    min_lag = int(self.sample_rate / 500)
                    max_lag = int(self.sample_rate / 50)
                    
                    if max_lag < len(autocorr):
                        search_range = autocorr[min_lag:max_lag]
                        if len(search_range) > 0:
                            max_autocorr = np.max(search_range)
                            
                            # HNR = 10 * log10(max_autocorr / (autocorr[0] - max_autocorr))
                            noise = autocorr[0] - max_autocorr
                            if noise > 0:
                                hnr = 10 * np.log10(max_autocorr / noise)
                                hnr_values.append(hnr)
            
            if len(hnr_values) == 0:
                return 0.0
            
            return float(np.mean(hnr_values))
            
        except Exception as e:
            if globals().get('DEBUG_MODE', False):
                if not hasattr(self, '_hnr_warned'):
                    print(f"[DEBUG] HNR computation error: {e}")
                    self._hnr_warned = True
            return 0.0
    
    def forward(self, audio_waveform):
        """
        Analyze audio for jitter, shimmer, and stress indicators.
        
        Args:
            audio_waveform: Audio tensor [batch, samples]
            
        Returns:
            Dictionary with jitter, shimmer, HNR, and stress score
        """
        batch_size = audio_waveform.shape[0]
        device = audio_waveform.device
        
        jitter_values = []
        shimmer_values = []
        hnr_values = []
        
        # Process each sample in batch
        for i in range(batch_size):
            audio = audio_waveform[i]
            
            jitter = self.compute_jitter(audio)
            shimmer = self.compute_shimmer(audio)
            hnr = self.compute_hnr(audio)
            
            jitter_values.append(jitter)
            shimmer_values.append(shimmer)
            hnr_values.append(hnr)
        
        # Convert to tensors
        jitter_tensor = torch.tensor(jitter_values, device=device).unsqueeze(1)
        shimmer_tensor = torch.tensor(shimmer_values, device=device).unsqueeze(1)
        hnr_tensor = torch.tensor(hnr_values, device=device).unsqueeze(1)
        
        # Create feature vector with statistical indicators
        features = []
        for i in range(batch_size):
            feat = [
                jitter_values[i],
                shimmer_values[i],
                hnr_values[i],
                1.0 if jitter_values[i] > 1.0 else 0.0,  # Abnormal jitter flag
                1.0 if shimmer_values[i] > 3.0 else 0.0,  # Abnormal shimmer flag
                1.0 if hnr_values[i] < 10.0 else 0.0,     # Low HNR flag
                jitter_values[i] / 5.0,  # Normalized jitter (0-1 scale, 5% max)
                shimmer_values[i] / 10.0,  # Normalized shimmer (0-1 scale, 10% max)
                hnr_values[i] / 30.0,  # Normalized HNR (0-1 scale, 30 dB max)
                (jitter_values[i] + shimmer_values[i]) / 2.0  # Combined instability
            ]
            features.append(feat)
        
        features = torch.tensor(features, device=device, dtype=torch.float32)
        
        # Extract learned features
        learned_features = self.feature_extractor(features)
        
        # Predict stress score
        stress_score = self.stress_classifier(learned_features)
        
        return {
            'jitter': jitter_tensor,
            'shimmer': shimmer_tensor,
            'hnr': hnr_tensor,
            'stress_score': stress_score,
            'features': learned_features
        }


class EmotionalStateDetector(nn.Module):
    """
    Detects emotional states from voice that indicate stress, anxiety, or fear.
    Deepfakes often fail to replicate natural emotional voice patterns.
    """
    
    def __init__(self, feature_dim=64):
        super(EmotionalStateDetector, self).__init__()
        
        # Emotion classifier
        self.emotion_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        
        # Emotion heads
        self.stress_head = nn.Linear(32, 1)
        self.anxiety_head = nn.Linear(32, 1)
        self.fear_head = nn.Linear(32, 1)
        self.anger_head = nn.Linear(32, 1)
        self.overall_emotion = nn.Linear(32, 7)  # 7 basic emotions
        
    def forward(self, voice_features):
        """
        Detect emotional states from voice features.
        
        Args:
            voice_features: Features from jitter/shimmer analyzer [batch, feature_dim]
            
        Returns:
            Dictionary with emotion scores
        """
        # Encode emotions
        emotion_features = self.emotion_encoder(voice_features)
        
        # Predict individual emotions
        stress = torch.sigmoid(self.stress_head(emotion_features))
        anxiety = torch.sigmoid(self.anxiety_head(emotion_features))
        fear = torch.sigmoid(self.fear_head(emotion_features))
        anger = torch.sigmoid(self.anger_head(emotion_features))
        
        # Overall emotion distribution
        emotion_dist = F.softmax(self.overall_emotion(emotion_features), dim=1)
        
        return {
            'stress': stress,
            'anxiety': anxiety,
            'fear': fear,
            'anger': anger,
            'emotion_distribution': emotion_dist,
            'features': emotion_features
        }


class FormantAnalyzer(nn.Module):
    """
    Analyzes voice formants (resonance frequencies of vocal tract).
    Formant patterns reveal voice synthesis artifacts and unnatural vocal tract modeling.
    """
    
    def __init__(self, sample_rate=16000):
        super(FormantAnalyzer, self).__init__()
        self.sample_rate = sample_rate
        
        # Formant feature encoder
        self.formant_encoder = nn.Sequential(
            nn.Linear(12, 32),  # F1, F2, F3, F4 + their statistics
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    def extract_formants(self, audio_waveform):
        """
        Extract first 4 formants (F1, F2, F3, F4) from audio.
        Uses LPC (Linear Predictive Coding) analysis.
        """
        try:
            audio_np = audio_waveform.cpu().numpy()
            
            # Use librosa for formant estimation
            frame_length = 2048
            hop_length = 512
            
            all_formants = []
            
            for i in range(0, len(audio_np) - frame_length, hop_length):
                frame = audio_np[i:i+frame_length]
                
                # Pre-emphasis filter
                emphasized = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
                
                # LPC analysis (order 12 for 4 formants: 2 coefficients per formant)
                try:
                    # Simple formant estimation using FFT peaks
                    spectrum = np.abs(fft(emphasized))
                    freqs = fftfreq(len(emphasized), 1/self.sample_rate)
                    
                    # Only positive frequencies
                    pos_mask = freqs > 0
                    freqs = freqs[pos_mask]
                    spectrum = spectrum[pos_mask]
                    
                    # Find peaks in spectrum (formants)
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(spectrum, height=np.max(spectrum)*0.1, distance=20)
                    
                    if len(peaks) >= 4:
                        # Take first 4 strongest peaks as formants
                        peak_freqs = freqs[peaks]
                        peak_amps = spectrum[peaks]
                        sorted_indices = np.argsort(peak_amps)[-4:]
                        formants = sorted(peak_freqs[sorted_indices])[:4]
                    else:
                        # Default formants if detection fails
                        formants = [500, 1500, 2500, 3500]  # Typical adult male formants
                    
                    all_formants.append(formants)
                    
                except Exception:
                    all_formants.append([500, 1500, 2500, 3500])
            
            if len(all_formants) == 0:
                return [500, 1500, 2500, 3500], [0, 0, 0, 0]
            
            all_formants = np.array(all_formants)
            
            # Statistics: mean and std of each formant
            mean_formants = np.mean(all_formants, axis=0)
            std_formants = np.std(all_formants, axis=0)
            
            return mean_formants.tolist(), std_formants.tolist()
            
        except Exception as e:
            if globals().get('DEBUG_MODE', False):
                if not hasattr(self, '_formant_warned'):
                    print(f"[DEBUG] Formant extraction error: {e}")
                    self._formant_warned = True
            return [500, 1500, 2500, 3500], [0, 0, 0, 0]
    
    def forward(self, audio_waveform):
        """
        Analyze formant patterns.
        
        Args:
            audio_waveform: Audio tensor [batch, samples]
            
        Returns:
            Dictionary with formant features
        """
        batch_size = audio_waveform.shape[0]
        device = audio_waveform.device
        
        formant_features = []
        
        for i in range(batch_size):
            audio = audio_waveform[i]
            mean_formants, std_formants = self.extract_formants(audio)
            
            # Combine mean and std into feature vector
            features = mean_formants + std_formants + [
                mean_formants[1] - mean_formants[0],  # F2-F1 spacing
                mean_formants[2] - mean_formants[1],  # F3-F2 spacing
                mean_formants[3] - mean_formants[2],  # F4-F3 spacing
                np.mean(std_formants)  # Average formant variation
            ]
            formant_features.append(features)
        
        formant_tensor = torch.tensor(formant_features, device=device, dtype=torch.float32)
        
        # Encode formant patterns
        encoded = self.formant_encoder(formant_tensor)
        
        return {
            'formant_features': formant_tensor,
            'encoded_features': encoded
        }


class VoiceStressAnalyzer(nn.Module):
    """
    Complete voice stress and emotional analysis system.
    Combines jitter/shimmer, emotional detection, and formant analysis.
    """
    
    def __init__(self, sample_rate=16000, feature_dim=64):
        super(VoiceStressAnalyzer, self).__init__()
        
        self.jitter_shimmer_analyzer = JitterShimmerAnalyzer(sample_rate)
        self.emotional_detector = EmotionalStateDetector(feature_dim)
        self.formant_analyzer = FormantAnalyzer(sample_rate)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32 + 32, 128),  # Jitter/shimmer + emotion + formant features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, audio_waveform):
        """
        Complete voice stress analysis.
        
        Args:
            audio_waveform: Audio tensor [batch, samples]
            
        Returns:
            Dictionary with comprehensive voice stress analysis
        """
        # Jitter/shimmer analysis
        js_results = self.jitter_shimmer_analyzer(audio_waveform)
        
        # Emotional state detection
        emotion_results = self.emotional_detector(js_results['features'])
        
        # Formant analysis
        formant_results = self.formant_analyzer(audio_waveform)
        
        # Fuse all features
        combined_features = torch.cat([
            js_results['features'],
            emotion_results['features'],
            formant_results['encoded_features']
        ], dim=1)
        
        # Overall voice stress/fakeness score
        fakeness_score = self.fusion(combined_features)
        
        return {
            'jitter': js_results['jitter'],
            'shimmer': js_results['shimmer'],
            'hnr': js_results['hnr'],
            'stress_score': js_results['stress_score'],
            'emotions': {
                'stress': emotion_results['stress'],
                'anxiety': emotion_results['anxiety'],
                'fear': emotion_results['fear'],
                'anger': emotion_results['anger'],
                'distribution': emotion_results['emotion_distribution']
            },
            'formants': formant_results['formant_features'],
            'fakeness_score': fakeness_score,
            'features': combined_features
        }


if __name__ == "__main__":
    """Test voice stress analyzer."""
    print("🎤 Voice Stress Analyzer Test")
    
    # Create test audio (1 second, 16kHz)
    sample_rate = 16000
    duration = 1.0
    batch_size = 2
    
    # Generate test waveform (440 Hz sine wave)
    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).repeat(batch_size, 1)
    
    # Add some noise to make it realistic
    test_audio = test_audio + torch.randn_like(test_audio) * 0.1
    
    # Create analyzer
    analyzer = VoiceStressAnalyzer(sample_rate=sample_rate)
    
    # Analyze
    with torch.no_grad():
        results = analyzer(test_audio)
        
        print(f"\n✅ Analysis Results:")
        print(f"   Jitter: {results['jitter'].mean():.2f}%")
        print(f"   Shimmer: {results['shimmer'].mean():.2f}%")
        print(f"   HNR: {results['hnr'].mean():.2f} dB")
        print(f"   Stress Score: {results['stress_score'].mean():.3f}")
        print(f"   Fakeness Score: {results['fakeness_score'].mean():.3f}")
        print(f"   Emotions:")
        print(f"      Stress: {results['emotions']['stress'].mean():.3f}")
        print(f"      Anxiety: {results['emotions']['anxiety'].mean():.3f}")
        print(f"      Fear: {results['emotions']['fear'].mean():.3f}")
        print(f"      Anger: {results['emotions']['anger'].mean():.3f}")
        
    print("\n✅ Voice stress analyzer test completed!")
