"""Prosodic and spectral feature extraction using Praat (parselmouth) and librosa."""

import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import pandas as pd

# Try to import parselmouth, make it optional
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Track if we've already warned about parselmouth
_PARSELMOUTH_WARNED = False

def _warn_parselmouth_once():
    """Warn about parselmouth only once."""
    global _PARSELMOUTH_WARNED
    if not PARSELMOUTH_AVAILABLE and not _PARSELMOUTH_WARNED:
        logger.warning(
            "Parselmouth not available. Using librosa-based F0 extraction. "
            "Install with: pip install parselmouth --no-deps"
        )
        _PARSELMOUTH_WARNED = True


class ProsodicFeatureExtractor:
    """Extract prosodic and spectral features from audio files."""
    
    def __init__(
        self,
        extract_prosodic: bool = True,
        extract_spectral: bool = True,
        extract_mfcc: bool = True,
        extract_formants: bool = True,
        mfcc_n_coeffs: int = 13,
        frame_length: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize feature extractor.
        
        Args:
            extract_prosodic: Extract prosodic features (F0, jitter, shimmer, etc.)
            extract_spectral: Extract spectral features
            extract_mfcc: Extract MFCC features
            extract_formants: Extract formant features
            mfcc_n_coeffs: Number of MFCC coefficients
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
        """
        self.extract_prosodic = extract_prosodic
        self.extract_spectral = extract_spectral
        self.extract_mfcc = extract_mfcc
        self.extract_formants = extract_formants
        self.mfcc_n_coeffs = mfcc_n_coeffs
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def extract_from_file(self, audio_path: str) -> Dict[str, float]:
        """
        Extract features from a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of feature names and values
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Extract features
            features = {}
            
            if self.extract_prosodic:
                prosodic_features = self._extract_prosodic_features(audio, sr)
                features.update(prosodic_features)
            
            if self.extract_spectral or self.extract_mfcc:
                spectral_features = self._extract_spectral_features(audio, sr)
                features.update(spectral_features)
            
            if self.extract_formants:
                formant_features = self._extract_formant_features(audio, sr)
                features.update(formant_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {str(e)}")
            return {}
    
    def _extract_prosodic_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """
        Extract prosodic features using Praat.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of prosodic features
        """
        features = {}
        
        if not PARSELMOUTH_AVAILABLE:
            _warn_parselmouth_once()
            return self._extract_prosodic_features_librosa(audio, sr)
        
        try:
            # Convert to Praat Sound object
            sound = parselmouth.Sound(audio, sampling_frequency=sr)
            
            # Extract pitch (F0)
            pitch = sound.to_pitch()
            f0_values = pitch.selected_array['frequency']
            f0_values = f0_values[f0_values > 0]  # Remove unvoiced frames
            
            if len(f0_values) > 0:
                features['f0_mean'] = np.mean(f0_values)
                features['f0_median'] = np.median(f0_values)
                features['f0_std'] = np.std(f0_values)
                features['f0_min'] = np.min(f0_values)
                features['f0_max'] = np.max(f0_values)
                features['f0_range'] = features['f0_max'] - features['f0_min']
            else:
                features.update({
                    'f0_mean': 0.0, 'f0_median': 0.0, 'f0_std': 0.0,
                    'f0_min': 0.0, 'f0_max': 0.0, 'f0_range': 0.0
                })
            
            # Extract jitter and shimmer
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_local_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            
            features['jitter_local'] = jitter_local
            features['jitter_local_abs'] = jitter_local_abs
            features['jitter_rap'] = jitter_rap
            features['jitter_ppq5'] = jitter_ppq5
            
            # Shimmer
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_local_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            features['shimmer_local'] = shimmer_local
            features['shimmer_local_db'] = shimmer_local_db
            features['shimmer_apq3'] = shimmer_apq3
            features['shimmer_apq5'] = shimmer_apq5
            
            # Voicing ratio
            total_frames = len(f0_values)
            voiced_frames = len(f0_values)
            if total_frames > 0:
                features['voicing_ratio'] = voiced_frames / total_frames
            else:
                features['voicing_ratio'] = 0.0
            
            # Intensity
            intensity = sound.to_intensity()
            intensity_values = intensity.values[0]
            intensity_values = intensity_values[intensity_values > 0]
            
            if len(intensity_values) > 0:
                features['intensity_mean'] = np.mean(intensity_values)
                features['intensity_std'] = np.std(intensity_values)
            else:
                features['intensity_mean'] = 0.0
                features['intensity_std'] = 0.0
            
            # Speaking rate (approximate using voiced frames)
            duration = len(audio) / sr
            if duration > 0:
                features['speaking_rate'] = voiced_frames / duration
            else:
                features['speaking_rate'] = 0.0
            
            # Pause detection (simplified)
            pause_count, avg_pause_duration = self._detect_pauses(audio, sr)
            features['pause_count'] = pause_count
            features['avg_pause_duration'] = avg_pause_duration
            
        except Exception as e:
            logger.warning(f"Error in prosodic feature extraction: {str(e)}")
            # Return default values
            features.update({
                'f0_mean': 0.0, 'f0_median': 0.0, 'f0_std': 0.0,
                'f0_min': 0.0, 'f0_max': 0.0, 'f0_range': 0.0,
                'jitter_local': 0.0, 'jitter_local_abs': 0.0,
                'jitter_rap': 0.0, 'jitter_ppq5': 0.0,
                'shimmer_local': 0.0, 'shimmer_local_db': 0.0,
                'shimmer_apq3': 0.0, 'shimmer_apq5': 0.0,
                'voicing_ratio': 0.0,
                'intensity_mean': 0.0, 'intensity_std': 0.0,
                'speaking_rate': 0.0,
                'pause_count': 0.0, 'avg_pause_duration': 0.0
            })
        
        return features
    
    def _extract_prosodic_features_librosa(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """
        Extract prosodic features using librosa (fallback when Praat is not available).
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of prosodic features
        """
        features = {}
        
        # Extract F0 using librosa's pyin algorithm
        f0_values, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # Remove NaN values (unvoiced frames)
        f0_values = f0_values[~np.isnan(f0_values)]
        
        if len(f0_values) > 0:
            features['f0_mean'] = float(np.mean(f0_values))
            features['f0_median'] = float(np.median(f0_values))
            features['f0_std'] = float(np.std(f0_values))
            features['f0_min'] = float(np.min(f0_values))
            features['f0_max'] = float(np.max(f0_values))
            features['f0_range'] = features['f0_max'] - features['f0_min']
        else:
            features.update({
                'f0_mean': 0.0, 'f0_median': 0.0, 'f0_std': 0.0,
                'f0_min': 0.0, 'f0_max': 0.0, 'f0_range': 0.0
            })
        
        # Jitter and shimmer approximations using librosa
        if len(f0_values) > 1:
            # Jitter: variation in F0 period
            periods = 1.0 / f0_values
            period_diffs = np.diff(periods)
            features['jitter_local'] = float(np.mean(np.abs(period_diffs)) / np.mean(periods[1:]))
            features['jitter_local_abs'] = float(np.mean(np.abs(period_diffs)))
            features['jitter_rap'] = features['jitter_local']  # Approximation
            features['jitter_ppq5'] = features['jitter_local']  # Approximation
        else:
            features.update({
                'jitter_local': 0.0, 'jitter_local_abs': 0.0,
                'jitter_rap': 0.0, 'jitter_ppq5': 0.0
            })
        
        # Shimmer: amplitude variation
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        if len(rms) > 1:
            rms_diffs = np.diff(rms)
            features['shimmer_local'] = float(np.mean(np.abs(rms_diffs)) / np.mean(rms[1:]))
            features['shimmer_local_db'] = float(20 * np.log10(features['shimmer_local'] + 1e-10))
            features['shimmer_apq3'] = features['shimmer_local']
            features['shimmer_apq5'] = features['shimmer_local']
        else:
            features.update({
                'shimmer_local': 0.0, 'shimmer_local_db': 0.0,
                'shimmer_apq3': 0.0, 'shimmer_apq5': 0.0
            })
        
        # Voicing ratio
        if voiced_flag is not None:
            features['voicing_ratio'] = float(np.sum(voiced_flag) / len(voiced_flag))
        else:
            features['voicing_ratio'] = 0.0 if len(f0_values) == 0 else 1.0
        
        # Intensity
        if len(rms) > 0:
            features['intensity_mean'] = float(np.mean(rms))
            features['intensity_std'] = float(np.std(rms))
        else:
            features['intensity_mean'] = 0.0
            features['intensity_std'] = 0.0
        
        # Speaking rate
        duration = len(audio) / sr
        if duration > 0:
            voiced_frames = len(f0_values)
            features['speaking_rate'] = float(voiced_frames / duration)
        else:
            features['speaking_rate'] = 0.0
        
        # Pause detection
        pause_count, avg_pause_duration = self._detect_pauses(audio, sr)
        features['pause_count'] = pause_count
        features['avg_pause_duration'] = avg_pause_duration
        
        return features
    
    def _extract_spectral_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """
        Extract spectral features using librosa.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        # MFCC features
        if self.extract_mfcc:
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.mfcc_n_coeffs,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )
            
            # Mean of each MFCC coefficient
            for i in range(self.mfcc_n_coeffs):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            
            # Delta MFCCs
            mfcc_deltas = librosa.feature.delta(mfccs)
            for i in range(self.mfcc_n_coeffs):
                features[f'mfcc_{i+1}_delta_mean'] = np.mean(mfcc_deltas[i])
        
        return features
    
    def _extract_formant_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """
        Extract formant features using Praat.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of formant features
        """
        features = {}
        
        if not PARSELMOUTH_AVAILABLE:
            _warn_parselmouth_once()
            features['formant_f1_mean'] = 0.0
            features['formant_f2_mean'] = 0.0
            return features
        
        try:
            sound = parselmouth.Sound(audio, sampling_frequency=sr)
            formant = sound.to_formant_burg(time_step=0.01)
            
            # Extract F1 and F2 at midpoint
            midpoint = sound.duration / 2
            f1 = call(formant, "Get value at time", 1, midpoint, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, midpoint, "Hertz", "Linear")
            
            features['formant_f1_mean'] = f1 if not np.isnan(f1) else 0.0
            features['formant_f2_mean'] = f2 if not np.isnan(f2) else 0.0
            
        except Exception as e:
            logger.warning(f"Error in formant extraction: {str(e)}")
            features['formant_f1_mean'] = 0.0
            features['formant_f2_mean'] = 0.0
        
        return features
    
    def _detect_pauses(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[float, float]:
        """
        Detect pauses in audio signal.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (pause_count, avg_pause_duration)
        """
        # Compute frame energy
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        frames = librosa.util.frame(
            audio,
            frame_length=frame_length,
            hop_length=hop_length,
            axis=0
        )
        frame_energy = np.mean(frames ** 2, axis=0)
        
        # Threshold for silence
        threshold = np.percentile(frame_energy, 10)
        is_silence = frame_energy < threshold
        
        # Count pauses (transitions from speech to silence)
        pause_count = 0
        pause_durations = []
        in_pause = False
        pause_start = 0
        
        for i, silent in enumerate(is_silence):
            if silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not silent and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * hop_length / sr
                if pause_duration > 0.1:  # Only count pauses > 100ms
                    pause_count += 1
                    pause_durations.append(pause_duration)
        
        avg_pause_duration = np.mean(pause_durations) if pause_durations else 0.0
        
        return pause_count, avg_pause_duration
    
    def extract_batch(
        self,
        file_paths: list,
        labels: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Extract features from multiple files.
        
        Args:
            file_paths: List of audio file paths
            labels: Optional list of labels
            
        Returns:
            DataFrame with features and labels
        """
        feature_dicts = []
        
        for i, file_path in enumerate(file_paths):
            features = self.extract_from_file(file_path)
            if labels is not None and i < len(labels):
                features['label'] = labels[i]
            features['file_path'] = file_path
            feature_dicts.append(features)
        
        df = pd.DataFrame(feature_dicts)
        return df

