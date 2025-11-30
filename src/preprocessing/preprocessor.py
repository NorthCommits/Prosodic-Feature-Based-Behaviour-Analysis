"""Audio preprocessing pipeline for speech analysis."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
import logging
from scipy import signal
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing pipeline with resampling, normalization, VAD, and noise reduction."""
    
    def __init__(
        self,
        target_sr: int = 16000,
        normalize_loudness: bool = True,
        trim_silence: bool = True,
        vad_threshold: float = 0.01,
        noise_reduction: bool = True,
        max_length_seconds: float = 10.0,
        chunk_long_utterances: bool = False,
        chunk_size_seconds: float = 5.0
    ):
        """
        Initialize audio preprocessor.
        
        Args:
            target_sr: Target sample rate in Hz
            normalize_loudness: Whether to normalize loudness
            trim_silence: Whether to trim leading/trailing silence
            vad_threshold: Threshold for voice activity detection
            noise_reduction: Whether to apply noise reduction
            max_length_seconds: Maximum length in seconds (truncate if longer)
            chunk_long_utterances: Whether to chunk long utterances
            chunk_size_seconds: Size of chunks in seconds
        """
        self.target_sr = target_sr
        self.normalize_loudness = normalize_loudness
        self.trim_silence = trim_silence
        self.vad_threshold = vad_threshold
        self.noise_reduction = noise_reduction
        self.max_length_seconds = max_length_seconds
        self.chunk_long_utterances = chunk_long_utterances
        self.chunk_size_seconds = chunk_size_seconds
    
    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Process a single audio file through the preprocessing pipeline.
        
        Args:
            input_path: Path to input audio file
            output_path: Optional path to save processed audio
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=None, mono=True)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            # Apply preprocessing steps
            audio = self._trim_silence(audio, sr) if self.trim_silence else audio
            audio = self._normalize_loudness(audio) if self.normalize_loudness else audio
            audio = self._reduce_noise(audio, sr) if self.noise_reduction else audio
            audio = self._limit_length(audio, sr)
            
            # Save if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(output_path, audio, sr)
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            raise
    
    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Trim leading and trailing silence using energy-based VAD.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Trimmed audio signal
        """
        # Compute frame energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate RMS energy per frame
        frames = librosa.util.frame(
            audio,
            frame_length=frame_length,
            hop_length=hop_length,
            axis=0
        )
        frame_energy = np.mean(frames ** 2, axis=0)
        
        # Find frames above threshold
        threshold = np.max(frame_energy) * self.vad_threshold
        active_frames = frame_energy > threshold
        
        if not np.any(active_frames):
            return audio
        
        # Find first and last active frames
        active_indices = np.where(active_frames)[0]
        start_frame = max(0, active_indices[0] - 1)
        end_frame = min(len(active_frames), active_indices[-1] + 2)
        
        # Convert frames to samples
        start_sample = start_frame * hop_length
        end_sample = min(len(audio), end_frame * hop_length + frame_length)
        
        return audio[start_sample:end_sample]
    
    def _normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target loudness level.
        
        Args:
            audio: Audio signal
            
        Returns:
            Normalized audio signal
        """
        # Compute RMS and normalize to -20 dB
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (-20 / 20)  # -20 dB
            audio = audio * (target_rms / rms)
        
        # Clip to prevent clipping
        audio = np.clip(audio, -1.0, 1.0)
        return audio
    
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply simple noise reduction using spectral subtraction.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Denoised audio signal
        """
        # Simple high-pass filter to remove low-frequency noise
        nyquist = sr / 2
        cutoff = 80  # Hz
        b, a = signal.butter(4, cutoff / nyquist, btype='high')
        audio = signal.filtfilt(b, a, audio)
        
        # Spectral subtraction (simplified)
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 0.1 seconds
        noise_frames = int(0.1 * sr / 512)
        noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Subtract noise estimate with over-subtraction factor
        alpha = 2.0
        enhanced_magnitude = magnitude - alpha * noise_estimate
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        audio = librosa.istft(enhanced_stft, hop_length=512)
        
        return audio
    
    def _limit_length(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Limit audio length to maximum duration.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Truncated audio signal
        """
        max_samples = int(self.max_length_seconds * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        return audio
    
    def process_batch(
        self,
        file_paths: list,
        output_dir: Optional[str] = None
    ) -> list:
        """
        Process multiple audio files.
        
        Args:
            file_paths: List of input file paths
            output_dir: Optional directory to save processed files
            
        Returns:
            List of processed audio arrays
        """
        processed_audios = []
        
        for file_path in file_paths:
            try:
                output_path = None
                if output_dir:
                    input_path = Path(file_path)
                    output_path = Path(output_dir) / input_path.name
                
                audio, sr = self.process_file(file_path, output_path)
                processed_audios.append((audio, sr))
                
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {str(e)}")
                processed_audios.append(None)
        
        return processed_audios

