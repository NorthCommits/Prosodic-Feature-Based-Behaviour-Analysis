"""RAVDESS dataset loader."""

import pandas as pd
import zipfile
import requests
from pathlib import Path
from typing import Optional
import os
from .base_loader import BaseDatasetLoader


def load_ravdess(data_dir: str = "./data/ravdess", download: bool = True) -> pd.DataFrame:
    """
    Load RAVDESS dataset.
    
    RAVDESS contains emotional speech with labels: neutral, calm, happy, sad, angry,
    fearful, surprise, disgust. Each file is named with metadata encoded in filename.
    
    Args:
        data_dir: Directory to store/load RAVDESS data
        download: Whether to download if not present
        
    Returns:
        DataFrame with columns: ['file_path', 'label', 'speaker_id', 'dataset']
    """
    loader = RAVDESSLoader(data_dir)
    if download:
        loader.download()
    return loader.load()


class RAVDESSLoader(BaseDatasetLoader):
    """Loader for RAVDESS dataset."""
    
    RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    
    def __init__(self, data_dir: str = "./data/ravdess"):
        """Initialize RAVDESS loader."""
        super().__init__(data_dir)
        self.dataset_name = "RAVDESS"
    
    def download(self) -> None:
        """Download RAVDESS dataset if not present."""
        # Check for both Speech and Song versions
        speech_zip = self.data_dir / "Audio_Speech_Actors_01-24.zip"
        song_zip = self.data_dir / "Audio_Song_Actors_01-24.zip"
        speech_dir = self.data_dir / "Audio_Speech_Actors_01-24"
        song_dir = self.data_dir / "Audio_Song_Actors_01-24"
        
        # Check if already extracted
        if speech_dir.exists() and any(speech_dir.iterdir()):
            print(f"RAVDESS Speech dataset already exists at {speech_dir}")
            return
        if song_dir.exists() and any(song_dir.iterdir()):
            print(f"RAVDESS Song dataset already exists at {song_dir}")
            return
        
        # Find which zip file exists
        zip_path = None
        if speech_zip.exists():
            zip_path = speech_zip
        elif song_zip.exists():
            zip_path = song_zip
        
        if not zip_path:
            print(f"RAVDESS dataset not found.")
            print("Note: RAVDESS requires manual download from:")
            print("https://zenodo.org/record/1188976")
            print("Please download and place Audio_Speech_Actors_01-24.zip or Audio_Song_Actors_01-24.zip in the data directory.")
            return
        
        print(f"Extracting RAVDESS dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        print("RAVDESS dataset extracted successfully.")
    
    def load(self) -> pd.DataFrame:
        """
        Load RAVDESS dataset metadata.
        
        Filename format: 03-01-01-01-01-01-01.wav
        Format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
        
        Returns:
            DataFrame with file paths and labels
        """
        # Check for both Speech and Song versions
        speech_dir = self.data_dir / "Audio_Speech_Actors_01-24"
        song_dir = self.data_dir / "Audio_Song_Actors_01-24"
        
        # Check if Actor directories are directly in data_dir (Song version structure)
        has_actor_dirs = any(
            (self.data_dir / f"Actor_{i:02d}").exists() 
            for i in range(1, 25)
        )
        
        if speech_dir.exists():
            base_dir = speech_dir
        elif song_dir.exists():
            base_dir = song_dir
        elif has_actor_dirs:
            # Song version extracted directly to data_dir
            base_dir = self.data_dir
        else:
            raise FileNotFoundError(
                f"RAVDESS audio directory not found. "
                "Please download and extract Audio_Speech_Actors_01-24.zip or Audio_Song_Actors_01-24.zip first."
            )
        
        records = []
        emotion_map = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
        
        # Look for Actor_XX directories
        for actor_dir in sorted(base_dir.iterdir()):
            if not actor_dir.is_dir():
                continue
            
            # Check if it's an Actor directory (Actor_01, Actor_02, etc.)
            if not actor_dir.name.startswith("Actor_"):
                continue
            
            for audio_file in actor_dir.glob("*.wav"):
                filename = audio_file.stem
                parts = filename.split("-")
                
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    actor_id = parts[6] if len(parts) > 6 else "unknown"
                    emotion = emotion_map.get(emotion_code, "unknown")
                    
                    if self._validate_file(audio_file):
                        records.append({
                            "file_path": self._standardize_path(audio_file),
                            "label": emotion,
                            "speaker_id": f"RAVDESS_{actor_id}",
                            "dataset": "RAVDESS"
                        })
        
        if not records:
            raise ValueError(f"No valid audio files found in {base_dir}")
        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} files from RAVDESS dataset")
        return df

