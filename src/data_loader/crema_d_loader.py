"""CREMA-D dataset loader."""

import pandas as pd
import zipfile
from pathlib import Path
from .base_loader import BaseDatasetLoader


def load_crema_d(data_dir: str = "./data/crema_d", download: bool = True) -> pd.DataFrame:
    """
    Load CREMA-D dataset.
    
    CREMA-D contains emotional speech with labels: happy, sad, angry, fearful,
    disgust, neutral. Filenames encode actor ID and emotion.
    
    Args:
        data_dir: Directory to store/load CREMA-D data
        download: Whether to download if not present
        
    Returns:
        DataFrame with columns: ['file_path', 'label', 'speaker_id', 'dataset']
    """
    loader = CREMADLoader(data_dir)
    if download:
        loader.download()
    return loader.load()


class CREMADLoader(BaseDatasetLoader):
    """Loader for CREMA-D dataset."""
    
    def __init__(self, data_dir: str = "./data/crema_d"):
        """Initialize CREMA-D loader."""
        super().__init__(data_dir)
        self.dataset_name = "CREMA-D"
    
    def download(self) -> None:
        """Download CREMA-D dataset if not present."""
        zip_path = self.data_dir / "AudioWAV.zip"
        extracted_dir = self.data_dir / "AudioWAV"
        
        if extracted_dir.exists() and any(extracted_dir.iterdir()):
            print(f"CREMA-D dataset already exists at {extracted_dir}")
            return
        
        if not zip_path.exists():
            print(f"CREMA-D dataset not found at {zip_path}")
            print("Note: CREMA-D requires manual download from:")
            print("https://github.com/CheyneyComputerScience/CREMA-D")
            print("Please download and place AudioWAV.zip in the data directory.")
            return
        
        print(f"Extracting CREMA-D dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        print("CREMA-D dataset extracted successfully.")
    
    def load(self) -> pd.DataFrame:
        """
        Load CREMA-D dataset metadata.
        
        Filename format: 1001_DFA_ANG_XX.wav
        Format: ActorID_Sentence_Emotion_Intensity
        
        Returns:
            DataFrame with file paths and labels
        """
        audio_dir = self.data_dir / "AudioWAV"
        
        if not audio_dir.exists():
            raise FileNotFoundError(
                f"CREMA-D audio directory not found: {audio_dir}. "
                "Please download the dataset first."
            )
        
        records = []
        emotion_map = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fearful",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad"
        }
        
        for audio_file in audio_dir.glob("*.wav"):
            filename = audio_file.stem
            parts = filename.split("_")
            
            if len(parts) >= 3:
                actor_id = parts[0]
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code, "unknown")
                
                if self._validate_file(audio_file):
                    records.append({
                        "file_path": self._standardize_path(audio_file),
                        "label": emotion,
                        "speaker_id": f"CREMA-D_{actor_id}",
                        "dataset": "CREMA-D"
                    })
        
        if not records:
            raise ValueError(f"No valid audio files found in {audio_dir}")
        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} files from CREMA-D dataset")
        return df

