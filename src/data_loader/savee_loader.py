"""SAVEE dataset loader."""

import pandas as pd
import zipfile
from pathlib import Path
from .base_loader import BaseDatasetLoader


def load_savee(data_dir: str = "./data/savee", download: bool = True) -> pd.DataFrame:
    """
    Load SAVEE dataset.
    
    SAVEE contains emotional speech with labels: happy, sad, angry, fearful,
    disgust, neutral, surprise. Filenames encode emotion and sentence.
    
    Args:
        data_dir: Directory to store/load SAVEE data
        download: Whether to download if not present
        
    Returns:
        DataFrame with columns: ['file_path', 'label', 'speaker_id', 'dataset']
    """
    loader = SAVEELoader(data_dir)
    if download:
        loader.download()
    return loader.load()


class SAVEELoader(BaseDatasetLoader):
    """Loader for SAVEE dataset."""
    
    def __init__(self, data_dir: str = "./data/savee"):
        """Initialize SAVEE loader."""
        super().__init__(data_dir)
        self.dataset_name = "SAVEE"
    
    def download(self) -> None:
        """Download SAVEE dataset if not present."""
        zip_path = self.data_dir / "AudioData.zip"
        extracted_dir = self.data_dir / "AudioData"
        
        if extracted_dir.exists() and any(extracted_dir.iterdir()):
            print(f"SAVEE dataset already exists at {extracted_dir}")
            return
        
        if not zip_path.exists():
            print(f"SAVEE dataset not found at {zip_path}")
            print("Note: SAVEE requires manual download from:")
            print("http://kahlan.eps.surrey.ac.uk/savee/Download.html")
            print("Please download and place AudioData.zip in the data directory.")
            return
        
        print(f"Extracting SAVEE dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        print("SAVEE dataset extracted successfully.")
    
    def load(self) -> pd.DataFrame:
        """
        Load SAVEE dataset metadata.
        
        Filename format: DC_a01.wav (Disgust, Channel, sentence)
        Format: Emotion_Channel_Sentence
        
        Returns:
            DataFrame with file paths and labels
        """
        audio_dir = self.data_dir / "AudioData"
        
        if not audio_dir.exists():
            raise FileNotFoundError(
                f"SAVEE audio directory not found: {audio_dir}. "
                "Please download the dataset first."
            )
        
        records = []
        emotion_map = {
            "a": "angry",
            "d": "disgust",
            "f": "fearful",
            "h": "happy",
            "n": "neutral",
            "sa": "sad",
            "su": "surprise"
        }
        
        for audio_file in audio_dir.glob("*.wav"):
            filename = audio_file.stem.upper()
            
            # SAVEE files have format: EMOTION_CHANNEL_SENTENCE
            # First letter(s) indicate emotion
            emotion_code = None
            for code, emotion in emotion_map.items():
                if filename.startswith(code.upper()):
                    emotion_code = code
                    break
            
            if emotion_code:
                emotion = emotion_map[emotion_code]
                # SAVEE has 4 speakers (DC, JE, JK, KL)
                speaker_id = "SAVEE_DC"  # Default, can be extracted from filename if needed
                
                if self._validate_file(audio_file):
                    records.append({
                        "file_path": self._standardize_path(audio_file),
                        "label": emotion,
                        "speaker_id": speaker_id,
                        "dataset": "SAVEE"
                    })
        
        if not records:
            raise ValueError(f"No valid audio files found in {audio_dir}")
        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} files from SAVEE dataset")
        return df

