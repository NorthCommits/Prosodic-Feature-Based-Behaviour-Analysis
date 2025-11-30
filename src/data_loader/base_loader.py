"""Base class for dataset loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Root directory containing the dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def download(self) -> None:
        """Download the dataset if not present."""
        pass
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load dataset metadata and return a DataFrame.
        
        Returns:
            DataFrame with columns: ['file_path', 'label', 'speaker_id', 'dataset']
        """
        pass
    
    def _standardize_path(self, file_path: Path) -> str:
        """
        Convert Path to standardized string path.
        
        Args:
            file_path: Path object
            
        Returns:
            String path
        """
        return str(file_path.resolve())
    
    def _validate_file(self, file_path: Path) -> bool:
        """
        Validate that audio file exists and is readable.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if file is valid, False otherwise
        """
        return file_path.exists() and file_path.is_file()

