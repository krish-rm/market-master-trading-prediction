"""
Data storage utilities for Market Master.
Handles local storage of market data for all asset classes.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pickle
from ..config.settings import Settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataStorage:
    """Handles local data storage for Market Master."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize data storage.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.settings.data_storage_path,
            self.settings.raw_data_path,
            self.settings.processed_data_path,
            self.settings.model_data_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        logger.info("Data storage directories ensured", directories=directories)
    
    def save_market_data(self, 
                        data: pd.DataFrame, 
                        asset_class: str, 
                        instrument: str, 
                        data_type: str = "raw",
                        version: Optional[str] = None) -> str:
        """
        Save market data to local storage.
        
        Args:
            data: Market data DataFrame
            asset_class: Asset class (crypto, forex, equity, etc.)
            instrument: Instrument name (BTC/USD, EUR/USD, etc.)
            data_type: Type of data (raw, processed, training)
            version: Data version (if None, uses timestamp)
            
        Returns:
            Path to saved file
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create filename
        safe_instrument = instrument.replace("/", "_").replace(" ", "_")
        filename = f"{asset_class}_{safe_instrument}_{data_type}_{version}.parquet"
        
        # Determine storage path
        if data_type == "raw":
            storage_path = Path(self.settings.raw_data_path)
        elif data_type == "processed":
            storage_path = Path(self.settings.processed_data_path)
        else:
            storage_path = Path(self.settings.data_storage_path)
            
        # Create asset-specific subdirectory
        asset_path = storage_path / asset_class
        asset_path.mkdir(parents=True, exist_ok=True)
        
        # Save file
        filepath = asset_path / filename
        data.to_parquet(filepath, index=False)
        
        # Save metadata
        metadata = {
            "asset_class": asset_class,
            "instrument": instrument,
            "data_type": data_type,
            "version": version,
            "rows": len(data),
            "columns": list(data.columns),
            "created_at": datetime.now().isoformat(),
            "filepath": str(filepath)
        }
        
        metadata_path = filepath.with_suffix(".json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("Market data saved successfully", 
                   filepath=str(filepath),
                   rows=len(data),
                   asset_class=asset_class,
                   instrument=instrument)
                   
        return str(filepath)
    
    def load_market_data(self, 
                        asset_class: str, 
                        instrument: str, 
                        data_type: str = "raw",
                        version: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load market data from local storage.
        
        Args:
            asset_class: Asset class
            instrument: Instrument name
            data_type: Type of data
            version: Data version (if None, loads latest)
            
        Returns:
            Market data DataFrame or None if not found
        """
        # Determine storage path
        if data_type == "raw":
            storage_path = Path(self.settings.raw_data_path)
        elif data_type == "processed":
            storage_path = Path(self.settings.processed_data_path)
        else:
            storage_path = Path(self.settings.data_storage_path)
            
        asset_path = storage_path / asset_class
        
        if not asset_path.exists():
            logger.warning(f"Asset directory not found: {asset_path}")
            return None
            
        # Find matching files
        safe_instrument = instrument.replace("/", "_").replace(" ", "_")
        pattern = f"{asset_class}_{safe_instrument}_{data_type}_*.parquet"
        files = list(asset_path.glob(pattern))
        
        if not files:
            logger.warning(f"No data files found for pattern: {pattern}")
            return None
            
        # Select file based on version
        if version is None:
            # Load latest version
            filepath = max(files, key=lambda x: x.stat().st_mtime)
        else:
            # Load specific version
            target_filename = f"{asset_class}_{safe_instrument}_{data_type}_{version}.parquet"
            filepath = asset_path / target_filename
            if not filepath.exists():
                logger.warning(f"Specific version not found: {target_filename}")
                return None
                
        # Load data
        try:
            data = pd.read_parquet(filepath)
            logger.info("Market data loaded successfully", 
                       filepath=str(filepath),
                       rows=len(data))
            return data
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            return None
    
    def list_available_data(self, 
                           asset_class: Optional[str] = None,
                           data_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available data files.
        
        Args:
            asset_class: Filter by asset class
            data_type: Filter by data type
            
        Returns:
            Dictionary of available data files
        """
        result = {}
        
        # Check all storage paths
        storage_paths = [
            (self.settings.raw_data_path, "raw"),
            (self.settings.processed_data_path, "processed"),
            (self.settings.data_storage_path, "other")
        ]
        
        for storage_path, path_type in storage_paths:
            if not Path(storage_path).exists():
                continue
                
            for asset_dir in Path(storage_path).iterdir():
                if not asset_dir.is_dir():
                    continue
                    
                asset_name = asset_dir.name
                
                # Apply filters
                if asset_class and asset_name != asset_class:
                    continue
                    
                files = []
                for file_path in asset_dir.glob("*.parquet"):
                    if data_type and data_type not in file_path.name:
                        continue
                    files.append(file_path.name)
                    
                if files:
                    result[f"{asset_name}_{path_type}"] = sorted(files)
                    
        return result
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of stored data.
        
        Returns:
            Data summary dictionary
        """
        summary = {
            "total_files": 0,
            "total_rows": 0,
            "asset_classes": {},
            "data_types": {},
            "storage_usage": {}
        }
        
        # Check all storage paths
        storage_paths = [
            (self.settings.raw_data_path, "raw"),
            (self.settings.processed_data_path, "processed"),
            (self.settings.data_storage_path, "other")
        ]
        
        for storage_path, path_type in storage_paths:
            path_obj = Path(storage_path)
            if not path_obj.exists():
                continue
                
            summary["storage_usage"][path_type] = {
                "path": str(path_obj),
                "size_mb": sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file()) / (1024 * 1024)
            }
            
            for asset_dir in path_obj.iterdir():
                if not asset_dir.is_dir():
                    continue
                    
                asset_name = asset_dir.name
                
                if asset_name not in summary["asset_classes"]:
                    summary["asset_classes"][asset_name] = {
                        "files": 0,
                        "rows": 0,
                        "data_types": set()
                    }
                    
                for file_path in asset_dir.glob("*.parquet"):
                    try:
                        data = pd.read_parquet(file_path)
                        rows = len(data)
                        
                        summary["total_files"] += 1
                        summary["total_rows"] += rows
                        summary["asset_classes"][asset_name]["files"] += 1
                        summary["asset_classes"][asset_name]["rows"] += rows
                        
                        # Extract data type from filename
                        if "_raw_" in file_path.name:
                            data_type = "raw"
                        elif "_processed_" in file_path.name:
                            data_type = "processed"
                        elif "_training_" in file_path.name:
                            data_type = "training"
                        else:
                            data_type = "other"
                            
                        summary["asset_classes"][asset_name]["data_types"].add(data_type)
                        
                        if data_type not in summary["data_types"]:
                            summary["data_types"][data_type] = 0
                        summary["data_types"][data_type] += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to read file {file_path}: {e}")
                        
        # Convert sets to lists for JSON serialization
        for asset_info in summary["asset_classes"].values():
            asset_info["data_types"] = list(asset_info["data_types"])
            
        return summary
    
    def cleanup_old_data(self, 
                        days_to_keep: int = 30,
                        asset_class: Optional[str] = None) -> int:
        """
        Clean up old data files.
        
        Args:
            days_to_keep: Number of days to keep data
            asset_class: Specific asset class to clean (if None, cleans all)
            
        Returns:
            Number of files deleted
        """
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        deleted_count = 0
        
        storage_paths = [
            self.settings.raw_data_path,
            self.settings.processed_data_path,
            self.settings.data_storage_path
        ]
        
        for storage_path in storage_paths:
            path_obj = Path(storage_path)
            if not path_obj.exists():
                continue
                
            for asset_dir in path_obj.iterdir():
                if not asset_dir.is_dir():
                    continue
                    
                if asset_class and asset_dir.name != asset_class:
                    continue
                    
                for file_path in asset_dir.glob("*.parquet"):
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            # Also delete metadata file if it exists
                            metadata_path = file_path.with_suffix(".json")
                            if metadata_path.exists():
                                metadata_path.unlink()
                            deleted_count += 1
                            logger.info(f"Deleted old file: {file_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete file {file_path}: {e}")
                            
        logger.info(f"Cleanup completed: {deleted_count} files deleted")
        return deleted_count


def save_market_data(data: pd.DataFrame, 
                    asset_class: str, 
                    instrument: str, 
                    data_type: str = "raw",
                    version: Optional[str] = None,
                    settings: Optional[Settings] = None) -> str:
    """
    Convenience function to save market data.
    
    Args:
        data: Market data DataFrame
        asset_class: Asset class
        instrument: Instrument name
        data_type: Type of data
        version: Data version
        settings: Application settings
        
    Returns:
        Path to saved file
    """
    storage = DataStorage(settings)
    return storage.save_market_data(data, asset_class, instrument, data_type, version)


def load_market_data(asset_class: str, 
                    instrument: str, 
                    data_type: str = "raw",
                    version: Optional[str] = None,
                    settings: Optional[Settings] = None) -> Optional[pd.DataFrame]:
    """
    Convenience function to load market data.
    
    Args:
        asset_class: Asset class
        instrument: Instrument name
        data_type: Type of data
        version: Data version
        settings: Application settings
        
    Returns:
        Market data DataFrame or None
    """
    storage = DataStorage(settings)
    return storage.load_market_data(asset_class, instrument, data_type, version) 