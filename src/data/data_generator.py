"""
Market data generator for Market Master.
Generates realistic market data for all asset classes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random
import logging
from .asset_classes import AssetClass, get_asset_config, get_instrument_config

# Use basic logging instead of relative imports
logger = logging.getLogger(__name__)


class MarketDataGenerator:
    """Generates realistic market data for Market Master."""
    
    def __init__(self, asset_class: AssetClass, instrument: str, seed: Optional[int] = None):
        """
        Initialize the market data generator.
        
        Args:
            asset_class: Asset class to generate data for
            instrument: Specific instrument name
            seed: Random seed for reproducibility
        """
        self.asset_class = asset_class
        self.instrument = instrument
        self.config = get_instrument_config(asset_class, instrument)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        logger.info("MarketDataGenerator initialized", 
                   asset_class=asset_class.value,
                   instrument=instrument,
                   base_price=self.config['base_price'])
    
    def generate_tick_data(self, n_ticks: int = 10000, start_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate realistic tick data.
        
        Args:
            n_ticks: Number of ticks to generate
            start_time: Start time for data generation
            
        Returns:
            DataFrame with OHLCV market data
        """
        logger.info("Generating tick data", n_ticks=n_ticks, instrument=self.instrument)
        
        # Set start time
        if start_time is None:
            start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=start_time,
            periods=n_ticks,
            freq=f"{self.config['tick_interval']}s"
        )
        
        # Generate price data
        prices = self._generate_price_series(n_ticks)
        
        # Generate OHLC data
        ohlc_data = self._generate_ohlc_data(prices, n_ticks)
        
        # Generate volume data
        volumes = self._generate_volume_series(n_ticks, prices)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': ohlc_data['open'],
            'high': ohlc_data['high'],
            'low': ohlc_data['low'],
            'close': ohlc_data['close'],
            'volume': volumes,
            'asset_class': self.asset_class.value,
            'instrument': self.instrument
        })
        
        # Add market session info
        data['session'] = self._get_session_info(timestamps)
        data['volatility_regime'] = self._get_volatility_regime(n_ticks)
        
        logger.info("Tick data generated successfully", 
                   data_shape=data.shape,
                   price_range=(data['close'].min(), data['close'].max()))
        
        return data
    
    def generate_training_data(self, n_samples: int = 10000, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate training data with labels.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with features and labels
        """
        logger.info("Generating training data", n_samples=n_samples, instrument=self.instrument)
        
        # Generate market data
        market_data = self.generate_tick_data(n_samples, seed=seed)
        
        # Generate labels
        labels = self._generate_labels(market_data)
        
        # Add labels to data
        market_data['action'] = labels
        
        # Add technical indicators (simplified)
        market_data['sma_20'] = market_data['close'].rolling(window=20).mean()
        market_data['sma_50'] = market_data['close'].rolling(window=50).mean()
        market_data['rsi'] = self._calculate_rsi(market_data['close'])
        market_data['volatility'] = market_data['close'].rolling(window=20).std()
        
        # Drop NaN values
        market_data = market_data.dropna()
        
        logger.info("Training data generated successfully", 
                   data_shape=market_data.shape,
                   label_distribution=market_data['action'].value_counts().to_dict())
        
        return market_data
    
    def _generate_price_series(self, n_ticks: int) -> np.ndarray:
        """Generate realistic price series."""
        base_price = self.config['base_price']
        volatility = self.config['volatility']
        trend_strength = self.config['trend_strength']
        
        # Generate random walk with trend
        returns = np.random.normal(trend_strength, volatility, n_ticks)
        prices = base_price * np.exp(np.cumsum(returns))
        
        return prices
    
    def _generate_ohlc_data(self, prices: np.ndarray, n_ticks: int) -> Dict[str, np.ndarray]:
        """Generate OHLC data from price series."""
        # Simple OHLC generation
        opens = prices[:-1]
        closes = prices[1:]
        
        # Generate high and low
        volatility = self.config['volatility']
        price_range = volatility * prices[1:] * 0.5
        
        highs = closes + np.random.uniform(0, price_range)
        lows = closes - np.random.uniform(0, price_range)
        
        # Ensure high >= close and low <= close
        highs = np.maximum(highs, closes)
        lows = np.minimum(lows, closes)
        
        return {
            'open': np.concatenate([[prices[0]], opens]),
            'high': np.concatenate([[prices[0]], highs]),
            'low': np.concatenate([[prices[0]], lows]),
            'close': prices
        }
    
    def _generate_volume_series(self, n_ticks: int, prices: np.ndarray) -> np.ndarray:
        """Generate realistic volume series."""
        base_volume = self.config['volume_base']
        volume_volatility = self.config['volume_volatility']
        
        # Generate base volume with some randomness
        volumes = np.random.lognormal(
            mean=np.log(base_volume),
            sigma=volume_volatility,
            size=n_ticks
        )
        
        # Add price-volume correlation
        price_changes = np.abs(np.diff(prices, prepend=prices[0]))
        volume_multiplier = 1 + price_changes / prices * 10
        volumes = volumes * volume_multiplier
        
        return volumes.astype(int)
    
    def _generate_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate trading action labels.
        
        Returns:
            Array of labels: 0 (hold), 1 (buy), 2 (sell)
        """
        # Simple labeling based on price movement
        close_prices = data['close'].values
        returns = np.diff(close_prices, prepend=close_prices[0])
        
        # Calculate thresholds
        volatility = self.config['volatility']
        buy_threshold = volatility * 0.5
        sell_threshold = -volatility * 0.5
        
        # Generate labels
        labels = np.zeros(len(returns))
        labels[returns > buy_threshold] = 1  # Buy
        labels[returns < sell_threshold] = 2  # Sell
        
        return labels
    
    def _get_session_info(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Get market session information."""
        sessions = np.zeros(len(timestamps))
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            if 9 <= hour < 16:  # Regular market hours
                sessions[i] = 1
            elif 16 <= hour < 20:  # After hours
                sessions[i] = 2
            else:  # Pre-market
                sessions[i] = 0
        
        return sessions
    
    def _get_volatility_regime(self, n_ticks: int) -> np.ndarray:
        """Generate volatility regime information."""
        # Simple regime switching
        regimes = np.random.choice([0, 1, 2], size=n_ticks, p=[0.6, 0.3, 0.1])
        return regimes
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def validate_generated_data(self, data: pd.DataFrame) -> Dict[str, any]:
        """Validate generated data quality."""
        validation = {
            'total_rows': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'price_range': (data['close'].min(), data['close'].max()),
            'volume_range': (data['volume'].min(), data['volume'].max()),
            'unique_actions': data['action'].nunique() if 'action' in data.columns else 0
        }
        return validation


def generate_market_data(asset_class: AssetClass, instrument: str, n_ticks: int = 10000, 
                        seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate market data for a specific asset class and instrument.
    
    Args:
        asset_class: Asset class
        instrument: Instrument name
        n_ticks: Number of ticks to generate
        seed: Random seed
        
    Returns:
        DataFrame with market data
    """
    generator = MarketDataGenerator(asset_class, instrument, seed=seed)
    return generator.generate_tick_data(n_ticks)


def generate_training_data(asset_class: AssetClass, instrument: str, n_samples: int = 10000,
                          seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate training data for a specific asset class and instrument.
    
    Args:
        asset_class: Asset class
        instrument: Instrument name
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        DataFrame with features and labels
    """
    generator = MarketDataGenerator(asset_class, instrument, seed=seed)
    return generator.generate_training_data(n_samples)


def generate_multi_asset_data(asset_classes: List[AssetClass], n_ticks: int = 10000,
                             seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate data for multiple asset classes.
    
    Args:
        asset_classes: List of asset classes
        n_ticks: Number of ticks per asset class
        seed: Random seed
        
    Returns:
        Dictionary mapping asset class to DataFrame
    """
    data = {}
    
    for asset_class in asset_classes:
        config = get_asset_config(asset_class)
        for instrument in config.instruments[:2]:  # Use first 2 instruments per class
            key = f"{asset_class.value}_{instrument}"
            data[key] = generate_market_data(asset_class, instrument, n_ticks, seed)
    
    return data 