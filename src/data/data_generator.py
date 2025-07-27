"""
Market data generator for Market Master.
Generates realistic market data for all asset classes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random
from .asset_classes import AssetClass, get_asset_config, get_instrument_config
from ..utils.logger import get_logger
from ..utils.helpers import calculate_technical_indicators, validate_market_data

logger = get_logger(__name__)


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
            n_samples: Number of training samples
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with features and labels
        """
        logger.info("Generating training data", n_samples=n_samples)
        
        # Generate market data
        market_data = self.generate_tick_data(n_samples)
        
        # Calculate technical indicators
        features = calculate_technical_indicators(market_data)
        
        # Generate labels based on price movements
        labels = self._generate_labels(market_data)
        
        # Combine features and labels
        training_data = features.copy()
        training_data['action'] = labels
        
        # Add metadata
        training_data['asset_class'] = self.asset_class.value
        training_data['instrument'] = self.instrument
        
        # Remove rows with NaN values (from technical indicators)
        training_data = training_data.dropna()
        
        logger.info("Training data generated successfully", 
                   final_samples=len(training_data),
                   label_distribution=training_data['action'].value_counts().to_dict())
        
        return training_data
    
    def _generate_price_series(self, n_ticks: int) -> np.ndarray:
        """Generate realistic price series."""
        base_price = self.config['base_price']
        volatility = self.config['volatility']
        trend_strength = self.config['trend_strength']
        
        # Generate random walk with trend
        returns = np.random.normal(trend_strength, volatility, n_ticks)
        
        # Add some mean reversion
        for i in range(1, n_ticks):
            if abs(returns[i]) > volatility * 2:
                returns[i] *= 0.5  # Reduce extreme moves
        
        # Add some momentum
        momentum_period = 20
        for i in range(momentum_period, n_ticks):
            recent_trend = np.mean(returns[i-momentum_period:i])
            returns[i] += recent_trend * 0.1
        
        # Convert to prices
        price_multipliers = np.exp(np.cumsum(returns))
        prices = base_price * price_multipliers
        
        return prices
    
    def _generate_ohlc_data(self, prices: np.ndarray, n_ticks: int) -> Dict[str, np.ndarray]:
        """Generate OHLC data from price series."""
        # Use close prices as base
        close = prices
        
        # Generate realistic OHLC relationships
        volatility = self.config['volatility']
        
        # High and low based on close
        high_low_range = np.random.uniform(0, volatility * 2, n_ticks)
        high = close + high_low_range * close
        low = close - high_low_range * close * 0.8
        
        # Open based on previous close with some gap
        open_prices = np.zeros_like(close)
        open_prices[0] = close[0]
        
        for i in range(1, n_ticks):
            gap = np.random.normal(0, volatility * 0.5)
            open_prices[i] = close[i-1] * (1 + gap)
            
            # Ensure OHLC relationships are valid
            high[i] = max(high[i], open_prices[i])
            low[i] = min(low[i], open_prices[i])
        
        return {
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close
        }
    
    def _generate_volume_series(self, n_ticks: int, prices: np.ndarray) -> np.ndarray:
        """Generate realistic volume data."""
        volume_base = self.config['volume_base']
        volume_volatility = self.config['volume_volatility']
        
        # Base volume with some randomness
        volumes = np.random.normal(volume_base, volume_base * volume_volatility, n_ticks)
        volumes = np.abs(volumes)  # Ensure positive
        
        # Volume tends to be higher during price moves
        price_changes = np.abs(np.diff(prices, prepend=prices[0]))
        volume_multiplier = 1 + price_changes / np.mean(price_changes) * 0.5
        volumes *= volume_multiplier
        
        # Add some intraday patterns
        for i in range(n_ticks):
            # Higher volume at market open/close
            if i < n_ticks * 0.1 or i > n_ticks * 0.9:
                volumes[i] *= 1.5
            
            # Lower volume during lunch hours (for equity markets)
            if self.asset_class == AssetClass.EQUITY:
                if 0.4 < i / n_ticks < 0.6:
                    volumes[i] *= 0.7
        
        return volumes.astype(int)
    
    def _generate_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Generate trading action labels based on price movements."""
        prices = data['close'].values
        volumes = data['volume'].values
        
        # Calculate some basic indicators for labeling
        price_changes = np.diff(prices, prepend=prices[0])
        volume_changes = np.diff(volumes, prepend=volumes[0])
        
        # Simple moving averages
        sma_short = pd.Series(prices).rolling(5).mean().values
        sma_long = pd.Series(prices).rolling(20).mean().values
        
        labels = []
        
        for i in range(len(prices)):
            if i < 20:  # Not enough data for indicators
                labels.append('hold')
                continue
            
            # Decision logic based on multiple factors
            price_trend = prices[i] > sma_short[i]
            long_trend = prices[i] > sma_long[i]
            price_momentum = price_changes[i] > 0
            volume_support = volume_changes[i] > 0
            
            # Combine signals
            bullish_signals = sum([price_trend, long_trend, price_momentum, volume_support])
            
            if bullish_signals >= 3:
                if price_changes[i] > np.std(price_changes) * 2:
                    labels.append('strong_buy')
                else:
                    labels.append('buy')
            elif bullish_signals <= 1:
                if price_changes[i] < -np.std(price_changes) * 2:
                    labels.append('strong_sell')
                else:
                    labels.append('sell')
            else:
                labels.append('hold')
        
        return np.array(labels)
    
    def _get_session_info(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Get market session information."""
        sessions = []
        
        for ts in timestamps:
            hour = ts.hour
            
            if self.asset_class == AssetClass.EQUITY or self.asset_class == AssetClass.INDICES:
                if 9 <= hour < 16:
                    sessions.append('regular')
                else:
                    sessions.append('closed')
            else:
                # 24/5 or 24/7 markets
                if self.asset_class == AssetClass.CRYPTO:
                    sessions.append('active')  # Always active
                else:
                    # 24/5 markets
                    if ts.weekday() < 5:  # Monday to Friday
                        sessions.append('active')
                    else:
                        sessions.append('closed')
        
        return np.array(sessions)
    
    def _get_volatility_regime(self, n_ticks: int) -> np.ndarray:
        """Generate volatility regime information."""
        # Simulate different volatility regimes
        regimes = []
        
        for i in range(n_ticks):
            # Higher volatility at market open/close
            if i < n_ticks * 0.1 or i > n_ticks * 0.9:
                regimes.append('high')
            elif i < n_ticks * 0.2 or i > n_ticks * 0.8:
                regimes.append('medium')
            else:
                regimes.append('low')
        
        return np.array(regimes)
    
    def validate_generated_data(self, data: pd.DataFrame) -> Dict[str, any]:
        """Validate generated data quality."""
        return validate_market_data(data)


# Convenience functions

def generate_market_data(asset_class: AssetClass, instrument: str, n_ticks: int = 10000, 
                        seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate market data for a specific asset and instrument.
    
    Args:
        asset_class: Asset class
        instrument: Instrument name
        n_ticks: Number of ticks
        seed: Random seed
        
    Returns:
        DataFrame with market data
    """
    generator = MarketDataGenerator(asset_class, instrument, seed=seed)
    return generator.generate_tick_data(n_ticks)


def generate_training_data(asset_class: AssetClass, instrument: str, n_samples: int = 10000,
                          seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate training data for a specific asset and instrument.
    
    Args:
        asset_class: Asset class
        instrument: Instrument name
        n_samples: Number of samples
        seed: Random seed
        
    Returns:
        DataFrame with training data
    """
    generator = MarketDataGenerator(asset_class, instrument, seed=seed)
    return generator.generate_training_data(n_samples)


def generate_multi_asset_data(asset_classes: List[AssetClass], n_ticks: int = 10000,
                             seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate market data for multiple asset classes.
    
    Args:
        asset_classes: List of asset classes
        n_ticks: Number of ticks per asset
        seed: Random seed
        
    Returns:
        Dictionary with data for each asset class
    """
    data = {}
    current_seed = seed
    
    for asset_class in asset_classes:
        config = get_asset_config(asset_class)
        instrument = config.instruments[0]  # Use first instrument
        
        generator = MarketDataGenerator(asset_class, instrument, seed=current_seed)
        data[asset_class.value] = generator.generate_tick_data(n_ticks)
        
        if current_seed is not None:
            current_seed += 1  # Different seed for each asset
    
    return data 