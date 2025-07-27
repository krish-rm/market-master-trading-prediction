"""
Helper functions for Market Master.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .logger import get_logger

logger = get_logger(__name__)


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 18+ technical indicators for market data.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators
    """
    logger.info("Calculating technical indicators", data_shape=data.shape)
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create copy to avoid modifying original data
    df = data.copy()
    
    # Price Action Indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(df['close'])
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # Volume Indicators
    df['obv'] = calculate_obv(df['close'], df['volume'])
    df['vwap'] = calculate_vwap(df['close'], df['volume'])
    
    # Momentum Indicators
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])
    df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'])
    df['cci'] = calculate_cci(df['high'], df['low'], df['close'])
    
    # Trend Indicators
    df['adx'] = calculate_adx(df['high'], df['low'], df['close'])
    df['supertrend'] = calculate_supertrend(df['high'], df['low'], df['close'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    
    # Volatility Indicators
    df['atr_ratio'] = df['atr'] / df['close']
    
    # Price Change Indicators
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Support/Resistance (simplified)
    df['pivot_point'] = calculate_pivot_point(df['high'], df['low'], df['close'])
    
    logger.info("Technical indicators calculated successfully", indicators_count=len(df.columns))
    return df


def validate_market_data(data: pd.DataFrame) -> Dict[str, any]:
    """
    Validate market data quality.
    
    Args:
        data: DataFrame with market data
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating market data", data_shape=data.shape)
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'data_quality_score': 1.0,
        'missing_values': {},
        'outliers': {},
        'data_types': {}
    }
    
    # Check for missing values
    missing_values = data.isnull().sum()
    validation_results['missing_values'] = missing_values.to_dict()
    
    if missing_values.sum() > 0:
        validation_results['issues'].append(f"Found {missing_values.sum()} missing values")
        validation_results['data_quality_score'] -= 0.1
    
    # Check data types
    validation_results['data_types'] = data.dtypes.to_dict()
    
    # Check for outliers in price data
    if 'close' in data.columns:
        q1 = data['close'].quantile(0.25)
        q3 = data['close'].quantile(0.75)
        iqr = q3 - q1
        outliers = data[(data['close'] < q1 - 1.5 * iqr) | (data['close'] > q3 + 1.5 * iqr)]
        
        if len(outliers) > 0:
            validation_results['issues'].append(f"Found {len(outliers)} price outliers")
            validation_results['outliers']['close'] = len(outliers)
            validation_results['data_quality_score'] -= 0.05
    
    # Check for negative prices
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in data.columns:
            negative_prices = (data[col] <= 0).sum()
            if negative_prices > 0:
                validation_results['issues'].append(f"Found {negative_prices} negative {col} prices")
                validation_results['data_quality_score'] -= 0.2
    
    # Check for negative volumes
    if 'volume' in data.columns:
        negative_volumes = (data['volume'] < 0).sum()
        if negative_volumes > 0:
            validation_results['issues'].append(f"Found {negative_volumes} negative volumes")
            validation_results['data_quality_score'] -= 0.2
    
    # Ensure data quality score is not negative
    validation_results['data_quality_score'] = max(0.0, validation_results['data_quality_score'])
    
    # Mark as invalid if too many issues
    if validation_results['data_quality_score'] < 0.5:
        validation_results['is_valid'] = False
    
    logger.info("Market data validation completed", 
                is_valid=validation_results['is_valid'],
                quality_score=validation_results['data_quality_score'])
    
    return validation_results


# Technical Indicator Calculation Functions

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def calculate_obv(prices: pd.Series, volumes: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = pd.Series(index=prices.index, dtype=float)
    obv.iloc[0] = volumes.iloc[0]
    
    for i in range(1, len(prices)):
        if prices.iloc[i] > prices.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
        elif prices.iloc[i] < prices.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_vwap(prices: pd.Series, volumes: pd.Series) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = prices  # Simplified - could use (high + low + close) / 3
    vwap = (typical_price * volumes).cumsum() / volumes.cumsum()
    return vwap


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=3).mean()
    return k_percent, d_percent


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - sma) / (0.015 * mad)
    return cci


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (simplified)."""
    # Simplified ADX calculation
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Simplified directional movement
    dm_plus = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
    dm_minus = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
    
    di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()
    
    return adx


def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Calculate SuperTrend indicator (simplified)."""
    atr = calculate_atr(high, low, close, period)
    
    # Basic upper and lower bands
    basic_upper = (high + low) / 2 + multiplier * atr
    basic_lower = (high + low) / 2 - multiplier * atr
    
    # Simplified SuperTrend
    supertrend = pd.Series(index=close.index, dtype=float)
    supertrend.iloc[0] = basic_lower.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = max(basic_lower.iloc[i], supertrend.iloc[i-1])
        else:
            supertrend.iloc[i] = basic_lower.iloc[i]
    
    return supertrend


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_pivot_point(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate Pivot Point (simplified)."""
    pivot = (high + low + close) / 3
    return pivot 