"""
Asset class configurations for Market Master.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import numpy as np
import logging

# Use basic logging instead of relative import
logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Supported asset classes."""
    EQUITY = "equity"
    COMMODITY = "commodity"
    FOREX = "forex"
    CRYPTO = "crypto"
    INDICES = "indices"


@dataclass
class AssetConfig:
    """Configuration for an asset class."""
    name: str
    base_price: float
    volatility: float
    volume_base: int
    volume_volatility: float
    trend_strength: float
    session_hours: int
    tick_interval: int
    instruments: List[str]
    characteristics: Dict[str, any]


def get_asset_config(asset_class: AssetClass) -> AssetConfig:
    """
    Get configuration for a specific asset class.
    
    Args:
        asset_class: Asset class enum
        
    Returns:
        Asset configuration
    """
    configs = {
        AssetClass.EQUITY: AssetConfig(
            name="Equity",
            base_price=150.0,
            volatility=0.02,
            volume_base=1000000,
            volume_volatility=0.3,
            trend_strength=0.001,
            session_hours=6.5,
            tick_interval=1,
            instruments=["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"],
            characteristics={
                "market_hours": "9:30-16:00",
                "liquidity": "high",
                "correlation": "medium",
                "news_sensitivity": "high"
            }
        ),
        AssetClass.COMMODITY: AssetConfig(
            name="Commodity",
            base_price=1850.0,  # Gold
            volatility=0.015,
            volume_base=500000,
            volume_volatility=0.4,
            trend_strength=0.002,
            session_hours=24,
            tick_interval=1,
            instruments=["GOLD", "SILVER", "OIL", "COPPER", "PLATINUM"],
            characteristics={
                "market_hours": "24/5",
                "liquidity": "medium",
                "correlation": "low",
                "news_sensitivity": "medium"
            }
        ),
        AssetClass.FOREX: AssetConfig(
            name="Forex",
            base_price=1.2000,  # EUR/USD
            volatility=0.008,
            volume_base=2000000,
            volume_volatility=0.2,
            trend_strength=0.0005,
            session_hours=24,
            tick_interval=1,
            instruments=["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD"],
            characteristics={
                "market_hours": "24/5",
                "liquidity": "very_high",
                "correlation": "high",
                "news_sensitivity": "very_high"
            }
        ),
        AssetClass.CRYPTO: AssetConfig(
            name="Crypto",
            base_price=45000.0,  # Bitcoin
            volatility=0.04,
            volume_base=300000,
            volume_volatility=0.6,
            trend_strength=0.003,
            session_hours=24,
            tick_interval=1,
            instruments=["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"],
            characteristics={
                "market_hours": "24/7",
                "liquidity": "medium",
                "correlation": "medium",
                "news_sensitivity": "very_high"
            }
        ),
        AssetClass.INDICES: AssetConfig(
            name="Indices",
            base_price=4500.0,  # S&P 500
            volatility=0.012,
            volume_base=800000,
            volume_volatility=0.25,
            trend_strength=0.0008,
            session_hours=6.5,
            tick_interval=1,
            instruments=["SPX", "NDX", "DJI", "RUT", "VIX"],
            characteristics={
                "market_hours": "9:30-16:00",
                "liquidity": "high",
                "correlation": "high",
                "news_sensitivity": "high"
            }
        )
    }
    
    return configs[asset_class]


def get_all_asset_configs() -> Dict[AssetClass, AssetConfig]:
    """Get all asset configurations."""
    return {
        AssetClass.EQUITY: get_asset_config(AssetClass.EQUITY),
        AssetClass.COMMODITY: get_asset_config(AssetClass.COMMODITY),
        AssetClass.FOREX: get_asset_config(AssetClass.FOREX),
        AssetClass.CRYPTO: get_asset_config(AssetClass.CRYPTO),
        AssetClass.INDICES: get_asset_config(AssetClass.INDICES)
    }


def get_instrument_config(asset_class: AssetClass, instrument: str) -> Dict[str, any]:
    """
    Get specific instrument configuration.
    
    Args:
        asset_class: Asset class
        instrument: Instrument name
        
    Returns:
        Instrument configuration
    """
    asset_config = get_asset_config(asset_class)
    
    # Base configuration from asset class
    config = {
        "base_price": asset_config.base_price,
        "volatility": asset_config.volatility,
        "volume_base": asset_config.volume_base,
        "volume_volatility": asset_config.volume_volatility,
        "trend_strength": asset_config.trend_strength,
        "session_hours": asset_config.session_hours,
        "tick_interval": asset_config.tick_interval,
        "asset_class": asset_class.value,
        "instrument": instrument
    }
    
    # Instrument-specific adjustments
    instrument_adjustments = {
        "AAPL": {"base_price": 150.0, "volatility": 0.025},
        "TSLA": {"base_price": 250.0, "volatility": 0.04},
        "MSFT": {"base_price": 300.0, "volatility": 0.02},
        "GOOGL": {"base_price": 2800.0, "volatility": 0.018},
        "AMZN": {"base_price": 3500.0, "volatility": 0.022},
        "GOLD": {"base_price": 1850.0, "volatility": 0.015},
        "SILVER": {"base_price": 25.0, "volatility": 0.025},
        "OIL": {"base_price": 70.0, "volatility": 0.03},
        "COPPER": {"base_price": 4.5, "volatility": 0.02},
        "PLATINUM": {"base_price": 1000.0, "volatility": 0.018},
        "EUR/USD": {"base_price": 1.2000, "volatility": 0.008},
        "GBP/USD": {"base_price": 1.3800, "volatility": 0.012},
        "USD/JPY": {"base_price": 110.0, "volatility": 0.006},
        "USD/CHF": {"base_price": 0.9200, "volatility": 0.007},
        "AUD/USD": {"base_price": 0.7500, "volatility": 0.010},
        "BTC/USD": {"base_price": 45000.0, "volatility": 0.04},
        "ETH/USD": {"base_price": 3000.0, "volatility": 0.045},
        "ADA/USD": {"base_price": 1.5, "volatility": 0.06},
        "DOT/USD": {"base_price": 25.0, "volatility": 0.05},
        "LINK/USD": {"base_price": 25.0, "volatility": 0.055},
        "SPX": {"base_price": 4500.0, "volatility": 0.012},
        "NDX": {"base_price": 15000.0, "volatility": 0.015},
        "DJI": {"base_price": 35000.0, "volatility": 0.010},
        "RUT": {"base_price": 2200.0, "volatility": 0.018},
        "VIX": {"base_price": 20.0, "volatility": 0.08}
    }
    
    if instrument in instrument_adjustments:
        config.update(instrument_adjustments[instrument])
    
    return config 