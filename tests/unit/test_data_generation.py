"""
Unit tests for data generation components
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.data import AssetClass, generate_training_data, generate_market_data
from src.data.data_generator import MarketDataGenerator


class TestDataGeneration(unittest.TestCase):
    """Test cases for data generation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.asset_class = AssetClass.EQUITY
        self.instrument = "TEST_STOCK"
        self.sample_size = 100

    def test_market_data_generator_creation(self):
        """Test MarketDataGenerator can be created"""
        generator = MarketDataGenerator(self.asset_class, self.instrument)
        self.assertIsInstance(generator, MarketDataGenerator)
        self.assertEqual(generator.asset_class, self.asset_class)
        self.assertEqual(generator.instrument, self.instrument)

    def test_generate_market_data(self):
        """Test market data generation"""
        data = generate_market_data(self.asset_class, self.instrument, self.sample_size)
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), self.sample_size)
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        for col in required_cols:
            self.assertIn(col, data.columns)
        
        # Check data quality
        self.assertFalse(data.isnull().any().any())  # No missing values
        self.assertTrue((data['high'] >= data['low']).all())  # Price consistency
        self.assertTrue((data['volume'] >= 0).all())  # Volume non-negative

    def test_generate_training_data(self):
        """Test training data generation with labels"""
        data = generate_training_data(self.asset_class, self.instrument, self.sample_size)
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)  # Some data generated (may be less due to indicators)
        
        # Check for action column
        self.assertIn('action', data.columns)
        
        # Check action values are valid
        valid_actions = ['buy', 'sell', 'hold', 'strong_buy', 'strong_sell']
        self.assertTrue(data['action'].isin(valid_actions).all())
        
        # Check technical indicators present
        indicator_cols = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'sma_50']
        for col in indicator_cols:
            self.assertIn(col, data.columns)

    def test_multi_asset_support(self):
        """Test generation for different asset classes"""
        asset_tests = [
            (AssetClass.CRYPTO, "BTC/USD"),
            (AssetClass.FOREX, "EUR/USD"),
            (AssetClass.COMMODITY, "GOLD")
        ]
        
        for asset_class, instrument in asset_tests:
            with self.subTest(asset_class=asset_class, instrument=instrument):
                data = generate_market_data(asset_class, instrument, 50)
                self.assertIsInstance(data, pd.DataFrame)
                self.assertEqual(len(data), 50)
                self.assertIn('asset_class', data.columns)
                self.assertEqual(data['asset_class'].iloc[0], asset_class.value)

    def test_data_quality_score(self):
        """Test data quality metrics"""
        data = generate_training_data(self.asset_class, self.instrument, 200)
        
        # Calculate quality score
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        quality_score = 1.0 - missing_ratio
        
        # Should have high quality (>0.9)
        self.assertGreater(quality_score, 0.9)

    def test_technical_indicators_calculation(self):
        """Test that technical indicators are calculated correctly"""
        data = generate_training_data(self.asset_class, self.instrument, 500)
        
        # RSI should be between 0 and 100
        if 'rsi' in data.columns:
            rsi_valid = data['rsi'].dropna()
            self.assertTrue((rsi_valid >= 0).all())
            self.assertTrue((rsi_valid <= 100).all())
        
        # Bollinger Bands relationship
        if all(col in data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            bb_data = data[['bb_upper', 'bb_middle', 'bb_lower']].dropna()
            self.assertTrue((bb_data['bb_upper'] >= bb_data['bb_middle']).all())
            self.assertTrue((bb_data['bb_middle'] >= bb_data['bb_lower']).all())


if __name__ == '__main__':
    unittest.main() 