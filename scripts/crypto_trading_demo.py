#!/usr/bin/env python3
"""
Crypto Trading Demo - Market Master
Shows complete workflow for crypto trading with BTC/USD
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data import AssetClass, generate_training_data, generate_market_data
from src.models import ActionPredictor, train_action_predictor
from src.mlops import ModelRegistry
from src.mlops.monitoring_simple import SimpleComprehensiveMonitor
from src.llm import MockTradingCoach, TradingPersona
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)


def print_step(step_num: int, title: str):
    """Print a formatted step."""
    print(f"\n📋 Step {step_num}: {title}")
    print("-" * 60)


def print_crypto_data(data: pd.DataFrame, title: str = "Crypto Market Data"):
    """Print formatted crypto data."""
    print(f"\n📊 {title}:")
    print(f"   Data Shape: {data.shape}")
    print(f"   Time Range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"   Price Range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   Volume Range: {data['volume'].min():,.0f} - {data['volume'].max():,.0f}")
    
    # Show sample data
    print(f"\n📈 Sample Data (Last 5 rows):")
    sample_data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(5)
    for _, row in sample_data.iterrows():
        print(f"   {row['timestamp'].strftime('%H:%M:%S')} | "
              f"O:${row['open']:.2f} H:${row['high']:.2f} L:${row['low']:.2f} "
              f"C:${row['close']:.2f} V:{row['volume']:,.0f}")


def print_predictions(predictions: np.ndarray, confidences: np.ndarray, data: pd.DataFrame):
    """Print formatted predictions."""
    print(f"\n🤖 Model Predictions:")
    print(f"   Total Predictions: {len(predictions)}")
    
    # Show prediction distribution
    pred_counts = pd.Series(predictions).value_counts()
    print(f"   Prediction Distribution:")
    for action, count in pred_counts.items():
        percentage = (count / len(predictions)) * 100
        print(f"     {action}: {count} ({percentage:.1f}%)")
    
    # Show recent predictions with confidence
    print(f"\n📊 Recent Predictions with Confidence:")
    recent_data = data.tail(10)
    recent_preds = predictions[-10:]
    recent_confs = confidences[-10:]
    
    for i, (_, row) in enumerate(recent_data.iterrows()):
        pred = recent_preds[i]
        conf = recent_confs[i]
        price = row['close']
        timestamp = row['timestamp'].strftime('%H:%M:%S')
        
        # Color code predictions
        if pred in ['strong_buy', 'buy']:
            emoji = "🟢"
        elif pred in ['strong_sell', 'sell']:
            emoji = "🔴"
        else:
            emoji = "🟡"
        
        print(f"   {timestamp} | ${price:.2f} | {emoji} {pred} (Confidence: {conf:.1%})")


def print_trading_advice(advice_list: list):
    """Print formatted trading advice."""
    print(f"\n💡 Trading Coach Advice:")
    print(f"   Total Advice Generated: {len(advice_list)}")
    
    if advice_list:
        latest_advice = advice_list[-1]
        print(f"\n🎯 Latest Trading Decision:")
        print(f"   Action: {latest_advice['action']}")
        print(f"   Confidence: {latest_advice['confidence']:.1%}")
        print(f"   Position Size: {latest_advice['position_size']:.1%}")
        print(f"   Stop Loss: {latest_advice['stop_loss']:.1%}")
        print(f"   Take Profit: {latest_advice['take_profit']:.1%}")
        print(f"   Market Condition: {latest_advice['market_condition']}")
        print(f"   Risk Level: {latest_advice['risk_level']}")
        print(f"   Explanation: {latest_advice['explanation']}")


def run_crypto_trading_demo():
    """Run complete crypto trading demonstration."""
    print_header("Crypto Trading Demo - Market Master")
    print("🚀 Demonstrating complete workflow for BTC/USD trading")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    asset_class = AssetClass.CRYPTO
    instrument = "BTC/USD"
    n_samples = 5000  # Smaller dataset for demo
    
    print(f"\n🎯 Demo Configuration:")
    print(f"   Asset Class: {asset_class.value}")
    print(f"   Instrument: {instrument}")
    print(f"   Training Samples: {n_samples:,}")
    print(f"   Market Hours: 24/7 (Crypto)")
    print(f"   Volatility: High")
    
    try:
        # Step 1: Generate Crypto Market Data
        print_step(1, "Generating Crypto Market Data")
        print("📈 Generating realistic BTC/USD market data...")
        
        # Generate training data
        training_data = generate_training_data(asset_class, instrument, n_samples)
        print_crypto_data(training_data, "Training Data")
        
        # Generate recent market data for inference
        recent_data = generate_market_data(asset_class, instrument, 100)
        print_crypto_data(recent_data, "Recent Market Data (for inference)")
        
        # Step 2: Train Crypto-Specific Model
        print_step(2, "Training Crypto Trading Model")
        print("🤖 Training Action Predictor model on crypto data...")
        
        # Generate raw market data for training (without technical indicators)
        raw_training_data = generate_market_data(asset_class, instrument, n_samples)
        
        # Split raw data
        train_size = int(len(raw_training_data) * 0.8)
        train_data = raw_training_data.iloc[:train_size].copy()
        test_data = raw_training_data.iloc[train_size:].copy()
        
        # Generate labels for the raw data
        from src.data.data_generator import MarketDataGenerator
        generator = MarketDataGenerator(asset_class, instrument)
        
        # Generate labels - ensure we pass the right data structure
        y_train = pd.Series(generator._generate_labels(train_data), index=train_data.index)
        y_test = pd.Series(generator._generate_labels(test_data), index=test_data.index)
        
        # Train model (pass raw data, let ActionPredictor handle feature engineering)
        # Remove metadata columns that shouldn't be used as features
        train_features = train_data.drop(['asset_class', 'instrument', 'session', 'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        test_features = test_data.drop(['asset_class', 'instrument', 'session', 'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        model, train_metrics = train_action_predictor(train_features, y_train, test_features, y_test)
        
        print(f"\n✅ Model Training Completed:")
        print(f"   Training Accuracy: {train_metrics['train_accuracy']:.1%}")
        print(f"   Validation Accuracy: {train_metrics['val_accuracy']:.1%}")
        print(f"   Training F1 Score: {train_metrics['train_f1']:.3f}")
        print(f"   Validation F1 Score: {train_metrics['val_f1']:.3f}")
        
        # Step 3: Make Crypto Predictions
        print_step(3, "Making Real-Time Crypto Predictions")
        print("🔮 Making predictions on recent BTC/USD data...")
        
        # Prepare recent data for prediction (remove metadata columns)
        recent_features = recent_data.drop(['asset_class', 'instrument', 'session', 'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        # Make predictions (ActionPredictor will handle feature engineering internally)
        predictions = model.predict(recent_features)
        probabilities = model.predict_proba(recent_features)
        confidences = np.max(probabilities, axis=1)
        
        print_predictions(predictions, confidences, recent_data)
        
        # Step 4: Generate Trading Advice
        print_step(4, "Generating Trading Coach Advice")
        print("🎯 Getting personalized trading advice...")
        
        # Initialize trading coach for crypto
        coach = MockTradingCoach(TradingPersona.MODERATE)
        trading_advice = []
        
        # Generate advice for each prediction
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            # Use original recent_data (with OHLCV) for trading advice
            advice = coach.get_trading_advice(
                pred, conf, recent_data.iloc[i:i+1]  # Use original OHLCV data for trading advice
            )
            trading_advice.append(advice)
        
        print_trading_advice(trading_advice)
        
        # Step 5: Monitor Performance
        print_step(5, "Monitoring Model Performance")
        print("🔍 Setting up comprehensive monitoring...")
        
        # Initialize monitoring with training features (not raw data)
        monitor = SimpleComprehensiveMonitor(train_features)
        monitoring_results = monitor.run_monitoring(test_features, model.predict(test_features))
        
        print(f"\n📊 Monitoring Results:")
        print(f"   Data Quality Score: {monitoring_results['data_quality_score']:.3f}")
        print(f"   Drift Detected: {'Yes' if monitoring_results['drift_detected'] else 'No'}")
        print(f"   Overall Status: {monitoring_results['overall_status']}")
        print(f"   Performance Metrics: {monitoring_results['performance_metrics']}")
        
        # Step 6: Crypto-Specific Analysis
        print_step(6, "Crypto Market Analysis")
        print("📊 Analyzing crypto-specific patterns...")
        
        # Analyze prediction patterns
        pred_series = pd.Series(predictions)
        print(f"\n🎯 Crypto Trading Pattern Analysis:")
        bullish_count = (pred_series.isin(['buy', 'strong_buy'])).sum()
        bullish_percentage = (bullish_count / len(pred_series) * 100)
        print(f"   Bullish Signals (buy/strong_buy): {bullish_count} ({bullish_percentage:.1f}%)")
        
        bearish_count = (pred_series.isin(['sell', 'strong_sell'])).sum()
        bearish_percentage = (bearish_count / len(pred_series) * 100)
        print(f"   Bearish Signals (sell/strong_sell): {bearish_count} ({bearish_percentage:.1f}%)")
        hold_count = (pred_series == 'hold').sum()
        hold_percentage = (hold_count / len(pred_series) * 100)
        print(f"   Neutral Signals (hold): {hold_count} ({hold_percentage:.1f}%)")
        
        # Analyze confidence patterns
        high_confidence = (confidences > 0.8).sum()
        medium_confidence = ((confidences >= 0.6) & (confidences <= 0.8)).sum()
        low_confidence = (confidences < 0.6).sum()
        
        print(f"\n🎲 Confidence Analysis:")
        print(f"   High Confidence (>80%): {high_confidence} ({high_confidence/len(confidences)*100:.1f}%)")
        print(f"   Medium Confidence (60-80%): {medium_confidence} ({medium_confidence/len(confidences)*100:.1f}%)")
        print(f"   Low Confidence (<60%): {low_confidence} ({low_confidence/len(confidences)*100:.1f}%)")
        
        # Step 7: Risk Assessment
        print_step(7, "Crypto Risk Assessment")
        print("⚠️  Assessing crypto trading risks...")
        
        # Calculate risk metrics
        avg_confidence = np.mean(confidences)
        volatility = recent_data['close'].pct_change().std()
        price_range = (recent_data['close'].max() - recent_data['close'].min()) / recent_data['close'].min()
        
        print(f"\n⚠️  Risk Metrics:")
        print(f"   Average Prediction Confidence: {avg_confidence:.1%}")
        print(f"   Price Volatility: {volatility:.1%}")
        print(f"   Price Range: {price_range:.1%}")
        
        # Risk assessment
        if avg_confidence > 0.7 and volatility < 0.05:
            risk_level = "LOW"
            risk_emoji = "🟢"
        elif avg_confidence > 0.6 and volatility < 0.1:
            risk_level = "MEDIUM"
            risk_emoji = "🟡"
        else:
            risk_level = "HIGH"
            risk_emoji = "🔴"
        
        print(f"   Overall Risk Level: {risk_emoji} {risk_level}")
        
        # Final Summary
        print_header("Crypto Trading Demo Summary")
        print("🎉 Crypto Trading Demo Completed Successfully!")
        
        print(f"\n📊 Key Results for BTC/USD:")
        print(f"   🎯 Model Accuracy: {train_metrics['val_accuracy']:.1%}")
        print(f"   📈 Prediction Distribution: {pred_series.value_counts().to_dict()}")
        print(f"   💡 Trading Advice Generated: {len(trading_advice)}")
        print(f"   🔍 Monitoring Status: {monitoring_results['overall_status']}")
        print(f"   ⚠️  Risk Level: {risk_level}")
        
        print(f"\n🚀 Crypto-Specific Insights:")
        print(f"   • 24/7 market requires continuous monitoring")
        print(f"   • High volatility affects prediction confidence")
        print(f"   • Model adapts to crypto market patterns")
        print(f"   • Automated risk management essential")
        
        return {
            'model_accuracy': train_metrics['val_accuracy'],
            'predictions': predictions,
            'confidences': confidences,
            'trading_advice': trading_advice,
            'monitoring_results': monitoring_results,
            'risk_level': risk_level
        }
        
    except Exception as e:
        logger.error(f"Crypto demo failed: {e}")
        print(f"\n❌ Crypto demo failed with error: {e}")
        return None


def main():
    """Main entry point for crypto trading demo."""
    print("🎯 Market Master - Crypto Trading Demo")
    print("=" * 50)
    
    # Run the crypto trading demo
    results = run_crypto_trading_demo()
    
    if results:
        print("\n🎊 Crypto trading demo completed successfully!")
        print("Ready to trade BTC/USD with AI assistance! 🚀")
    else:
        print("\n💥 Crypto trading demo encountered an error.")
        print("Please check the logs for details.")


if __name__ == "__main__":
    main() 