#!/usr/bin/env python
"""Prefect Inference Deployment Script"""
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mlops.pipeline import market_master_inference_flow

if __name__ == "__main__":
    print("Running Market Master Inference Flow via Prefect...")
    # Create dummy data for inference
    dummy_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'action': ['buy', 'sell', 'hold']
    })
    
    result = market_master_inference_flow(
        model_name="action_predictor_equity_AAPL",
        data=dummy_data
    )
    print(f"Inference completed! Predictions: {len(result.get('predictions', []))}")
