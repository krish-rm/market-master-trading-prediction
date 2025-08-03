#!/usr/bin/env python
"""Inference Pipeline Runner"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

if __name__ == "__main__":
    print("Running Market Master Inference Pipeline...")
    try:
        # Import here to avoid circular imports
        from src.mlops.pipeline import market_master_inference_flow
        from src.data import generate_training_data, AssetClass
        
        # Generate some sample data for inference
        sample_data = generate_training_data(AssetClass.EQUITY, "AAPL", 100)
        
        result = market_master_inference_flow(
            model_name="action_predictor_equity_AAPL",
            data=sample_data
        )
        
        print("Inference completed successfully!")
        print(f"Predictions made: {len(result.get('predictions', []))}")
        print(f"Prediction distribution: {result.get('prediction_distribution', {})}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        print("Note: Make sure you've trained a model first!")
        import traceback
        traceback.print_exc()
