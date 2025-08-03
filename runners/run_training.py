#!/usr/bin/env python
"""Training Pipeline Runner"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

if __name__ == "__main__":
    print("Running Market Master Training Pipeline...")
    try:
        # Import here to avoid circular imports
        from src.mlops.pipeline import market_master_training_flow
        
        result = market_master_training_flow(
            asset_class="equity",
            instrument="AAPL", 
            n_samples=10000
        )
        print("Training completed successfully!")
        print(f"Model URI: {result.get('model_uri', 'N/A')}")
        print(f"Training Accuracy: {result.get('training_metrics', {}).get('accuracy', 'N/A')}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
