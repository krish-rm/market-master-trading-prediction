"""
Action Predictor model for Market Master.
Predicts trading actions (buy/sell/hold) based on technical indicators.
"""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple, List
import os
import logging

# Use basic logging instead of relative imports
logger = logging.getLogger(__name__)


class ActionPredictor:
    """Action Predictor model for trading decisions."""
    
    def __init__(self, model_path: Optional[str] = None, use_mlflow: bool = True):
        """
        Initialize the Action Predictor.
        
        Args:
            model_path: Path to pre-trained model file
            use_mlflow: Whether to use MLflow for tracking
        """
        self.model = None
        self.scaler = StandardScaler()
        self.use_mlflow = use_mlflow
        
        # Feature names for technical indicators
        self.feature_names = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'obv', 'vwap', 'stoch_k', 'stoch_d',
            'williams_r', 'cci', 'adx', 'supertrend',
            'atr', 'atr_ratio', 'price_change', 'volume_change',
            'pivot_point'
        ]
        
        # Target classes
        self.classes = ['buy', 'sell', 'hold', 'strong_buy', 'strong_sell']
        
        # Model configuration
        self.model_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model."""
        self.model = RandomForestClassifier(**self.model_config)
        logger.info("Action Predictor model initialized", config=self.model_config)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw market data.
        
        Args:
            data: Raw market data with OHLCV columns
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Preparing features for Action Predictor")
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate basic technical indicators
        features = self._calculate_basic_indicators(data)
        
        # Select only the features we need
        available_features = [col for col in self.feature_names if col in features.columns]
        missing_features = [col for col in self.feature_names if col not in features.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # Create feature DataFrame
        feature_df = features[available_features].copy()
        
        # Fill missing values
        feature_df = feature_df.fillna(0)
        
        logger.info(f"Prepared {len(feature_df.columns)} features for {len(feature_df)} samples")
        return feature_df
    
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators."""
        features = data.copy()
        
        # Simple Moving Averages
        features['sma_20'] = data['close'].rolling(window=20).mean()
        features['sma_50'] = data['close'].rolling(window=50).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Price changes
        features['price_change'] = data['close'].pct_change()
        features['volume_change'] = data['volume'].pct_change()
        
        # Volatility
        features['atr'] = data['close'].rolling(window=14).std()
        
        return features
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        Train the Action Predictor model.
        
        Args:
            X_train: Training features (raw market data with OHLCV columns)
            y_train: Training labels
            X_val: Validation features (raw market data with OHLCV columns)
            y_val: Validation labels
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training Action Predictor model")
        
        # Debug: Check what columns we have
        logger.info(f"X_train columns before cleaning: {list(X_train.columns)}")
        logger.info(f"X_val columns before cleaning: {list(X_val.columns)}")
        
        # Drop action column if present in training data
        X_train_clean = X_train.drop(['action'], axis=1, errors='ignore')
        X_val_clean = X_val.drop(['action'], axis=1, errors='ignore')
        
        # Debug: Check what columns we have after cleaning
        logger.info(f"X_train_clean columns after cleaning: {list(X_train_clean.columns)}")
        logger.info(f"X_val_clean columns after cleaning: {list(X_val_clean.columns)}")
        
        # Prepare features
        X_train_features = self.prepare_features(X_train_clean)
        X_val_features = self.prepare_features(X_val_clean)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        X_val_scaled = self.scaler.transform(X_val_features)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
            'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
            'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
            'val_precision': precision_score(y_val, y_val_pred, average='weighted'),
            'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
            'val_recall': recall_score(y_val, y_val_pred, average='weighted')
        }
        
        # Log to MLflow if enabled
        if self.use_mlflow:
            with mlflow.start_run(run_name="action_predictor_training"):
                # Log parameters
                mlflow.log_params(self.model_config)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(self.model, "action_predictor")
                
                # Log feature importance
                feature_importance = self.get_feature_importance()
                mlflow.log_dict(feature_importance, "feature_importance.json")
        
        logger.info("Model training completed", metrics=metrics)
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare features
        X_features = self.prepare_features(features)
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare features
        X_features = self.prepare_features(features)
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def predict_with_confidence(self, features: pd.DataFrame) -> List[Dict[str, any]]:
        """
        Make predictions with confidence scores.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            List of predictions with confidence
        """
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = np.max(prob)
            results.append({
                'prediction': pred,
                'confidence': confidence,
                'probabilities': dict(zip(self.classes, prob))
            })
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature importance
        """
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        
        # Use the feature names directly instead of calling prepare_features with empty DataFrame
        # Since we know the expected features from technical indicators
        return dict(zip(self.feature_names[:len(importance)], importance))
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'classes': self.classes,
            'model_config': self.model_config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.classes = model_data['classes']
            self.model_config = model_data['model_config']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, any]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model performance")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Per-class metrics
        per_class_metrics = {}
        for class_name in self.classes:
            if class_name in report:
                per_class_metrics[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1_score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
        
        results = {
            'metrics': metrics,
            'per_class_metrics': per_class_metrics,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        logger.info("Model evaluation completed", metrics=metrics)
        return results


# Convenience functions

def train_action_predictor(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          use_mlflow: bool = True) -> Tuple[ActionPredictor, Dict[str, float]]:
    """
    Train an Action Predictor model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        use_mlflow: Whether to use MLflow
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    model = ActionPredictor(use_mlflow=use_mlflow)
    metrics = model.train(X_train, y_train, X_val, y_val)
    return model, metrics


def evaluate_model(model: ActionPredictor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, any]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained ActionPredictor model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Evaluation results
    """
    return model.evaluate(X_test, y_test) 