"""
Model factory for Market Master.
Manages different model types and configurations.
"""

from typing import Dict, Any, Optional, Type
from .action_predictor import ActionPredictor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Factory for creating and managing ML models."""
    
    def __init__(self):
        """Initialize the model factory."""
        self.models = {
            'action_predictor': ActionPredictor
        }
        
        self.configs = {
            'action_predictor': {
                'default': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                'fast': {
                    'n_estimators': 50,
                    'max_depth': 8,
                    'min_samples_split': 10,
                    'min_samples_leaf': 4,
                    'random_state': 42
                },
                'accurate': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 3,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            }
        }
    
    def create_model(self, model_type: str, config_name: str = 'default', 
                    **kwargs) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            config_name: Configuration preset name
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get base configuration
        config = self.configs[model_type].get(config_name, {}).copy()
        
        # Override with kwargs
        config.update(kwargs)
        
        # Create model
        model_class = self.models[model_type]
        model = model_class(**config)
        
        logger.info(f"Created {model_type} model", config_name=config_name, config=config)
        return model
    
    def get_available_models(self) -> list:
        """Get list of available model types."""
        return list(self.models.keys())
    
    def get_available_configs(self, model_type: str) -> list:
        """Get list of available configurations for a model type."""
        if model_type not in self.configs:
            return []
        return list(self.configs[model_type].keys())
    
    def register_model(self, name: str, model_class: Type, configs: Dict[str, Dict] = None):
        """
        Register a new model type.
        
        Args:
            name: Model name
            model_class: Model class
            configs: Default configurations
        """
        self.models[name] = model_class
        if configs:
            self.configs[name] = configs
        
        logger.info(f"Registered new model type: {name}")


def get_model_config(model_type: str, config_name: str = 'default') -> Dict[str, Any]:
    """
    Get model configuration.
    
    Args:
        model_type: Type of model
        config_name: Configuration preset name
        
    Returns:
        Model configuration
    """
    factory = ModelFactory()
    
    if model_type not in factory.configs:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if config_name not in factory.configs[model_type]:
        raise ValueError(f"Unknown config: {config_name} for model: {model_type}")
    
    return factory.configs[model_type][config_name].copy()


# Global factory instance
model_factory = ModelFactory() 