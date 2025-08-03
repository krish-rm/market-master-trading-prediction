"""
Settings configuration for Market Master.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = Field(default="market-master", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database
    database_url: str = Field(
        default="postgresql://market_user:market_password@localhost:5432/market_master",
        env="DATABASE_URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL"
    )
    
    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="market_master_ai",
        env="MLFLOW_EXPERIMENT_NAME"
    )
    
    # AWS
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_s3_bucket: str = Field(
        default="market-master-mlflow-artifacts",
        env="AWS_S3_BUCKET"
    )
    
    # Monitoring
    evidently_service_url: str = Field(
        default="http://localhost:8080",
        env="EVIDENTLY_SERVICE_URL"
    )
    grafana_url: str = Field(
        default="http://localhost:3000",
        env="GRAFANA_URL"
    )
    prometheus_url: str = Field(
        default="http://localhost:9090",
        env="PROMETHEUS_URL"
    )
    
    # Prefect
    prefect_api_url: str = Field(
        default="http://localhost:4200/api",
        env="PREFECT_API_URL"
    )
    
    # Model Configuration
    model_accuracy_threshold: float = Field(default=0.6, env="MODEL_ACCURACY_THRESHOLD")
    model_f1_threshold: float = Field(default=0.5, env="MODEL_F1_THRESHOLD")
    retraining_interval_days: int = Field(default=7, env="RETRAINING_INTERVAL_DAYS")
    
    # Trading Configuration
    game_duration_seconds: int = Field(default=300, env="GAME_DURATION_SECONDS")
    tick_interval_seconds: int = Field(default=1, env="TICK_INTERVAL_SECONDS")
    max_position_size: float = Field(default=2.0, env="MAX_POSITION_SIZE")
    risk_free_rate: float = Field(default=0.02, env="RISK_FREE_RATE")
    
    # Asset Classes
    supported_assets: List[str] = Field(
        default=["equity", "commodity", "forex", "crypto", "indices"],
        env="SUPPORTED_ASSETS"
    )
    default_asset_class: str = Field(default="equity", env="DEFAULT_ASSET_CLASS")
    
    # Security
    secret_key: str = Field(default="your_secret_key_here", env="SECRET_KEY")
    jwt_secret_key: str = Field(default="your_jwt_secret_key_here", env="JWT_SECRET_KEY")
    
    # Email
    smtp_host: str = Field(default="smtp.gmail.com", env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    
    # Slack
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    
    # Docker
    docker_registry: str = Field(default="your_docker_registry", env="DOCKER_REGISTRY")
    docker_image_name: str = Field(default="market-master", env="DOCKER_IMAGE_NAME")
    
    # Kubernetes
    kubernetes_namespace: str = Field(default="market-master", env="KUBERNETES_NAMESPACE")
    kubernetes_replicas: int = Field(default=3, env="KUBERNETES_REPLICAS")
    
    # Data Storage
    data_storage_path: str = Field(
        default="./data",
        env="DATA_STORAGE_PATH"
    )
    raw_data_path: str = Field(
        default="./data/raw",
        env="RAW_DATA_PATH"
    )
    processed_data_path: str = Field(
        default="./data/processed",
        env="PROCESSED_DATA_PATH"
    )
    model_data_path: str = Field(
        default="./data/models",
        env="MODEL_DATA_PATH"
    )
    
    # Terraform
    terraform_state_bucket: str = Field(
        default="market-master-terraform-state",
        env="TERRAFORM_STATE_BUCKET"
    )
    terraform_state_region: str = Field(default="us-east-1", env="TERRAFORM_STATE_REGION")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Parse supported assets from string if needed
        if isinstance(self.supported_assets, str):
            self.supported_assets = [asset.strip() for asset in self.supported_assets.split(",")]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return not self.debug
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.debug
    
    @property
    def has_aws_credentials(self) -> bool:
        """Check if AWS credentials are configured."""
        return bool(self.aws_access_key_id and self.aws_secret_access_key)
    
    @property
    def has_email_config(self) -> bool:
        """Check if email configuration is complete."""
        return bool(self.smtp_username and self.smtp_password)
    
    @property
    def has_slack_config(self) -> bool:
        """Check if Slack configuration is available."""
        return bool(self.slack_webhook_url)


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings 