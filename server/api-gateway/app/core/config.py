"""Application configuration management."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import List, Optional
from enum import Enum
import os


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = "Musical Spork API Gateway"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Security settings
    jwt_secret_key: str = Field(..., min_length=32, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT expiration time in minutes")
    allowed_hosts_str: str = Field(
        default="localhost,127.0.0.1,*.musical-spork.com",
        description="Allowed hosts for CORS (comma-separated)"
    )
    
    # Database settings
    database_url: Optional[str] = Field(None, description="Database connection URL")
    database_pool_size: int = Field(default=10, description="Database connection pool size")
    database_max_overflow: int = Field(default=20, description="Database max overflow connections")
    
    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_max_connections: int = Field(default=10, description="Redis max connections")
    
    # Service URLs
    user_service_url: Optional[str] = Field(None, description="User service URL")
    auth_service_url: Optional[str] = Field(None, description="Auth service URL")
    payment_service_url: Optional[str] = Field(None, description="Payment service URL")
    notification_service_url: Optional[str] = Field(None, description="Notification service URL")
    
    # External service API keys
    stripe_api_key: Optional[str] = Field(None, description="Stripe API key")
    sendgrid_api_key: Optional[str] = Field(None, description="SendGrid API key")
    google_oauth_client_id: Optional[str] = Field(None, description="Google OAuth client ID")
    google_oauth_client_secret: Optional[str] = Field(None, description="Google OAuth client secret")
    
    # Rate limiting settings
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(default=100, description="Requests per minute per user")
    rate_limit_burst_size: int = Field(default=200, description="Burst size for rate limiting")
    
    # Caching settings
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_default_ttl: int = Field(default=300, description="Default cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache entries")
    
    # Monitoring and logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    jaeger_endpoint: Optional[str] = Field(None, description="Jaeger tracing endpoint")
    
    # Health check settings
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    health_check_timeout: int = Field(default=10, description="Health check timeout in seconds")
    
    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v):
        """Validate JWT secret key length."""
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v
    
    @property
    def allowed_hosts(self) -> List[str]:
        """Get allowed hosts as a list."""
        return [host.strip() for host in self.allowed_hosts_str.split(",")]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
