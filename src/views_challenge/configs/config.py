"""
Config file for accessing settings located from environment variables
Using pydantic-settings would be cleaner, but it conflicts with views_pipeline_core
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()  # load .env into os.environ


class Settings(BaseModel):
    port: int = Field(8000, description="Server port")
        
    database_url: str = Field("", description="Database connection URL")
    database_user: str = Field("", description="Database username")
    database_password: str = Field("", description="Database password")
    database_port: int = Field(5432, description="Database port")
    
    default_daily_requests_limit: int = Field(1000, description="Default daily requests limit per API key")
    
    key_life_span: int = Field(90, description="API key life span in days")
    keys_mode: bool = Field(True, description="Enable API key support")

    rate_limit_requests: int = Field(100, description="Number of requests allowed in the rate limit window")
    rate_limit_window: int = Field(60, description="Rate limit window in minutes")

    response_compression_min: int = Field(1024, description="Minimum size for compression")
    timeout: int = Field(60, description="Request timeout in seconds")
    
    debug: bool = Field(False, description="Enable debug mode")

    @classmethod
    def from_env(cls):
        raw = {}
        for field in cls.model_fields.keys():
            env_key = field.upper()
            if env_key in os.environ:
                raw[field] = os.environ[env_key]
        return cls(**raw)


try:
    settings = Settings.from_env()
except ValidationError as e:
    raise RuntimeError(f"Invalid environment configuration: {e}")
