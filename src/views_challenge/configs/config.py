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
    use_api_keys: bool = Field(True, description="Use API keys for authentification")
    
    response_compression_min: int = Field(1024, description="Minimum size for compression")
    cell_request_default_limit: int = Field(50, description="Default request limit")
    cell_request_max_limit: int = Field(500, description="Maximum request limit")
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
