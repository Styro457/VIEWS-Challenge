from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    port: int = 8000

    database_url: str = ""

    response_compression_min: int = 1024
    cell_request_default_limit: int = 50
    cell_request_max_limit: int = 500
    timeout: int = 60

    debug: bool = False


    class Config:
        env_file = ".env"

settings = Settings()
