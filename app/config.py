from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings.

    This class defines all environment variables and default values used
    across the image classification microservice, including server setup,
    model parameters, and file constraints.
    """

    host: str = "0.0.0.0"
    port: int = 4123

    model_name: str = "resnet18"
    model_weights: str = "DEFAULT"

    max_file_size: int = 25 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
