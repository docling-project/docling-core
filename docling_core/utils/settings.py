from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCLINGCORE_")

    allow_image_file_uri: bool = False
    max_image_decoded_size: int = 20 * 1024 * 1024  # 20MB


settings = CoreSettings()
