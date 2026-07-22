from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCLINGCORE_")

    allow_image_file_uri: bool = False
    max_image_decoded_size: int = 20 * 1024 * 1024  # 20MB
    allowed_private_ips: list[str] = []

    # DocLang deserialize budgets (DoS protection for untrusted markup / .dclx)
    max_doclang_xml_bytes: int = 128 * 1024 * 1024  # 128 MiB
    max_doclang_xml_depth: int = 128
    max_doclang_xml_elements: int = 1_000_000


settings = CoreSettings()
