from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCLINGCORE_")

    allow_image_file_uri: bool = False
    max_image_decoded_size: int = 20 * 1024 * 1024  # 20MB
    allowed_private_ips: Annotated[
        list[str],
        Field(
            description=(
                "List of IP addresses and/or CIDR ranges that bypass SSRF protection. "
                "Defaults to empty, which preserves full SSRF protection."
            ),
            examples=[
                ["192.168.1.0/24", "10.0.0.5", "127.0.0.1"],
            ],
        ),
    ] = []

    # DocLang deserialize budgets (DoS protection for untrusted markup / .dclx)
    max_doclang_xml_bytes: int = 128 * 1024 * 1024  # 128 MiB
    max_doclang_xml_depth: int = 128
    max_doclang_xml_elements: int = 1_000_000


settings = CoreSettings()
