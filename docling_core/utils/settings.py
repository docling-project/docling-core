from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCLINGCORE_")

    allow_image_file_uri: Annotated[
        bool,
        Field(
            description=(
                "Allow reading local image files referenced by an ImageRef, "
                "either as file:// URIs or as plain filesystem paths."
            ),
        ),
    ] = False
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

    max_table_cells: Annotated[
        int,
        Field(
            description=(
                "Maximum number of grid cells (num_rows * num_cols) a table may declare. "
                "The grid is materialized in memory on export, so this bounds the cost of untrusted tables."
            ),
        ),
    ] = 1_000_000

    max_total_table_cells: Annotated[
        int,
        Field(
            description=(
                "Maximum total number of grid cells summed across all tables in a single document. "
                "Bounds the cost of documents that contain many tables each just under the per-table limit."
            ),
        ),
    ] = 5_000_000

    # DocLang deserialize budgets (DoS protection for untrusted markup / .dclx)
    max_doclang_xml_bytes: int = 128 * 1024 * 1024  # 128 MiB
    max_doclang_xml_depth: int = 128
    max_doclang_xml_elements: int = 1_000_000


settings = CoreSettings()
