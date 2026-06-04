"""Helpers for validating DocLang XML against the reference doclang package."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union

import pytest

try:
    from doclang import ValidationError, validate as doclang_validate
except ImportError:
    doclang_validate = None  # type: ignore[assignment,misc]
    ValidationError = Exception  # type: ignore[misc,assignment]


def doclang_validator_available() -> bool:
    """Return True when the reference ``doclang`` package is importable."""
    return doclang_validate is not None


doclang_validator = pytest.mark.skipif(
    not doclang_validator_available(),
    reason="reference doclang package not installed (uv sync --extra doclang-validation)",
)

xfail_invalid_dclg_xml = pytest.mark.xfail(
    reason="Serializer output fails DocLang reference validation (known bug)",
    raises=ValidationError,
    strict=False,
)

xfail_layer_token_deferred = pytest.mark.xfail(
    reason="<layer> emission deferred until doclang>=0.5",
    strict=False,
)


def assert_valid_dclg_xml(
    xml_text: str,
    *,
    allow_empty_namespace: bool = True,
) -> None:
    """Validate DocLang XML; no-op when the reference validator is not installed."""
    if doclang_validate is None:
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".dclg.xml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(xml_text)
        path = tmp.name
    try:
        doclang_validate(path, allow_empty_namespace=allow_empty_namespace)
    finally:
        os.unlink(path)


def validate_dclg_xml(
    xml_text: str,
    *,
    allow_empty_namespace: bool = True,
) -> None:
    """Validate DocLang XML text; raises on failure when validator is available."""
    if doclang_validate is None:
        pytest.skip("reference doclang package not installed")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".dclg.xml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(xml_text)
        path = tmp.name

    try:
        doclang_validate(path, allow_empty_namespace=allow_empty_namespace)
    finally:
        os.unlink(path)


def validate_dclg_file(
    path: Union[str, Path],
    *,
    allow_empty_namespace: bool = True,
) -> None:
    """Validate a DocLang XML file path."""
    if doclang_validate is None:
        pytest.skip("reference doclang package not installed")
    doclang_validate(Path(path), allow_empty_namespace=allow_empty_namespace)


def assert_invalid_dclg_xml(
    xml_text: str,
    *,
    allow_empty_namespace: bool = True,
) -> None:
    """Assert DocLang XML is rejected by the reference validator (known bad output)."""
    if doclang_validate is None:
        pytest.skip("reference doclang package not installed")
    with pytest.raises(ValidationError):
        validate_dclg_xml(xml_text, allow_empty_namespace=allow_empty_namespace)
