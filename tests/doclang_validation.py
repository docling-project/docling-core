"""Helpers for validating DocLang XML against the reference doclang package."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import pytest

from docling_core.transforms.serializer._doclang_utils import _DOCLANG_VERSION

try:
    from doclang import ValidationError
    from doclang import validate as doclang_validate
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

# The Schematron backend (doclang[schematron-saxon]) has no wheels for every Python
# version; without it only the XSD part of the reference validation runs.
SCHEMATRON_BACKEND_AVAILABLE = importlib.util.find_spec("saxonche") is not None

requires_schematron_backend = pytest.mark.skipif(
    not SCHEMATRON_BACKEND_AVAILABLE,
    reason="Schematron backend not installed (doclang[schematron-saxon])",
)


def _spec_version(path: str | Path) -> str | None:
    """Spec version to validate against: the serializer's, unless the document declares one."""
    try:
        _, root = next(ET.iterparse(path, events=("start",)))
    except (ET.ParseError, StopIteration):
        return None  # malformed: left for the validator to report
    return None if root.get("version") is not None else _DOCLANG_VERSION


def _validate(path: str | Path, *, allow_empty_namespace: bool) -> None:
    doclang_validate(
        path,
        allow_empty_namespace=allow_empty_namespace,
        spec_version=_spec_version(path),
        xsd_only=not SCHEMATRON_BACKEND_AVAILABLE,
    )


xfail_invalid_dclg_xml = pytest.mark.xfail(
    reason="Serializer output fails DocLang reference validation (known bug)",
    raises=ValidationError,
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
        _validate(path, allow_empty_namespace=allow_empty_namespace)
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
        _validate(path, allow_empty_namespace=allow_empty_namespace)
    finally:
        os.unlink(path)


def validate_dclg_file(
    path: str | Path,
    *,
    allow_empty_namespace: bool = True,
) -> None:
    """Validate a DocLang XML file path."""
    if doclang_validate is None:
        pytest.skip("reference doclang package not installed")
    _validate(Path(path), allow_empty_namespace=allow_empty_namespace)


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
