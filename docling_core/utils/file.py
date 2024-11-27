#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""File-related utilities."""

import importlib
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

import requests
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError


def resolve_file_source(
    source: Union[Path, AnyHttpUrl, str], headers: Optional[Dict[str, str]] = None
) -> Path:
    """Resolves the source (URL, path) of a file to a local file path.

    If a URL is provided, the content is first downloaded to a temporary local file.

    Args:
        source (Path | AnyHttpUrl | str): The file input source. Can be a path or URL.

    Raises:
        ValueError: If source is of unexpected type.

    Returns:
        Path: The local file path.
    """
    try:
        http_url: AnyHttpUrl = TypeAdapter(AnyHttpUrl).validate_python(source)

        # make all header keys lower case
        _headers = headers or {}
        req_headers = {k.lower(): v for k, v in _headers.items()}
        # add user-agent is not set
        if "user-agent" not in req_headers:
            agent_name = f"docling-core/{importlib.metadata.version('docling-core')}"
            req_headers["user-agent"] = agent_name

        # fetch the page
        res = requests.get(http_url, stream=True, headers=req_headers)
        res.raise_for_status()
        fname = None
        # try to get filename from response header
        if cont_disp := res.headers.get("Content-Disposition"):
            for par in cont_disp.strip().split(";"):
                # currently only handling directive "filename" (not "*filename")
                if (split := par.split("=")) and split[0].strip() == "filename":
                    fname = "=".join(split[1:]).strip().strip("'\"") or None
                    break
        # otherwise, use name from URL:
        if fname is None:
            fname = Path(http_url.path or "").name or "file"
        local_path = Path(tempfile.mkdtemp()) / fname
        with open(local_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=1024):  # using 1-KB chunks
                f.write(chunk)
    except ValidationError:
        try:
            local_path = TypeAdapter(Path).validate_python(source)
        except ValidationError:
            raise ValueError(f"Unexpected source type encountered: {type(source)}")
    return local_path


def relative_path(src: Path, target: Path) -> Path:
    """Compute the relative path from `src` to `target`.

    Args:
        src (str | Path): The source directory or file path (must be absolute).
        target (str | Path): The target directory or file path (must be absolute).

    Returns:
        Path: The relative path from `src` to `target`.

    Raises:
        ValueError: If either `src` or `target` is not an absolute path.
    """
    src = Path(src).resolve()
    target = Path(target).resolve()

    # Ensure both paths are absolute
    if not src.is_absolute():
        raise ValueError(f"The source path must be absolute: {src}")
    if not target.is_absolute():
        raise ValueError(f"The target path must be absolute: {target}")

    # Find the common ancestor
    common_parts = []
    for src_part, target_part in zip(src.parts, target.parts):
        if src_part == target_part:
            common_parts.append(src_part)
        else:
            break

    # Determine the path to go up from src to the common ancestor
    up_segments = [".."] * (len(src.parts) - len(common_parts))

    # Add the path from the common ancestor to the target
    down_segments = target.parts[len(common_parts) :]

    # Combine and return the result
    return Path(*up_segments, *down_segments)
