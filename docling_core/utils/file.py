#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""File-related utilities."""

import importlib
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Union

import requests
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError
from typing_extensions import deprecated

from docling_core.types.doc.utils import relative_path  # noqa
from docling_core.types.io import DocumentStream


def resolve_remote_filename(
    http_url: AnyHttpUrl,
    response_headers: Dict[str, str],
    fallback_filename="file",
) -> str:
    """Resolves the filename from a remote url and its response headers.

    Args:
        source AnyHttpUrl: The source http url.
        response_headers Dict: Headers received while fetching the remote file.
        fallback_filename str: Filename to use in case none can be determined.

    Returns:
        str: The actual filename of the remote url.
    """
    fname = None
    # try to get filename from response header
    if cont_disp := response_headers.get("Content-Disposition"):
        for par in cont_disp.strip().split(";"):
            # currently only handling directive "filename" (not "*filename")
            if (split := par.split("=")) and split[0].strip() == "filename":
                fname = "=".join(split[1:]).strip().strip("'\"") or None
                break
    # otherwise, use name from URL:
    if fname is None:
        fname = Path(http_url.path or "").name or fallback_filename

    return fname


def resolve_source_to_stream(
    source: Union[Path, AnyHttpUrl, str], headers: Optional[Dict[str, str]] = None
) -> DocumentStream:
    """Resolves the source (URL, path) of a file to a binary stream.

    Args:
        source (Path | AnyHttpUrl | str): The file input source. Can be a path or URL.
        headers (Dict | None): Optional set of headers to use for fetching
            the remote URL.

    Raises:
        ValueError: If source is of unexpected type.

    Returns:
        DocumentStream: The resolved file loaded as a stream.
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
        fname = resolve_remote_filename(http_url=http_url, response_headers=res.headers)

        stream = BytesIO(res.content)
        doc_stream = DocumentStream(name=fname, stream=stream)
    except ValidationError:
        try:
            local_path = TypeAdapter(Path).validate_python(source)
            stream = BytesIO(local_path.read_bytes())
            doc_stream = DocumentStream(name=local_path.name, stream=stream)
        except ValidationError:
            raise ValueError(f"Unexpected source type encountered: {type(source)}")
    return doc_stream


def _resolve_source_to_path(
    source: Union[Path, AnyHttpUrl, str],
    headers: Optional[Dict[str, str]] = None,
    workdir: Optional[Path] = None,
) -> Path:
    doc_stream = resolve_source_to_stream(source=source, headers=headers)

    # use a temporary directory if not specified
    if workdir is None:
        workdir = Path(tempfile.mkdtemp())

    # create the parent workdir if it doesn't exist
    workdir.mkdir(exist_ok=True, parents=True)

    # save result to a local file
    local_path = workdir / doc_stream.name
    with local_path.open("wb") as f:
        f.write(doc_stream.stream.read())

    return local_path


def resolve_source_to_path(
    source: Union[Path, AnyHttpUrl, str],
    headers: Optional[Dict[str, str]] = None,
    workdir: Optional[Path] = None,
) -> Path:
    """Resolves the source (URL, path) of a file to a local file path.

    If a URL is provided, the content is first downloaded to a local file, located in
      the provided workdir or in a temporary directory if no workdir provided.

    Args:
        source (Path | AnyHttpUrl | str): The file input source. Can be a path or URL.
        headers (Dict | None): Optional set of headers to use for fetching
            the remote URL.
        workdir (Path | None): If set, the work directory where the file will
            be downloaded, otherwise a temp dir will be used.

    Raises:
        ValueError: If source is of unexpected type.

    Returns:
        Path: The local file path.
    """
    return _resolve_source_to_path(
        source=source,
        headers=headers,
        workdir=workdir,
    )


@deprecated("Use `resolve_source_to_path()` or `resolve_source_to_stream()`  instead")
def resolve_file_source(
    source: Union[Path, AnyHttpUrl, str],
    headers: Optional[Dict[str, str]] = None,
) -> Path:
    """Resolves the source (URL, path) of a file to a local file path.

    If a URL is provided, the content is first downloaded to a temporary local file.

    Args:
        source (Path | AnyHttpUrl | str): The file input source. Can be a path or URL.
        headers (Dict | None): Optional set of headers to use for fetching
            the remote URL.

    Raises:
        ValueError: If source is of unexpected type.

    Returns:
        Path: The local file path.
    """
    return _resolve_source_to_path(
        source=source,
        headers=headers,
    )
