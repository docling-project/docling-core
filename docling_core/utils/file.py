"""File-related utilities."""

import ipaddress
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import requests
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError
from typing_extensions import deprecated

from docling_core.types.doc.utils import relative_path
from docling_core.types.io import DocumentStream

_MAX_REDIRECTS = 5


def _is_safe_url(url: str) -> bool:
    """Return whether a URL resolves to a globally routable address."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            return False

        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            import socket

            try:
                ip_str = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(ip_str)
            except (socket.gaierror, socket.herror):
                return False

        return ip.is_global and not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    except Exception:
        return False


def _sanitize_filename(filename: str) -> Optional[str]:
    """Return a basename-safe filename, or None if no usable basename remains."""
    normalized = filename.replace("\\", "/")
    basename = Path(normalized).name

    if not basename or basename in (".", "..") or "/" in basename:
        return None

    return basename


def resolve_remote_filename(
    http_url: AnyHttpUrl,
    response_headers: dict[str, str],
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
    raw_fname = None
    if cont_disp := response_headers.get("Content-Disposition"):
        for par in cont_disp.strip().split(";"):
            if (split := par.split("=")) and split[0].strip() == "filename":
                raw_fname = "=".join(split[1:]).strip().strip("'\"") or None
                break

    if raw_fname is None:
        raw_fname = Path(http_url.path or "").name or fallback_filename

    if fname := _sanitize_filename(raw_fname):
        return fname

    if fname := _sanitize_filename(fallback_filename):
        return fname

    raise ValueError("Could not derive a safe filename")


def resolve_source_to_stream(
    source: Union[Path, AnyHttpUrl, str], headers: Optional[dict[str, str]] = None
) -> DocumentStream:
    """Resolves the source (URL, path) of a file to a binary stream.

    Args:
        source (Path | AnyHttpUrl | str): The file input source. Can be a path or URL.
        headers (Optional[dict[str, str]]): Optional set of headers to use for fetching
            the remote URL.

    Raises:
        ValueError: If source is of unexpected type.

    Returns:
        DocumentStream: The resolved file loaded as a stream.
    """
    try:
        http_url: AnyHttpUrl = TypeAdapter(AnyHttpUrl).validate_python(source)
        url_str = str(http_url)

        if not _is_safe_url(url_str):
            raise ValueError(f"URL is not allowed: {url_str}")

        _headers = headers or {}
        req_headers = {k.lower(): v for k, v in _headers.items()}
        if "user-agent" not in req_headers:
            try:
                import importlib.metadata

                agent_name = f"docling-core/{importlib.metadata.version('docling-core')}"
            except Exception:
                agent_name = "docling-core"
            req_headers["user-agent"] = agent_name

        google_doc_id = re.search(
            r"google\.com\/(file|document|spreadsheets|presentation)\/d\/([\w-]+)",
            url_str,
        )
        if google_doc_id:
            doc_type = google_doc_id.group(1)
            doc_id = google_doc_id.group(2)

            if doc_type == "file":
                url_str = f"https://drive.google.com/uc?export=download&id={doc_id}"
            elif doc_type == "document":
                url_str = f"https://docs.google.com/document/d/{doc_id}/export?format=docx"
            elif doc_type == "spreadsheets":
                url_str = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=xlsx"
            elif doc_type == "presentation":
                url_str = f"https://docs.google.com/presentation/d/{doc_id}/export?format=pptx"
            else:
                raise ValueError(f"Unexpected Google doc type: {doc_type}")

            http_url = TypeAdapter(AnyHttpUrl).validate_python(url_str)

        session = requests.Session()
        session.max_redirects = _MAX_REDIRECTS

        def _check_redirect_safety(response, *args, **kwargs):
            """Validate each redirect target before following it."""
            if response.is_redirect or response.is_permanent_redirect:
                redirect_url = response.headers.get("location")
                if redirect_url:
                    if not redirect_url.startswith(("http://", "https://")):
                        from urllib.parse import urljoin

                        redirect_url = urljoin(response.url, redirect_url)

                    if not _is_safe_url(redirect_url):
                        raise ValueError(f"Redirect target is not allowed: {redirect_url}")

        session.hooks["response"].append(_check_redirect_safety)

        res = session.get(
            url_str,
            stream=True,
            headers=req_headers,
            allow_redirects=True,
        )
        res.raise_for_status()

        response_headers = dict(res.headers)
        fname = resolve_remote_filename(http_url=http_url, response_headers=response_headers)

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
    headers: Optional[dict[str, str]] = None,
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
    headers: Optional[dict[str, str]] = None,
    workdir: Optional[Path] = None,
) -> Path:
    """Resolves the source (URL, path) of a file to a local file path.

    If a URL is provided, the content is first downloaded to a local file, located in
      the provided workdir or in a temporary directory if no workdir provided.

    Args:
        source (Path | AnyHttpUrl | str): The file input source. Can be a path or URL.
        headers (Optional[dict[str, str]]): Optional set of headers to use for fetching
            the remote URL.
        workdir (Optional[Path]): If set, the work directory where the file will
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
    headers: Optional[dict[str, str]] = None,
) -> Path:
    """Resolves the source (URL, path) of a file to a local file path.

    If a URL is provided, the content is first downloaded to a temporary local file.

    Args:
        source (Path | AnyHttpUrl | str): The file input source. Can be a path or URL.
        headers (Optional[dict[str, str]]): Optional set of headers to use for fetching
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
