"""File-related utilities."""

import ipaddress
import logging
import re
import socket
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import requests
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError
from requests.adapters import HTTPAdapter
from requests.utils import select_proxy
from typing_extensions import deprecated

from docling_core.types.doc.utils import relative_path
from docling_core.types.io import DocumentStream
from docling_core.utils.settings import settings

_logger = logging.getLogger(__name__)

_MAX_REDIRECTS = 5

# Chunk size for streaming remote downloads. Sized for document payloads (PDFs
# and similar), which are either small or in the multi-megabyte range, so a
# large chunk keeps the read loop short without wasting memory.
_DOWNLOAD_CHUNK_SIZE = 512 * 1024


class FileSizeLimitExceededError(ValueError):
    """Raised when a remote file exceeds the configured download size limit."""

    def __init__(self, filename: str, size: int, limit: int):
        self.filename = filename
        self.size = size
        self.limit = limit
        super().__init__(f"Remote file exceeds the maximum allowed size ({size} > {limit} bytes).")


def _ip_in_allowlist(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, allowlist: list[str]) -> bool:
    """Return whether a ip matches any IP addresss or CIDR entry in allowlist."""
    for entry in allowlist:
        entry = entry.strip()
        if not entry:
            continue
        try:
            network = ipaddress.ip_network(entry, strict=False)
            normalized = str(network)
            if normalized != entry:
                _logger.warning(
                    f"DOCLINGCORE_ALLOWED_PRIVATE_IPS entry {entry!r} was normalized to "
                    f"{normalized!r}. Consider using the explicit network address."
                )
            if ip in network:
                return True
        except ValueError:
            _logger.warning(f"Skipping malformed entry in DOCLINGCORE_ALLOWED_PRIVATE_IPS: {entry!r} ")
    return False


def _is_address_safe(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Check whether a single resolved IP address is globally routable.

    IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1) are unwrapped and
    evaluated as their underlying IPv4 address.

    Args:
        ip: The IP address to evaluate.

    Returns:
        True if the address is safe to connect to, False otherwise.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    if settings.allowed_private_ips and _ip_in_allowlist(ip, settings.allowed_private_ips):
        return True

    return ip.is_global and not (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified
    )


def _is_safe_url(url: str) -> bool:
    """Check whether every address a URL's hostname resolves to is globally routable.

    Args:
        url: The URL to validate.

    Returns:
        True if the URL is safe to fetch, False otherwise.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            return False

        try:
            ip = ipaddress.ip_address(hostname)
            return _is_address_safe(ip)
        except ValueError:
            pass

        try:
            results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except (socket.gaierror, socket.herror):
            return False

        if not results:
            return False

        for _family, _type, _proto, _canonname, sockaddr in results:
            try:
                ip = ipaddress.ip_address(sockaddr[0])
            except ValueError:
                return False
            if not _is_address_safe(ip):
                return False

        return True
    except Exception:
        return False


def _sanitize_filename(filename: str) -> str | None:
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


def _resolve_safe_addresses(host: str, port: int | None) -> list[str]:
    """Resolve a host once and return its addresses, rejecting it if any is not allowed.

    Args:
        host: The hostname or IP address to resolve.
        port: The port to resolve for.

    Returns:
        The resolved addresses, in resolver order.
    """
    try:
        results = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except (socket.gaierror, socket.herror) as exc:
        raise ValueError(f"Could not resolve host at connect time: {host}") from exc

    addresses = []
    for _family, _type, _proto, _canonname, sockaddr in results:
        addr_str = str(sockaddr[0])
        try:
            ip = ipaddress.ip_address(addr_str)
        except ValueError:
            raise ValueError(f"Unexpected address at connect time: {addr_str}") from None
        if not _is_address_safe(ip):
            raise ValueError(f"Connect-time address is not allowed: {addr_str}")
        addresses.append(addr_str)

    if not addresses:
        raise ValueError(f"No addresses returned for host: {host}")
    return addresses


class _SafeConnectionAdapter(HTTPAdapter):
    """HTTPAdapter that connects directly to a validated address.

    Prevents DNS rebinding: the host is resolved and validated once, and the
    connection pool is opened to the validated IP itself, so no later DNS
    lookup can change the target. The original hostname is kept for the
    ``Host`` header and for TLS (SNI and certificate verification).

    Proxied requests connect to the proxy, which resolves the target itself,
    so they rely on the pre-flight check alone.
    """

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        if select_proxy(request.url, proxies):
            return super().get_connection_with_tls_context(request, verify, proxies, cert)

        host_params, pool_kwargs = self.build_connection_pool_key_attributes(request, verify, cert)
        host = host_params["host"]
        address = _resolve_safe_addresses(host, host_params["port"])[0]
        if host_params["scheme"] == "https":
            pool_kwargs = {**pool_kwargs, "server_hostname": host, "assert_hostname": host}
        request.headers["Host"] = urlparse(request.url).netloc.rpartition("@")[2]

        return self.poolmanager.connection_from_host(
            host=address,
            port=host_params["port"],
            scheme=host_params["scheme"],
            pool_kwargs=pool_kwargs,
        )


def resolve_source_to_stream(
    source: Path | AnyHttpUrl | str,
    headers: dict[str, str] | None = None,
    max_file_size: int | None = None,
) -> DocumentStream:
    """Resolves the source (URL, path) of a file to a binary stream.

    Args:
        source: The file input source. Can be a path or URL.
        headers: Optional set of headers to use for fetching the remote URL.
        max_file_size: Optional maximum size, in bytes, for a remote download.
            When set, the download is rejected upfront if the declared
            ``Content-Length`` exceeds it, and aborted while streaming as soon as
            the received bytes exceed it.

    Raises:
        ValueError: If source is of unexpected type.
        FileSizeLimitExceededError: If a remote download exceeds ``max_file_size``.

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

        with requests.Session() as session:
            session.max_redirects = _MAX_REDIRECTS
            session.hooks["response"].append(_check_redirect_safety)
            session.mount("https://", _SafeConnectionAdapter())
            session.mount("http://", _SafeConnectionAdapter())

            with session.get(
                url_str,
                stream=True,
                headers=req_headers,
                allow_redirects=True,
            ) as res:
                res.raise_for_status()

                response_headers = dict(res.headers)
                fname = resolve_remote_filename(http_url=http_url, response_headers=response_headers)

                if max_file_size is not None:
                    content_length = res.headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            content_length_value = int(content_length)
                        except ValueError:
                            content_length_value = None

                        if content_length_value is not None and content_length_value > max_file_size:
                            raise FileSizeLimitExceededError(
                                filename=fname,
                                size=content_length_value,
                                limit=max_file_size,
                            )

                stream = BytesIO()
                downloaded = 0
                for chunk in res.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if max_file_size is not None and downloaded > max_file_size:
                        raise FileSizeLimitExceededError(
                            filename=fname,
                            size=downloaded,
                            limit=max_file_size,
                        )
                    stream.write(chunk)
                stream.seek(0)
                doc_stream = DocumentStream(name=fname, stream=stream)
    except ValidationError:
        if isinstance(source, str) and "://" in source:
            scheme = source.split("://", 1)[0].lower()
            if scheme not in ("http", "https"):
                raise ValueError(f"Unsupported URL scheme: '{scheme}'. Only http:// and https:// are supported.")
        try:
            local_path = TypeAdapter(Path).validate_python(source)
            stream = BytesIO(local_path.read_bytes())
            doc_stream = DocumentStream(name=local_path.name, stream=stream)
        except ValidationError:
            raise ValueError(f"Unexpected source type encountered: {type(source)}")
    return doc_stream


def _resolve_source_to_path(
    source: Path | AnyHttpUrl | str,
    headers: dict[str, str] | None = None,
    workdir: Path | None = None,
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
    source: Path | AnyHttpUrl | str,
    headers: dict[str, str] | None = None,
    workdir: Path | None = None,
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
    source: Path | AnyHttpUrl | str,
    headers: dict[str, str] | None = None,
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
