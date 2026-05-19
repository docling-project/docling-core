"""CLI for docling serializer."""

import importlib
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.base import ImageRefMode
from docling_core.utils.file import resolve_source_to_path

app = typer.Typer(
    name="docling-serialize",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


class OutputFormat(str, Enum):
    """Supported output formats."""

    HTML = "html"
    MARKDOWN = "markdown"
    DOCTAGS = "doctags"
    TEXT = "text"
    JSON = "json"


def version_callback(value: bool):
    """Callback for version inspection."""
    if value:
        docling_core_version = importlib.metadata.version("docling-core")
        print(f"Docling Core version: {docling_core_version}")
        raise typer.Exit()


@app.command(no_args_is_help=True)
def serialize(
    source: Annotated[
        str,
        typer.Argument(
            ...,
            metavar="source",
            help="Docling JSON or YAML file to serialize.",
        ),
    ],
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--to",
            "-t",
            help="Target output format.",
            case_sensitive=False,
        ),
    ] = OutputFormat.HTML,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path. If omitted, writes to stdout.",
        ),
    ] = None,
    split_page_view: Annotated[
        bool,
        typer.Option(
            "--split-page-view",
            "-s",
            help="Use split page view (HTML only).",
        ),
    ] = False,
    embed_images: Annotated[
        bool,
        typer.Option(
            "--embed-images/--no-embed-images",
            help="Embed images as base64 (HTML only). Defaults to embedded.",
        ),
    ] = True,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version information.",
        ),
    ] = None,
):
    """Serialize a DoclingDocument to the desired output format."""
    path = resolve_source_to_path(source=source)
    if path.suffix == ".json":
        doc = DoclingDocument.load_from_json(filename=path)
    elif path.suffix in [".yaml", ".yml"]:
        doc = DoclingDocument.load_from_yaml(filename=path)
    else:
        raise typer.BadParameter(f"Unsupported source file type: {path.suffix}")

    if output_format is OutputFormat.HTML:
        image_mode = ImageRefMode.EMBEDDED if embed_images else ImageRefMode.PLACEHOLDER
        result = doc.export_to_html(
            image_mode=image_mode,
            split_page_view=split_page_view,
        )
    elif output_format is OutputFormat.MARKDOWN:
        result = doc.export_to_markdown()
    elif output_format is OutputFormat.DOCTAGS:
        result = doc.export_to_doctags()
    elif output_format is OutputFormat.TEXT:
        result = doc.export_to_text()
    elif output_format is OutputFormat.JSON:
        result = doc.model_dump_json(indent=2)
    else:
        raise typer.BadParameter(f"Unsupported output format: {output_format}")

    if output is None:
        sys.stdout.write(result)
        if not result.endswith("\n"):
            sys.stdout.write("\n")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result, encoding="utf-8")
        typer.echo(f"Wrote {output}", err=True)


click_app = typer.main.get_command(app)


if __name__ == "__main__":
    app()
