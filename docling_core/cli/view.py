"""CLI for docling viewer."""

import importlib
import tempfile
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.base import ImageRefMode
from docling_core.utils.file import resolve_source_to_path

app = typer.Typer(
    name="Docling",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def version_callback(value: bool):
    """Callback for version inspection."""
    if value:
        docling_core_version = importlib.metadata.version("docling-core")
        print(f"Docling Core version: {docling_core_version}")
        raise typer.Exit()


@app.command(no_args_is_help=True)
def view(
    source: Annotated[
        str,
        typer.Argument(
            ...,
            metavar="source",
            help="Docling JSON or YAML file to view.",
        ),
    ],
    split_view: Annotated[
        bool,
        typer.Option(
            "--split-view",
            "-s",
            help="Split view of the document.",
        ),
    ] = False,
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
    """Display a DoclingDocument file on the default browser."""
    path = resolve_source_to_path(source=source)
    if path.suffix == ".json":
        doc = DoclingDocument.load_from_json(filename=path)
    elif path.suffix in [".yaml", ".yml"]:
        doc = DoclingDocument.load_from_yaml(filename=path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    target_path = Path(tempfile.mkdtemp()) / f"{path.stem}.html"
    html_output = doc.export_to_html(
        image_mode=ImageRefMode.EMBEDDED,
        split_page_view=split_view,
    )
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(html_output)
    webbrowser.open(url=f"file://{target_path.absolute().resolve()}")


click_app = typer.main.get_command(app)

if __name__ == "__main__":
    app()
