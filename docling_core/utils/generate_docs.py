"""Generate documentation of Docling types as JSON schema.

Example:
    python docling_core/utils/generate_docs.py /tmp/docling_core_files
"""

import argparse
import json
from argparse import BooleanOptionalAction
from pathlib import Path
from shutil import rmtree
from typing import Final

from docling_core.utils.generate_jsonschema import generate_json_schema

MODELS: Final = ["DoclingDocument", "Record", "Generic"]


def _prepare_directory(folder: str, clean: bool = False) -> None:
    """Create a directory or empty its content if it already exists.

    Args:
        folder: The name of the directory.
        clean: Whether any existing content in the directory should be removed.
    """
    folder_path = Path(folder)
    if folder_path.is_dir():
        if clean:
            for path in folder_path.glob("**/*"):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    rmtree(path)
    else:
        folder_path.mkdir(parents=True, exist_ok=True)


def generate_collection_jsonschema(folder: str):
    """Generate the JSON schema of Docling collections and export them to a folder.

    Args:
        folder: The name of the directory.
    """
    folder_path = Path(folder)
    for item in MODELS:
        json_schema = generate_json_schema(item)
        output_file = folder_path / f"{item}.json"
        output_file.write_text(
            json.dumps(json_schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> None:
    """Generate the JSON Schema of Docling collections and export documentation."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "directory",
        help=(
            "Directory to generate files. If it exists, any existing content will be"
            " removed."
        ),
    )
    argparser.add_argument(
        "--clean",
        help="Whether any existing content in directory should be removed.",
        action=BooleanOptionalAction,
        dest="clean",
        default=False,
        required=False,
    )
    args = argparser.parse_args()

    _prepare_directory(args.directory, args.clean)

    generate_collection_jsonschema(args.directory)


if __name__ == "__main__":
    main()
