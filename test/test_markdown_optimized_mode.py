"""Regression tests for Python optimized mode (-O).

Python strips `assert` statements under -O. If production runs with -O / PYTHONOPTIMIZE,
serialization must not rely on side effects inside asserts.

See: https://github.com/docling-project/docling-core/issues/460
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_markdown_export_works_in_optimized_mode() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    code = """
from docling_core.types.doc.document import DoclingDocument

# Construct a doc with list items whose marker defaults to "" (not a valid markdown marker).
# This triggers marker normalization in the markdown serializer.
doc = DoclingDocument(name="t")
lst = doc.add_list_group()
doc.add_list_item("Item A", parent=lst)
doc.add_list_item("Item B", parent=lst)

print(doc.export_to_markdown())
"""

    proc = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        "Subprocess failed under -O.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}\n"
    )
    assert "Item A" in proc.stdout
