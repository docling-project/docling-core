#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

from docling_core.transforms.chunker.base import Chunk


def test_chunk_migration():
    input_path = "$.main-text[42]"  # deprected path format
    expected_path = "#/main-text/42"
    chunk = Chunk(
        path=input_path,
        text="foo",
    )
    assert chunk.path == expected_path
