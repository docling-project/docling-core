from pathlib import Path

from docling_core.experimental.serializer.outline import (
    OutlineDocSerializer,
    OutlineMode,
    OutlineParams,
)
from docling_core.types.doc import DoclingDocument


def test_outline_serializer_basic():
    # src = Path("test/data/doc/2408.09869.json")
    # src = Path("test/data/doc/2501.17887v1.json")
    # src = Path("test/data/doc/2106.09680v1.json")
    # src = Path("test/data/doc/2408.09869v3_enriched.json")

    for src in [
        Path("test/data/doc/2501.17887v1.json"),
        Path("test/data/doc/2106.09680v1.json"),
        Path("test/data/doc/2408.09869v3_enriched.json"),
    ]:
        doc = DoclingDocument.load_from_json(filename=src)

        # print("\n\nMARKDOWN: \n\n")
        # print(doc.export_to_markdown())

        # Only serialize metadata to focus on outline-like content
        params = OutlineParams(
            include_non_meta=True, mode=OutlineMode.TABLE_OF_CONTENTS
        )
        ser = OutlineDocSerializer(doc=doc, params=params)

        print("===========================================")
        res = ser.serialize()
        actual = res.text

        print("\n\nSUMMARY: \n\n")
        print(actual)

    assert isinstance(actual, str)
    # Expect summaries from title and section header to appear
    # assert "This is a title." in actual
    # assert "This is a section header." in actual
