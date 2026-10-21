# SPDX-License-Identifier: MIT

"""List existing source regions alongside Markdown parts, using synthetic Chinese text.

Run from the repository root: python examples/serialization_with_provenance.py
No conversion, OCR, model download or network access is needed. For a converted
document, pass its DoclingDocument to export_markdown_parts instead.

Page numbers are the supplied provenance page numbers, not printed page labels.
Each bbox keeps its coordinate origin; charspan belongs to the source item, not
the Markdown string. A part can refer to multiple items and multiple regions.
This example does not verify PDF text or locate arbitrary edited quotations.
"""

import json
from typing import Any

from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)


def export_markdown_parts(doc: DoclingDocument) -> list[dict[str, Any]]:
    """Keep item identity and every supplied provenance entry, including none."""
    serializer = MarkdownDocSerializer(
        doc=doc,
        # Opt in to page furniture explicitly; do not change library defaults.
        params=MarkdownParams(labels=set(DocItemLabel), layers=set(ContentLayer)),
    )
    return [
        {
            "text": part.text,
            "items": [
                {
                    "ref": item.self_ref,
                    "provenance": [
                        {
                            "page_no": prov.page_no,
                            "bbox": prov.bbox.model_dump(mode="json"),
                            "charspan": list(prov.charspan),
                        }
                        for prov in item.prov
                    ],
                }
                for item in part.get_unique_doc_items()
            ],
        }
        for part in serializer.get_parts()
        if part.text
    ]


if __name__ == "__main__":
    document = DoclingDocument(name="自制中文溯源示例")
    for page_no in (1, 2):
        document.add_page(page_no=page_no, size=Size(width=595, height=842))
        text = "测试金额50,000.00元。"
        document.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            prov=ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=50, t=100, r=500, b=120, coord_origin=CoordOrigin.TOPLEFT),
                charspan=(0, len(text)),
            ),
        )
    # Missing provenance remains empty; do not guess a page number.
    document.add_text(label=DocItemLabel.TEXT, text="未提供来源位置的片段。")
    print(json.dumps(export_markdown_parts(document), ensure_ascii=False, indent=2))
