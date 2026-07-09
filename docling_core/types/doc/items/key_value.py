"""Key-value and form region items, with their graph data models."""

import typing
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field, field_validator

from docling_core.types.doc.common.reference import ProvenanceItem, RefItem
from docling_core.types.doc.items.node import FloatingItem
from docling_core.types.doc.labels import DocItemLabel, GraphCellLabel, GraphLinkLabel

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class GraphCell(BaseModel):
    """GraphCell."""

    label: GraphCellLabel

    cell_id: int

    text: str  # sanitized text
    orig: str  # text as seen on document

    prov: Optional[ProvenanceItem] = None

    # in case you have a text, table or picture item
    item_ref: Optional[RefItem] = None


class GraphLink(BaseModel):
    """GraphLink."""

    label: GraphLinkLabel

    source_cell_id: int
    target_cell_id: int


class GraphData(BaseModel):
    """GraphData."""

    cells: list[GraphCell] = Field(default_factory=list)
    links: list[GraphLink] = Field(default_factory=list)

    @field_validator("links")
    @classmethod
    def validate_links(cls, links, info):
        """Ensure that each link is valid."""
        cells = info.data.get("cells", [])

        valid_cell_ids = {cell.cell_id for cell in cells}

        for link in links:
            if link.source_cell_id not in valid_cell_ids:
                raise ValueError(f"Invalid source_cell_id {link.source_cell_id} in GraphLink")
            if link.target_cell_id not in valid_cell_ids:
                raise ValueError(f"Invalid target_cell_id {link.target_cell_id} in GraphLink")

        return links


class KeyValueItem(FloatingItem):
    """KeyValueItem."""

    label: typing.Literal[DocItemLabel.KEY_VALUE_REGION] = DocItemLabel.KEY_VALUE_REGION

    graph: GraphData

    def export_to_document_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        r"""Export key value item to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")  Deprecated
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)

        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        serializer = DocTagsDocSerializer(
            doc=doc,
            params=DocTagsParams(
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                add_content=add_content,
            ),
        )
        text = serializer.serialize(item=self).text
        return text


class FormItem(FloatingItem):
    """FormItem."""

    label: typing.Literal[DocItemLabel.FORM] = DocItemLabel.FORM

    graph: GraphData
