"""Optional source bindings collected while deserializing DocLang."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, Optional
from xml.dom.minidom import Element, Node

from docling_core.transforms.serializer._doclang_utils import DocLangToken
from docling_core.types.doc import NodeItem, TableData

__all__ = ["DocLangSourceMap", "DocLangSourceTarget"]


@dataclass(frozen=True)
class DocLangSourceTarget:
    """Semantic target created from one DocLang source element."""

    kind: Literal["item", "table_cell", "page"]
    item_ref: Optional[str] = None
    row: Optional[int] = None
    col: Optional[int] = None
    page_no: Optional[int] = None


@dataclass
class DocLangSourceMap:
    """Bidirectional XPath-to-semantic bindings collected during deserialization."""

    targets_by_xpath: dict[str, DocLangSourceTarget] = field(default_factory=dict)
    xpaths_by_target: dict[DocLangSourceTarget, list[str]] = field(default_factory=dict)

    def bind(self, xpath: str, target: DocLangSourceTarget) -> None:
        """Bind a canonical DocLang XPath to a semantic target."""
        previous = self.targets_by_xpath.get(xpath)
        if previous == target:
            return
        if previous is not None:
            self.xpaths_by_target[previous].remove(xpath)
        self.targets_by_xpath[xpath] = target
        self.xpaths_by_target.setdefault(target, []).append(xpath)


_TABLE_CELL_MARKER_TAGS = frozenset(
    {
        DocLangToken.FCEL.value,
        DocLangToken.ECEL.value,
        DocLangToken.LCEL.value,
        DocLangToken.UCEL.value,
        DocLangToken.XCEL.value,
        DocLangToken.CHED.value,
        DocLangToken.RHED.value,
        DocLangToken.SROW.value,
        DocLangToken.CORN.value,
    }
)


@dataclass
class _DocLangSourceRecorder:
    """Optional sidecar that records source bindings without affecting parsing."""

    source_map: Optional[DocLangSourceMap] = None
    _suspended: bool = False

    @staticmethod
    def _xpath(el: Element) -> str:
        parts: list[str] = []
        current: Optional[Node] = el
        while isinstance(current, Element):
            name = current.localName or current.tagName.rsplit(":", maxsplit=1)[-1]
            parent = current.parentNode
            if isinstance(parent, Element):
                siblings = [
                    child
                    for child in parent.childNodes
                    if isinstance(child, Element)
                    and (child.localName or child.tagName.rsplit(":", maxsplit=1)[-1]) == name
                ]
                parts.append(f"d:{name}[{siblings.index(current) + 1}]")
            else:
                parts.append(f"d:{name}")
            current = parent
        return "/" + "/".join(reversed(parts))

    def bind_item(self, el: Element, item: NodeItem) -> None:
        if self.source_map is not None and not self._suspended:
            self.source_map.bind(
                self._xpath(el),
                DocLangSourceTarget(kind="item", item_ref=item.self_ref),
            )

    def bind_page(self, el: Element, page_no: int) -> None:
        if self.source_map is not None and not self._suspended:
            self.source_map.bind(
                self._xpath(el),
                DocLangSourceTarget(kind="page", page_no=page_no),
            )

    def bind_table_cells(
        self,
        el: Element,
        body_nodes: Sequence[Node],
        data: TableData,
        item_ref: str,
        *,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> None:
        if self.source_map is None or self._suspended:
            return
        row = 0
        col = 0
        xpath_by_position: dict[tuple[int, int], str] = {}
        for node in body_nodes:
            if not isinstance(node, Element):
                continue
            if node.tagName == DocLangToken.NL.value:
                row += 1
                col = 0
            elif node.tagName in _TABLE_CELL_MARKER_TAGS:
                xpath_by_position[(row + row_offset, col + col_offset)] = self._xpath(node)
                col += 1
        for cell in data.table_cells:
            position = (cell.start_row_offset_idx, cell.start_col_offset_idx)
            if xpath := xpath_by_position.get(position):
                self.source_map.bind(
                    xpath,
                    DocLangSourceTarget(
                        kind="table_cell",
                        item_ref=item_ref,
                        row=position[0],
                        col=position[1],
                    ),
                )

    @contextmanager
    def suspend(self) -> Iterator[None]:
        """Temporarily suppress bindings for semantic children of one source unit."""
        previous = self._suspended
        self._suspended = True
        try:
            yield
        finally:
            self._suspended = previous
