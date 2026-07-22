"""DocLang loading and semantic search primitives."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from docling_core.transforms.deserializer import (
    DocLangDocDeserializer,
    DocLangSourceMap,
    DocLangSourceTarget,
)
from docling_core.types.doc import (
    CodeItem,
    ContentLayer,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    ListGroup,
    ListItem,
    NodeItem,
    PictureItem,
    RefItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from lxml import etree

DOCLANG_NS = "https://www.doclang.ai/ns/v0"
NS = {"d": DOCLANG_NS}
MAX_XML_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024


class DlgrepError(ValueError):
    """Expected user-facing input or query error."""


@dataclass
class Unit:
    """One addressable semantic search unit."""

    target: DocLangSourceTarget
    logical_type: str
    text: str
    xpaths: tuple[str, ...]
    item_ref: str | None
    page: int | None
    pages: tuple[int, ...]
    layer: str
    item: NodeItem | None = None
    row: int | None = None
    col: int | None = None
    container: bool = False
    metadata: str = ""

    @property
    def xpath(self) -> str:
        return self.xpaths[0]


@dataclass
class LoadedDocument:
    """The raw XML, semantic document, source map, and derived units for one input."""

    name: str
    input_type: str
    raw: bytes
    raw_root: etree._Element
    query_root: etree._Element
    document: DoclingDocument
    source_map: DocLangSourceMap
    members: tuple[str, ...] = ()
    units: list[Unit] = field(default_factory=list)
    context_units: list[Unit] = field(default_factory=list)
    raw_elements: dict[str, etree._Element] = field(default_factory=dict)
    source_pages: dict[str, int] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    @classmethod
    def load(
        cls,
        source: str,
        *,
        stdin_bytes: bytes | None = None,
        validate: bool = False,
    ) -> LoadedDocument:
        raw, input_type, members = _read_source(source, stdin_bytes=stdin_bytes)
        if len(raw) > MAX_XML_BYTES:
            raise DlgrepError(f"DocLang XML exceeds the {MAX_XML_BYTES}-byte limit")

        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
        try:
            raw_root = etree.fromstring(raw, parser=parser)
        except (etree.XMLSyntaxError, ValueError) as exc:
            raise DlgrepError(f"invalid DocLang XML: {exc}") from exc

        root_name = etree.QName(raw_root).localname
        namespace = etree.QName(raw_root).namespace
        if root_name != "doclang" or namespace not in {None, "", DOCLANG_NS}:
            raise DlgrepError(f"expected a DocLang root element, got {raw_root.tag!r}")

        if validate:
            from doclang.validation import validate as validate_doclang

            with tempfile.TemporaryDirectory(prefix="dlgrep-") as temporary:
                validation_path = Path(temporary) / "document.xml"
                validation_path.write_bytes(raw)
                try:
                    validate_doclang(validation_path, allow_empty_namespace=not namespace)
                except Exception as exc:
                    raise DlgrepError(str(exc)) from exc

        query_root = copy.deepcopy(raw_root)
        if not namespace:
            for element in query_root.iter():
                if isinstance(element.tag, str):
                    element.tag = f"{{{DOCLANG_NS}}}{etree.QName(element).localname}"
            etree.cleanup_namespaces(query_root, top_nsmap={None: DOCLANG_NS})

        try:
            raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DlgrepError("DocLang XML must be UTF-8") from exc

        semantic_root = copy.deepcopy(raw_root)
        for element in semantic_root.iter():
            if isinstance(element.tag, str):
                element.tag = etree.QName(element).localname
        etree.cleanup_namespaces(semantic_root)
        text = etree.tostring(semantic_root, encoding="unicode")

        source_map = DocLangSourceMap()
        try:
            document = DocLangDocDeserializer().deserialize_str(text, source_map=source_map)
            document.name = Path(source).stem if source != "-" else "stdin"
            document.hierarchize()
        except Exception as exc:
            raise DlgrepError(f"could not deserialize DocLang: {exc}") from exc

        loaded = cls(
            name=source,
            input_type=input_type,
            raw=raw,
            raw_root=raw_root,
            query_root=query_root,
            document=document,
            source_map=source_map,
            members=members,
        )
        page_no = 1
        for element in raw_root.iter():
            if not _is_element(element):
                continue
            if _local_name(element) == "page_break":
                page_no += 1
            xpath = _canonical_xpath(element)
            loaded.raw_elements[xpath] = element
            loaded.source_pages[xpath] = page_no
        loaded._build_units()
        return loaded

    def evaluate_xpath(self, xpath: str) -> Any:
        try:
            return self.query_root.xpath(xpath, namespaces=NS)
        except etree.XPathError as exc:
            raise DlgrepError(f"invalid XPath: {exc}") from exc

    def selected_paths(self, xpath: str) -> set[str]:
        result = self.evaluate_xpath(xpath)
        if not isinstance(result, list) or not result or not all(_is_element(node) for node in result):
            raise DlgrepError("XPath must select one or more elements")
        return {_canonical_xpath(node) for node in result}

    def target_item(self, target: DocLangSourceTarget) -> NodeItem | None:
        if target.item_ref is None:
            return None
        return RefItem(cref=target.item_ref).resolve(self.document)

    def unit_for_target(self, target: DocLangSourceTarget) -> Unit | None:
        return next((unit for unit in self.units if unit.target == target), None)

    def is_descendant(self, item_ref: str | None, ancestor_ref: str) -> bool:
        if item_ref is None:
            return False
        item = RefItem(cref=item_ref).resolve(self.document)
        while True:
            if item.self_ref == ancestor_ref:
                return True
            if item.parent is None:
                return False
            item = item.parent.resolve(self.document)

    def heading_chain(self, unit: Unit) -> list[str]:
        item = unit.item or self.target_item(unit.target)
        headings: list[str] = []
        while item is not None and item.parent is not None:
            item = item.parent.resolve(self.document)
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                headings.append(item.text)
        return list(reversed(headings))

    def nearest_heading_ref(self, unit: Unit) -> str | None:
        item = unit.item or self.target_item(unit.target)
        while item is not None:
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                return item.self_ref
            item = item.parent.resolve(self.document) if item.parent is not None else None
        return None

    def context_for(
        self, unit: Unit, before: int, after: int, scope: str, eligible: list[Unit]
    ) -> tuple[list[Unit], list[Unit]]:
        scoped = self._scope_units(unit, scope, eligible)
        try:
            index = scoped.index(unit)
        except ValueError:
            return [], []
        return scoped[max(0, index - before) : index], scoped[index + 1 : index + 1 + after]

    def inventory(self) -> dict[str, Any]:
        source_counts = Counter(
            etree.QName(element).localname for element in self.raw_root.iter() if _is_element(element)
        )
        heading_levels = Counter(
            element.get("level", "1")
            for element in self.raw_root.iter()
            if _is_element(element) and etree.QName(element).localname == "heading"
        )
        layers = Counter(unit.layer for unit in self.context_units)
        thread_elements = [element for element in self.raw_root.iter() if _local_name(element) == "thread"]
        return {
            "document": self.name,
            "input_type": self.input_type,
            "sha256": self.sha256,
            "namespace": etree.QName(self.raw_root).namespace or "",
            "version": self.raw_root.get("version"),
            "pages": len(self.document.pages),
            "page_images": sum(
                1
                for name in self.members
                if name.startswith("pages/") and Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ),
            "archive_assets": sum(1 for name in self.members if name.startswith("assets/") and not name.endswith("/")),
            "headings_by_level": dict(sorted(heading_levels.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "layers": dict(sorted(layers.items())),
            "threads": len({thread_id for element in thread_elements if (thread_id := element.get("thread_id"))}),
            "thread_fragments": len(thread_elements),
            "cross_references": sum(1 for element in self.raw_root.iter() if _local_name(element) == "xref"),
            "hyperlinks": sum(1 for element in self.raw_root.iter() if _local_name(element) == "href"),
            "source_map": {
                "bound_xpaths": len(self.source_map.targets_by_xpath),
                "semantic_units": len(self.context_units),
                "unbound_semantic_units": sum(1 for unit in self.context_units if not unit.xpaths),
            },
        }

    def _build_units(self) -> None:
        layers = set(ContentLayer)
        for item, _ in self.document.iterate_items(
            with_groups=True,
            traverse_pictures=True,
            included_content_layers=layers,
        ):
            if item.self_ref in {"#/body", "#/furniture"}:
                continue
            target = DocLangSourceTarget(kind="item", item_ref=item.self_ref)
            xpaths = tuple(self.source_map.xpaths_by_target.get(target, ()))
            if xpaths:
                unit = Unit(
                    target=target,
                    logical_type=_logical_type(item),
                    text=_item_text(item, self.document),
                    xpaths=xpaths,
                    item_ref=item.self_ref,
                    page=_item_page(item) or self.source_pages[xpaths[0]],
                    pages=_unit_pages(item, xpaths, self.source_pages),
                    layer=item.content_layer.value,
                    item=item,
                    container=not isinstance(item, TextItem),
                    metadata=_metadata_text(item),
                )
                self.units.append(unit)
                if isinstance(item, TextItem) and not _covered_by_parent(item, self.document):
                    self.context_units.append(unit)

            if isinstance(item, TableItem):
                self._add_cells(item, item.data.table_cells)
            elif isinstance(item, PictureItem) and item.meta is not None and item.meta.tabular_chart is not None:
                self._add_cells(item, item.meta.tabular_chart.chart_data.table_cells, logical_type="chart_cell")

    def _add_cells(self, container: NodeItem, cells: Iterable[Any], *, logical_type: str = "table_cell") -> None:
        for cell in sorted(cells, key=lambda value: (value.start_row_offset_idx, value.start_col_offset_idx)):
            target = DocLangSourceTarget(
                kind="table_cell",
                item_ref=container.self_ref,
                row=cell.start_row_offset_idx,
                col=cell.start_col_offset_idx,
            )
            xpaths = tuple(self.source_map.xpaths_by_target.get(target, ()))
            if not xpaths:
                continue
            pages = tuple(dict.fromkeys(self.source_pages[xpath] for xpath in xpaths))
            cell_type = (
                "index_cell"
                if isinstance(container, TableItem) and container.label == DocItemLabel.DOCUMENT_INDEX
                else logical_type
            )
            unit = Unit(
                target=target,
                logical_type=cell_type,
                text=cell.text,
                xpaths=xpaths,
                item_ref=container.self_ref,
                page=_item_page(container) or pages[0],
                pages=pages,
                layer=container.content_layer.value,
                item=container,
                row=cell.start_row_offset_idx,
                col=cell.start_col_offset_idx,
            )
            self.units.append(unit)
            self.context_units.append(unit)

    def _scope_units(self, unit: Unit, scope: str, eligible: list[Unit]) -> list[Unit]:
        if scope == "document":
            return eligible
        if scope == "page":
            return [candidate for candidate in eligible if set(candidate.pages) & set(unit.pages)]

        item = unit.item or self.target_item(unit.target)
        if scope == "auto":
            if unit.target.kind == "table_cell" or isinstance(item, ListItem):
                scope = "container"
            else:
                scope = "section"
        if scope == "section":
            heading_ref = self.nearest_heading_ref(unit)
            if heading_ref is None:
                return eligible
            return [candidate for candidate in eligible if self.is_descendant(candidate.item_ref, heading_ref)]
        if scope == "container":
            if unit.target.kind == "table_cell":
                return [
                    candidate
                    for candidate in eligible
                    if candidate.item_ref == unit.item_ref and candidate.target.kind == "table_cell"
                ]
            parent_ref = item.parent.cref if item is not None and item.parent is not None else None
            siblings: list[Unit] = []
            for candidate in eligible:
                candidate_item = candidate.item or self.target_item(candidate.target)
                if (
                    candidate_item is not None
                    and candidate_item.parent is not None
                    and candidate_item.parent.cref == parent_ref
                ):
                    siblings.append(candidate)
            return siblings
        raise DlgrepError(f"unknown context scope: {scope}")


def _read_source(source: str, *, stdin_bytes: bytes | None) -> tuple[bytes, str, tuple[str, ...]]:
    if source == "-":
        if stdin_bytes is None:
            raise DlgrepError("standard input was not provided")
        return stdin_bytes, "stdin", ()

    path = Path(source)
    if path.suffix.lower() == ".dclx":
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                for info in infos:
                    member = PurePosixPath(info.filename.replace("\\", "/"))
                    if member.is_absolute() or ".." in member.parts:
                        raise DlgrepError(f"unsafe archive member: {info.filename!r}")
                    if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                        raise DlgrepError(f"archive member exceeds the size limit: {info.filename!r}")
                if sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
                    raise DlgrepError("archive exceeds the total uncompressed size limit")
                documents = [info for info in infos if info.filename == "document.xml"]
                if len(documents) != 1:
                    raise DlgrepError("DocLang archive must contain exactly one document.xml")
                if documents[0].file_size > MAX_XML_BYTES:
                    raise DlgrepError(f"document.xml exceeds the {MAX_XML_BYTES}-byte limit")
                return archive.read(documents[0]), "dclx", tuple(info.filename for info in infos)
        except (OSError, zipfile.BadZipFile) as exc:
            raise DlgrepError(f"could not read DocLang archive: {exc}") from exc

    if path.suffix.lower() not in {".dclg", ".xml"}:
        raise DlgrepError(f"unsupported input type: {path.suffix or source!r}")
    try:
        return path.read_bytes(), path.suffix.lower().lstrip("."), ()
    except OSError as exc:
        raise DlgrepError(f"could not read {source}: {exc}") from exc


def _canonical_xpath(element: etree._Element) -> str:
    parts: list[str] = []
    current: etree._Element | None = element
    while current is not None and _is_element(current):
        name = etree.QName(current).localname
        parent = current.getparent()
        if parent is None:
            parts.append(f"d:{name}")
        else:
            siblings = [child for child in parent if _is_element(child) and etree.QName(child).localname == name]
            parts.append(f"d:{name}[{siblings.index(current) + 1}]")
        current = parent
    return "/" + "/".join(reversed(parts))


def _is_element(value: Any) -> bool:
    return isinstance(value, etree._Element) and isinstance(value.tag, str)


def _local_name(value: Any) -> str | None:
    return etree.QName(value).localname if _is_element(value) else None


def _item_page(item: NodeItem) -> int | None:
    return item.prov[0].page_no if isinstance(item, DocItem) and item.prov else None


def _unit_pages(item: NodeItem, xpaths: tuple[str, ...], source_pages: dict[str, int]) -> tuple[int, ...]:
    pages = [prov.page_no for prov in item.prov] if isinstance(item, DocItem) else []
    pages.extend(source_pages[xpath] for xpath in xpaths)
    return tuple(dict.fromkeys(pages))


def _logical_type(item: NodeItem) -> str:
    if isinstance(item, (TitleItem, SectionHeaderItem)):
        return "heading"
    if isinstance(item, ListGroup):
        return "list"
    if isinstance(item, TableItem) and item.label == DocItemLabel.DOCUMENT_INDEX:
        return "index"
    if isinstance(item, CodeItem):
        return "code"
    if isinstance(item, DocItem):
        return item.label.value
    return item.label.value


def _item_text(item: NodeItem, document: DoclingDocument) -> str:
    if isinstance(item, (TitleItem, SectionHeaderItem, CodeItem)):
        return item.text
    if isinstance(item, TextItem) and item.label == DocItemLabel.CHECKBOX_SELECTED:
        return "[x]"
    if isinstance(item, TextItem) and item.label == DocItemLabel.CHECKBOX_UNSELECTED:
        return "[ ]"
    if isinstance(item, TextItem) and not item.children:
        return f"{item.marker} {item.text}".strip() if isinstance(item, ListItem) and item.marker else item.text
    if isinstance(item, TableItem):
        return "\n".join(cell.text for cell in item.data.table_cells if cell.text)

    parts: list[str] = []
    if isinstance(item, TextItem) and item.text:
        parts.append(item.text)
    for child_ref in item.children:
        child = child_ref.resolve(document)
        text = _item_text(child, document)
        if text:
            parts.append(text)
    text = "\n".join(parts)
    if isinstance(item, ListItem) and item.marker:
        return f"{item.marker} {text}".strip()
    return text


def _covered_by_parent(item: TextItem, document: DoclingDocument) -> bool:
    parent = item.parent.resolve(document) if item.parent is not None else None
    while parent is not None:
        if isinstance(parent, (TableItem, ListItem)):
            return True
        if isinstance(parent, TextItem) and parent.label in {
            DocItemLabel.FIELD_KEY,
            DocItemLabel.FIELD_VALUE,
            DocItemLabel.FIELD_HINT,
        }:
            return True
        parent = parent.parent.resolve(document) if parent.parent is not None else None
    return False


def _metadata_text(item: NodeItem) -> str:
    if not isinstance(item, DocItem) or item.meta is None:
        return ""
    return "\n".join(_strings(item.meta.model_dump(mode="json")))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value:
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)
