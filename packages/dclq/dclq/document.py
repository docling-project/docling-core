"""DocLang loading and semantic search primitives."""

from __future__ import annotations

import copy
import hashlib
import re
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from lxml import etree

from docling_core.transforms.deserializer import (
    DocLangDocDeserializer,
    DocLangSourceMap,
    DocLangSourceTarget,
)
from docling_core.transforms.serializer.plain_text import PlainTextDocSerializer, PlainTextParams
from docling_core.types.doc import (
    CodeItem,
    ContentLayer,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    GroupItem,
    InlineGroup,
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

DOCLANG_NS = "https://www.doclang.ai/ns/v0"
NS = {"d": DOCLANG_NS}
SEMANTIC_ELEMENTS = {
    "caption",
    "code",
    "field_heading",
    "field_item",
    "field_region",
    "footnote",
    "formula",
    "group",
    "heading",
    "hint",
    "index",
    "key",
    "list",
    "page_footer",
    "page_header",
    "picture",
    "table",
    "text",
    "value",
}
SEMANTIC_INLINE_TYPES = {
    "caption": "caption",
    "code": "code",
    "field_heading": "field_heading",
    "formula": "formula",
    "heading": "heading",
    "hint": "field_hint",
    "key": "field_key",
    "page_footer": "page_footer",
    "page_header": "page_header",
    "text": "text",
    "value": "field_value",
}
XPATH_OPERATORS = {"and", "div", "mod", "or"}
MAX_XML_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024


class DclqError(ValueError):
    """Expected user-facing input or query error."""


@dataclass
class Unit:
    """One addressable semantic search unit."""

    target: DocLangSourceTarget
    logical_type: str
    text: str
    all_text: str
    xpaths: tuple[str, ...]
    doc_items: tuple[str, ...]
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
            raise DclqError(f"DocLang XML exceeds the {MAX_XML_BYTES}-byte limit")

        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
        try:
            raw_root = etree.fromstring(raw, parser=parser)
        except (etree.XMLSyntaxError, ValueError) as exc:
            raise DclqError(f"invalid DocLang XML: {exc}") from exc

        root_name = etree.QName(raw_root).localname
        namespace = etree.QName(raw_root).namespace
        if root_name != "doclang" or namespace not in {None, "", DOCLANG_NS}:
            raise DclqError(f"expected a DocLang root element, got {raw_root.tag!r}")

        if validate:
            from doclang.validation import validate as validate_doclang

            with tempfile.TemporaryDirectory(prefix="dclq-") as temporary:
                validation_path = Path(temporary) / "document.xml"
                validation_path.write_bytes(raw)
                try:
                    validate_doclang(validation_path, allow_empty_namespace=not namespace)
                except Exception as exc:
                    raise DclqError(str(exc)) from exc

        query_root = copy.deepcopy(raw_root)
        if not namespace:
            for element in query_root.iter():
                if isinstance(element.tag, str):
                    element.tag = f"{{{DOCLANG_NS}}}{etree.QName(element).localname}"
            etree.cleanup_namespaces(query_root, top_nsmap={None: DOCLANG_NS})

        try:
            raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DclqError("DocLang XML must be UTF-8") from exc

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
            document._hierarchize()
        except Exception as exc:
            raise DclqError(f"could not deserialize DocLang: {exc}") from exc

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
            return self.query_root.xpath(_normalize_xpath(xpath), namespaces=NS)
        except etree.XPathError as exc:
            raise DclqError(f"invalid XPath: {exc}") from exc

    def selected_paths(self, xpath: str) -> set[str]:
        result = self.evaluate_xpath(xpath)
        if not isinstance(result, list) or not result or not all(_is_element(node) for node in result):
            raise DclqError("XPath must select one or more elements")
        return {_canonical_xpath(node) for node in result}

    def target_item(self, target: DocLangSourceTarget) -> NodeItem | None:
        if target.item_ref is None:
            return None
        return RefItem(cref=target.item_ref).resolve(self.document)

    def unit_for_target(self, target: DocLangSourceTarget) -> Unit | None:
        return next((unit for unit in self.units if unit.target == target), None)

    def display_unit(self, unit: Unit) -> Unit:
        item = unit.item or self.target_item(unit.target)
        if item is None or not _covered_by_parent(item, self.document):
            return unit
        while item.parent is not None:
            item = item.parent.resolve(self.document)
            if parent := next(
                (candidate for candidate in self.context_units if candidate.item_ref == item.self_ref), None
            ):
                return parent
        return unit

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

    def nearest_heading_ref(self, unit: Unit) -> str | None:
        item = unit.item or self.target_item(unit.target)
        while item is not None:
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                return item.self_ref
            item = item.parent.resolve(self.document) if item.parent is not None else None
        return None

    def context_for(
        self,
        unit: Unit,
        before: int,
        after: int,
        scope: str,
        eligible: list[Unit],
        *,
        include_overlapping_before: bool = False,
    ) -> tuple[list[Unit], list[Unit], tuple[str, int, int] | None]:
        scoped, sequence = self._scope_units(unit, scope, eligible)
        indexes = [
            index
            for index, candidate in enumerate(scoped)
            if any(
                selected == context or selected.startswith(context + "/") or context.startswith(selected + "/")
                for selected in unit.xpaths
                for context in candidate.xpaths
            )
            or (
                unit.item_ref is not None
                and candidate.item_ref is not None
                and unit.item_ref != candidate.item_ref
                and candidate.item is not None
                and _covers_nested_items(candidate.item)
                and self.is_descendant(unit.item_ref, candidate.item_ref)
            )
        ]
        if not indexes:
            return [], [], None

        first, last = indexes[0], indexes[-1]
        before_end = last + 1 if include_overlapping_before else first
        start = max(0, first - before)
        end = min(len(scoped), last + 1 + after)
        return (
            scoped[start:before_end],
            scoped[last + 1 : end],
            (sequence, start, end - 1),
        )

    def inventory(self) -> dict[str, Any]:
        element_counts = Counter(
            _local_name(self.raw_elements[xpath])
            for xpath in self.source_map.targets_by_xpath
            if _local_name(self.raw_elements[xpath]) in SEMANTIC_ELEMENTS
        )
        metadata_counts = Counter(
            etree.QName(element).localname
            for element in self.raw_root.iter()
            if _is_element(element) and _is_metadata_element(element)
        )
        return {
            "document": self.name,
            "input_type": self.input_type,
            "page_count": len(self.document.pages),
            "semantic_units": len(self.context_units),
            "elements": dict(sorted(element_counts.items())),
            "metadata_elements": dict(sorted(metadata_counts.items())),
        }

    def _build_units(self) -> None:
        layers = set(ContentLayer)
        serializer = PlainTextDocSerializer(doc=self.document)
        visible_serializer = PlainTextDocSerializer(
            doc=self.document,
            params=PlainTextParams(allowed_meta_names=set(), include_annotations=False),
        )
        metadata_serializer = PlainTextDocSerializer(
            doc=self.document,
            params=PlainTextParams(include_non_meta=False),
        )
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
                logical_type = _logical_type(item, _local_name(self.raw_elements[xpaths[0]]))
                serialized = visible_serializer.serialize(item=item)
                serialized_all = serializer.serialize(item=item)
                serialized_metadata = metadata_serializer.serialize(item=item)
                unit = Unit(
                    target=target,
                    logical_type=logical_type,
                    text=serialized.text,
                    all_text=serialized_all.text,
                    xpaths=xpaths,
                    doc_items=tuple(dict.fromkeys(span.item.self_ref for span in serialized_all.spans)),
                    item_ref=item.self_ref,
                    page=_item_page(item) or self.source_pages[xpaths[0]],
                    pages=_unit_pages(item, xpaths, self.source_pages),
                    layer=item.content_layer.value,
                    item=item,
                    container=not isinstance(item, TextItem)
                    and not (isinstance(item, InlineGroup) and logical_type != "inline"),
                    metadata=serialized_metadata.text,
                )
                self.units.append(unit)
                if isinstance(item, (TextItem, InlineGroup)) and not _covered_by_parent(item, self.document):
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
                all_text=cell.text,
                xpaths=xpaths,
                doc_items=(container.self_ref,),
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

    def _scope_units(self, unit: Unit, scope: str, eligible: list[Unit]) -> tuple[list[Unit], str]:
        if scope == "document":
            return eligible, "document"

        item = unit.item or self.target_item(unit.target)
        if scope == "auto":
            if unit.target.kind == "table_cell" or isinstance(item, ListItem):
                scope = "container"
            else:
                scope = "section"
        if scope == "section":
            heading_ref = self.nearest_heading_ref(unit)
            if heading_ref is None:
                return eligible, "document"
            return (
                [candidate for candidate in eligible if self.is_descendant(candidate.item_ref, heading_ref)],
                f"section:{heading_ref}",
            )
        if scope == "container":
            if unit.target.kind == "table_cell":
                return (
                    [
                        candidate
                        for candidate in eligible
                        if candidate.item_ref == unit.item_ref and candidate.target.kind == "table_cell"
                    ],
                    f"table:{unit.item_ref}",
                )
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
            return siblings, f"container:{parent_ref}"
        raise DclqError(f"unknown context scope: {scope}")


def _read_source(source: str, *, stdin_bytes: bytes | None) -> tuple[bytes, str, tuple[str, ...]]:
    if source == "-":
        if stdin_bytes is None:
            raise DclqError("standard input was not provided")
        return stdin_bytes, "stdin", ()

    path = Path(source)
    if path.suffix.lower() == ".dclx":
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                for info in infos:
                    member = PurePosixPath(info.filename.replace("\\", "/"))
                    if member.is_absolute() or ".." in member.parts:
                        raise DclqError(f"unsafe archive member: {info.filename!r}")
                    if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                        raise DclqError(f"archive member exceeds the size limit: {info.filename!r}")
                if sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
                    raise DclqError("archive exceeds the total uncompressed size limit")
                documents = [info for info in infos if info.filename == "document.xml"]
                if len(documents) != 1:
                    raise DclqError("DocLang archive must contain exactly one document.xml")
                if documents[0].file_size > MAX_XML_BYTES:
                    raise DclqError(f"document.xml exceeds the {MAX_XML_BYTES}-byte limit")
                return archive.read(documents[0]), "dclx", tuple(info.filename for info in infos)
        except (OSError, zipfile.BadZipFile) as exc:
            raise DclqError(f"could not read DocLang archive: {exc}") from exc

    if path.suffix.lower() not in {".dclg", ".xml"}:
        raise DclqError(f"unsupported input type: {path.suffix or source!r}")
    try:
        return path.read_bytes(), path.suffix.lower().lstrip("."), ()
    except OSError as exc:
        raise DclqError(f"could not read {source}: {exc}") from exc


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


def _logical_type(item: NodeItem, source_type: str | None = None) -> str:
    if isinstance(item, InlineGroup):
        return SEMANTIC_INLINE_TYPES.get(source_type or "", "inline")
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
    assert isinstance(item, GroupItem)
    return item.label.value


def _covered_by_parent(item: NodeItem, document: DoclingDocument) -> bool:
    parent = item.parent.resolve(document) if item.parent is not None else None
    while parent is not None:
        if _covers_nested_items(parent):
            return True
        parent = parent.parent.resolve(document) if parent.parent is not None else None
    return False


def _covers_nested_items(item: NodeItem) -> bool:
    return isinstance(item, (InlineGroup, TableItem, ListItem)) or (
        isinstance(item, TextItem)
        and item.label in {DocItemLabel.FIELD_KEY, DocItemLabel.FIELD_VALUE, DocItemLabel.FIELD_HINT}
    )


def _is_metadata_element(element: etree._Element) -> bool:
    parent = element.getparent()
    while parent is not None:
        if _local_name(parent) in {"custom", "head"}:
            return _local_name(element) not in {"custom", "head"}
        parent = parent.getparent()
    return False


def _normalize_xpath(xpath: str) -> str:
    """Add the DocLang namespace and optional root to shorthand XPath."""

    def prefix_names(expression: str) -> str:
        def prefix_axis(match: re.Match[str]) -> str:
            after = expression[match.end() :].lstrip()
            return match.group() if after.startswith(("(", ":")) else f"::d:{match.group(1)}"

        expression = re.sub(r"::([A-Za-z_][\w.-]*)", prefix_axis, expression)

        def replace(match: re.Match[str]) -> str:
            name = match.group()
            before = expression[: match.start()].rstrip()
            after = expression[match.end() :].lstrip()
            if name in XPATH_OPERATORS or before.endswith(("@", "$")) or after.startswith(("(", "::")):
                return name
            return f"d:{name}"

        return re.sub(r"(?<![\w:.-])[A-Za-z_][\w.-]*(?![\w:.-])", replace, expression)

    parts = re.split(r"""((?:'[^']*')|(?:"[^"]*"))""", xpath)
    normalized = "".join(part if index % 2 else prefix_names(part) for index, part in enumerate(parts))
    return re.sub(
        r"(^|[(=,\s|])/(?!/|d:doclang(?:\[|/|\s|$))",
        r"\1/d:doclang/",
        normalized,
    )
