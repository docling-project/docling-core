"""Define classes for DocLang deserialization.

Aligned to the DocLang specification version ``_DOCLANG_VERSION``.
"""

from collections.abc import Callable, Sequence
from itertools import groupby
from typing import Any, ClassVar, Literal, Optional, cast
from xml.dom.minidom import Element, Node, Text

from defusedxml.minidom import parseString
from pydantic import BaseModel, PrivateAttr
from typing_extensions import override

from docling_core.transforms.deserializer.base import BaseDocDeserializer
from docling_core.transforms.serializer._doclang_utils import (
    _DOCLANG_META_TAG_DESCRIPTION,
    _DOCLANG_META_TAG_SMILES,
    _DOCLANG_META_TAG_SUMMARY,
    _ELEMENT_HEAD_TAGS,
    DOCLANG_DFLT_RESOLUTION,
    DocLangAttributeKey,
    DocLangAttributeValue,
    DocLangCategory,
    DocLangToken,
    DocLangVocabulary,
    _append_textual_fragment,
    _code_language_label_from_doclang,
    _merge_table_data,
    _picture_classification_label_from_doclang,
    _provenance_with_charspan,
    _thread_table_merge_offset,
    _wrap,
    _xml_error_context,
)
from docling_core.types.doc import (
    BaseMeta,
    BoundingBox,
    CodeItem,
    ContentLayer,
    DescriptionMetaField,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    FloatingItem,
    FloatingMeta,
    Formatting,
    FormItem,
    GroupItem,
    InlineGroup,
    KeyValueItem,
    ListGroup,
    ListItem,
    MetaUtils,
    MoleculeMetaField,
    NodeItem,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureItem,
    PictureMeta,
    ProvenanceItem,
    Script,
    SectionHeaderItem,
    Size,
    SummaryMetaField,
    TableCell,
    TableData,
    TableItem,
    TabularChartMetaField,
    TextItem,
)
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.document import (
    FieldHeadingItem,
    FieldItem,
    FieldRegionItem,
    FieldValueItem,
    FormulaItem,
    RichTableCell,
    TitleItem,
)
from docling_core.types.doc.document import (
    GroupItem as GroupItemType,
)
from docling_core.types.doc.labels import CodeLanguageLabel, GroupLabel

__all__ = ["DocLangDocDeserializer"]


class DocLangDocDeserializer(BaseDocDeserializer, BaseModel):
    """DocLang deserializer."""

    # Internal state used while walking the tree (private instance attributes)
    _page_no: int = PrivateAttr(default=1)
    _default_resolution: int = PrivateAttr(default=DOCLANG_DFLT_RESOLUTION)
    _thread_registry: dict[tuple[str, str], NodeItem] = PrivateAttr(default_factory=dict)

    def _thread_registry_key(self, *, thread_id: str, host: str) -> tuple[str, str]:
        return (thread_id, host)

    @override
    def deserialize_str(self, text: str, **kwargs: Any) -> DoclingDocument:
        """Deserialize DocLang XML into a DoclingDocument.

        Args:
            text: DocLang XML string to parse.
            page_no: Starting page number (default 1), passed via ``kwargs``.

        Returns:
            A populated `DoclingDocument` parsed from the input.
        """
        page_no: int = kwargs.get("page_no", 1)
        try:
            root_node = parseString(text).documentElement
        except Exception as e:
            ctx = _xml_error_context(text, e)
            raise ValueError(f"Invalid DocLang XML: {e}\n--- XML context ---\n{ctx}") from e
        if root_node is None:
            raise ValueError("Invalid DocLang XML: missing documentElement")
        root: Element = cast(Element, root_node)
        if root.tagName != DocLangToken.DOCUMENT.value:
            candidates = root.getElementsByTagName(DocLangToken.DOCUMENT.value)
            if candidates:
                root = cast(Element, candidates[0])

        doc = DoclingDocument(name="Document")
        self._page_no = page_no
        self._default_resolution = DOCLANG_DFLT_RESOLUTION
        self._thread_registry = {}
        self._ensure_page_exists(doc=doc, page_no=self._page_no, resolution=self._default_resolution)
        self._parse_document_root(doc=doc, root=root)
        return doc

    def _extract_thread_id_from_nodes(self, nodes: Sequence[Node]) -> Optional[str]:
        """Read ``thread_id`` from a ``<thread/>`` element in a node sequence."""
        for node in nodes:
            if isinstance(node, Element) and node.tagName == DocLangToken.THREAD.value:
                thread_id = node.getAttribute(DocLangAttributeKey.THREAD_ID.value)
                if thread_id:
                    return thread_id
        return None

    def _extract_thread_id(self, el: Element) -> Optional[str]:
        """Read ``thread_id`` from an element head."""
        head_nodes, _ = self._split_element_children_head_body(el)
        return self._extract_thread_id_from_nodes(head_nodes)

    def _register_thread(self, *, thread_id: str, host: str, item: NodeItem) -> None:
        self._thread_registry[self._thread_registry_key(thread_id=thread_id, host=host)] = item

    def _get_thread_item(self, thread_id: str, *, host: str) -> Optional[NodeItem]:
        return self._thread_registry.get(self._thread_registry_key(thread_id=thread_id, host=host))

    def _advance_page_break(self, *, doc: DoclingDocument) -> None:
        self._page_no += 1
        self._ensure_page_exists(doc=doc, page_no=self._page_no, resolution=self._default_resolution)

    def _provenance_from_nodes_with_page_breaks(
        self,
        *,
        doc: DoclingDocument,
        nodes: Sequence[Node],
    ) -> list[ProvenanceItem]:
        """Collect provenance quartets, advancing ``_page_no`` at ``<page_break/>``."""
        provs: list[ProvenanceItem] = []
        batch: list[Node] = []
        for node in nodes:
            if isinstance(node, Element) and node.tagName == DocLangToken.PAGE_BREAK.value:
                provs.extend(self._provenance_from_location_nodes(doc=doc, nodes=batch))
                batch = []
                self._advance_page_break(doc=doc)
            elif isinstance(node, Element) and node.tagName == DocLangToken.LOCATION.value:
                batch.append(node)
        provs.extend(self._provenance_from_location_nodes(doc=doc, nodes=batch))
        return provs

    def _virtual_text_from_nodes_with_page_breaks(self, nodes: Sequence[Node]) -> str:
        """Extract virtual-text payload, ignoring element-head tokens and page breaks."""
        parts: list[str] = []
        for node in nodes:
            if isinstance(node, Element) and node.tagName == DocLangToken.PAGE_BREAK.value:
                continue
            if isinstance(node, Element) and self._is_element_head_tag(node):
                continue
            if isinstance(node, Text):
                if not node.data.strip():
                    continue
                parts.append(node.data)
            elif isinstance(node, Element) and node.tagName == DocLangToken.CONTENT.value:
                parts.append(self._get_text(node))
        return "".join(parts)

    def _merge_threaded_text_item(
        self,
        *,
        text: str,
        prov_list: list[ProvenanceItem],
        existing: TextItem,
    ) -> TextItem:
        _append_textual_fragment(existing, text=text, prov_list=prov_list)
        return existing

    def _apply_initial_text_provenance(
        self,
        item: TextItem,
        *,
        text: str,
        prov_list: list[ProvenanceItem],
    ) -> None:
        if not prov_list:
            return
        item.prov = _provenance_with_charspan(prov_list[:1], (0, len(text)))
        for prov in prov_list[1:]:
            item.prov.append(prov)

    # ------------- Core walkers -------------
    def _parse_document_root(self, *, doc: DoclingDocument, root: Element) -> None:
        for node in root.childNodes:
            if isinstance(node, Element):
                self._dispatch_element(doc=doc, el=node, parent=None)

    def _dispatch_element(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        name = el.tagName
        if name in {
            DocLangToken.TEXT.value,
            DocLangToken.CAPTION.value,
            DocLangToken.FOOTNOTE.value,
            DocLangToken.PAGE_HEADER.value,
            DocLangToken.PAGE_FOOTER.value,
            DocLangToken.CODE.value,
            DocLangToken.FORMULA.value,
            DocLangToken.LDIV.value,
            DocLangToken.BOLD.value,
            DocLangToken.ITALIC.value,
            DocLangToken.UNDERLINE.value,
            DocLangToken.STRIKETHROUGH.value,
            DocLangToken.SUBSCRIPT.value,
            DocLangToken.SUPERSCRIPT.value,
            DocLangToken.CONTENT.value,
        }:
            self._parse_text_like(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.PAGE_BREAK.value:
            # Start a new page; keep a default square page using the configured resolution
            self._page_no += 1
            self._ensure_page_exists(doc=doc, page_no=self._page_no, resolution=self._default_resolution)
        elif name == DocLangToken.HEADING.value:
            self._parse_heading(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.FIELD_HEADING.value:
            self._parse_field_heading(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.FIELD_REGION.value:
            self._parse_field_region(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.FIELD_ITEM.value:
            self._parse_field_item(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.FIELD_KEY.value:
            self._parse_field_key(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.FIELD_VALUE.value:
            self._parse_field_value(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.FIELD_HINT.value:
            self._parse_field_hint(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.CHECKBOX.value:
            self._parse_checkbox(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.LIST.value:
            self._parse_list(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.GROUP.value:
            # Float + footnote siblings: parse as one unit (not a Docling GroupItem).
            if self._first_child(el, DocLangToken.TABLE.value) or self._first_child(el, DocLangToken.INDEX.value):
                self._parse_table(doc=doc, el=el, parent=parent)
            elif self._first_child(el, DocLangToken.PICTURE.value):
                self._parse_picture(doc=doc, el=el, parent=parent)
            else:
                self._walk_children(doc=doc, el=el, parent=parent)
        elif name in {DocLangToken.TABLE.value, DocLangToken.INDEX.value}:
            self._parse_table(doc=doc, el=el, parent=parent)
        elif name == DocLangToken.PICTURE.value:
            self._parse_picture(doc=doc, el=el, parent=parent)
        else:
            self._walk_children(doc=doc, el=el, parent=parent)

    def _walk_children(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        for node in el.childNodes:
            if isinstance(node, Element):
                # Ignore geometry/meta containers at this level; pass through page breaks
                if node.tagName in {
                    DocLangToken.HEAD.value,
                    DocLangToken.LOCATION.value,
                    DocLangToken.LAYER.value,
                    DocLangToken.LABEL.value,
                    DocLangToken.CUSTOM.value,
                    DocLangToken.CAPTION.value,
                    DocLangToken.SRC.value,
                }:
                    continue
                self._dispatch_element(doc=doc, el=node, parent=parent)

    # ------------- Text blocks -------------

    def _should_preserve_space(self, el: Element) -> bool:
        return el.tagName == DocLangToken.CONTENT.value  # and el.getAttribute("xml:space") == "preserve"

    def _get_children_simple_text_block(self, element: Element) -> Optional[str]:
        result = None
        for el in element.childNodes:
            if isinstance(el, Element):
                if self._is_element_head_tag(el):
                    continue
                if el.tagName not in {
                    DocLangToken.LOCATION.value,
                    DocLangToken.LAYER.value,
                    DocLangToken.LABEL.value,
                    DocLangToken.BR.value,
                    DocLangToken.BOLD.value,
                    DocLangToken.ITALIC.value,
                    DocLangToken.UNDERLINE.value,
                    DocLangToken.STRIKETHROUGH.value,
                    DocLangToken.SUBSCRIPT.value,
                    DocLangToken.SUPERSCRIPT.value,
                    DocLangToken.RTL.value,
                    DocLangToken.HANDWRITING.value,
                    DocLangToken.CHECKBOX.value,
                    DocLangToken.CONTENT.value,
                }:
                    return None
                elif tmp := self._get_children_simple_text_block(el):
                    result = tmp
            elif isinstance(el, Text) and el.data.strip():  # TODO should still support whitespace-only
                if result is None:
                    result = el.data if element.tagName == DocLangToken.CONTENT.value else el.data.strip()
                else:
                    return None
        return result

    def _parse_text_like(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        """Parse text-like tokens (text, caption, footnotes, code, formula)."""
        element_children = [
            node for node in el.childNodes if isinstance(node, Element) and not self._is_element_head_tag(node)
        ]

        thread_id = self._extract_thread_id(el)
        simple_text = self._get_children_simple_text_block(el)
        if len(element_children) > 1 or (simple_text is None and thread_id is None):
            self._parse_inline_group(doc=doc, el=el, parent=parent)
            return

        prov_list = self._extract_provenance(doc=doc, el=el)
        content_layer = self._extract_layer(el=el)
        text, formatting = self._extract_text_with_formatting(el)
        if not text:
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=el.tagName)) is not None
                and isinstance(existing, TextItem)
            ):
                if prov_list:
                    self._merge_threaded_text_item(text="", prov_list=prov_list, existing=existing)
            return

        nm = el.tagName

        # Handle code separately (language + content extraction)
        if nm == DocLangToken.CODE.value:
            code_text, lang_label = self._extract_code_content_and_language(el)
            if not code_text.strip():
                return
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=nm)) is not None
                and isinstance(existing, CodeItem)
            ):
                self._merge_threaded_text_item(text=code_text, prov_list=prov_list, existing=existing)
                return
            item = doc.add_code(
                text=code_text,
                code_language=lang_label,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                content_layer=content_layer,
            )
            self._apply_initial_text_provenance(item, text=code_text, prov_list=prov_list)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=nm, item=item)
            self._apply_custom_meta_from_element(item=item, el=el)

        # Map text-like tokens to text item labels
        elif nm in (
            text_label_map := {
                DocLangToken.TEXT.value: DocItemLabel.TEXT,
                DocLangToken.CAPTION.value: DocItemLabel.CAPTION,
                DocLangToken.FOOTNOTE.value: DocItemLabel.FOOTNOTE,
                DocLangToken.PAGE_HEADER.value: DocItemLabel.PAGE_HEADER,
                DocLangToken.PAGE_FOOTER.value: DocItemLabel.PAGE_FOOTER,
                DocLangToken.BOLD.value: DocItemLabel.TEXT,
                DocLangToken.ITALIC.value: DocItemLabel.TEXT,
                DocLangToken.UNDERLINE.value: DocItemLabel.TEXT,
                DocLangToken.STRIKETHROUGH.value: DocItemLabel.TEXT,
                DocLangToken.SUBSCRIPT.value: DocItemLabel.TEXT,
                DocLangToken.SUPERSCRIPT.value: DocItemLabel.TEXT,
                DocLangToken.RTL.value: DocItemLabel.TEXT,
                DocLangToken.CONTENT.value: DocItemLabel.TEXT,
            }
        ):
            is_bold = nm == DocLangToken.BOLD.value
            is_italic = nm == DocLangToken.ITALIC.value
            is_underline = nm == DocLangToken.UNDERLINE.value
            is_strikethrough = nm == DocLangToken.STRIKETHROUGH.value
            is_subscript = nm == DocLangToken.SUBSCRIPT.value
            is_superscript = nm == DocLangToken.SUPERSCRIPT.value

            if is_bold or is_italic or is_underline or is_strikethrough or is_subscript or is_superscript:
                formatting = formatting or Formatting()
                if is_bold:
                    formatting.bold = True
                elif is_italic:
                    formatting.italic = True
                elif is_underline:
                    formatting.underline = True
                elif is_strikethrough:
                    formatting.strikethrough = True
                elif is_subscript:
                    formatting.script = Script.SUB
                elif is_superscript:
                    formatting.script = Script.SUPER
            label = text_label_map[nm]
            if nm == DocLangToken.TEXT.value and any(
                c.tagName == DocLangToken.HANDWRITING.value for c in element_children
            ):
                label = DocItemLabel.HANDWRITTEN_TEXT
            elif nm == DocLangToken.TEXT.value:
                # Check for checkbox elements with class attribute
                for c in element_children:
                    if c.tagName == DocLangToken.CHECKBOX.value:
                        checkbox_class = c.getAttribute(DocLangAttributeKey.CLASS.value)
                        if checkbox_class == DocLangAttributeValue.SELECTED.value:
                            label = DocItemLabel.CHECKBOX_SELECTED
                            break
                        elif checkbox_class == DocLangAttributeValue.UNSELECTED.value:
                            label = DocItemLabel.CHECKBOX_UNSELECTED
                            break
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=nm)) is not None
                and isinstance(existing, TextItem)
            ):
                self._merge_threaded_text_item(text=text, prov_list=prov_list, existing=existing)
                return
            item = doc.add_text(
                label=label,
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
                content_layer=content_layer,
            )
            self._apply_initial_text_provenance(item, text=text, prov_list=prov_list)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=nm, item=item)
            self._apply_custom_meta_from_element(item=item, el=el)

        elif nm == DocLangToken.FORMULA.value:
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=nm)) is not None
                and isinstance(existing, FormulaItem)
            ):
                self._merge_threaded_text_item(text=text, prov_list=prov_list, existing=existing)
                return
            item = doc.add_formula(
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
            )
            self._apply_initial_text_provenance(item, text=text, prov_list=prov_list)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=nm, item=item)
            self._apply_custom_meta_from_element(item=item, el=el)

    def _extract_code_content_and_language(self, el: Element) -> tuple[str, CodeLanguageLabel]:
        """Extract code content and language from a <code> element."""
        lang_label = CodeLanguageLabel.UNKNOWN
        for node in el.childNodes:
            if isinstance(node, Element) and node.tagName == DocLangToken.LABEL.value:
                label_val = node.getAttribute(DocLangAttributeKey.VALUE.value)
                if label_val:
                    lang_label = _code_language_label_from_doclang(label_val)
                break
        parts: list[str] = []
        for node in el.childNodes:
            if isinstance(node, Text):
                if node.data.strip():
                    parts.append(node.data)
            elif isinstance(node, Element):
                nm_child = node.tagName
                if nm_child in {
                    DocLangToken.LOCATION.value,
                    DocLangToken.LAYER.value,
                    DocLangToken.LABEL.value,
                }:
                    continue
                elif nm_child == DocLangToken.BR.value:
                    parts.append("\n")
                else:
                    parts.append(self._get_text(node))

        return "".join(parts), lang_label

    def _parse_heading(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        lvl_txt = el.getAttribute(DocLangAttributeKey.LEVEL.value) or "1"
        try:
            level = int(lvl_txt)
        except Exception:
            level = 1
        # Extract provenance from heading token (if any)
        prov_list = self._extract_provenance(doc=doc, el=el)
        content_layer = self._extract_layer(el=el)
        text = self._get_text(el)
        text_stripped = text.strip()
        if text_stripped:
            thread_id = self._extract_thread_id(el)
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=DocLangToken.HEADING.value)) is not None
                and isinstance(existing, TextItem)
            ):
                self._merge_threaded_text_item(text=text_stripped, prov_list=prov_list, existing=existing)
                return
            # Level 1 maps to TitleItem, level > 1 maps to SectionHeaderItem with level-1
            if level == 1:
                item = doc.add_title(
                    text=text_stripped,
                    parent=parent,
                    prov=(prov_list[0] if prov_list else None),
                    content_layer=content_layer,
                )
            else:
                item = doc.add_heading(
                    text=text_stripped,
                    level=level - 1,
                    parent=parent,
                    prov=(prov_list[0] if prov_list else None),
                    content_layer=content_layer,
                )
            self._apply_initial_text_provenance(item, text=text_stripped, prov_list=prov_list)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=DocLangToken.HEADING.value, item=item)
            self._apply_custom_meta_from_element(item=item, el=el)

    def _parse_field_heading(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        lvl_txt = el.getAttribute(DocLangAttributeKey.LEVEL.value) or "1"
        try:
            level = int(lvl_txt)
        except Exception:
            level = 1
        prov_list = self._extract_provenance(doc=doc, el=el)
        content_layer = self._extract_layer(el=el)
        text = self._get_text(el)
        text_stripped = text.strip()
        if text_stripped:
            thread_id = self._extract_thread_id(el)
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=DocLangToken.FIELD_HEADING.value)) is not None
                and isinstance(existing, TextItem)
            ):
                self._merge_threaded_text_item(text=text_stripped, prov_list=prov_list, existing=existing)
                return
            item = doc.add_field_heading(
                text=text_stripped,
                level=level,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                content_layer=content_layer,
            )
            self._apply_initial_text_provenance(item, text=text_stripped, prov_list=prov_list)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=DocLangToken.FIELD_HEADING.value, item=item)

    _FIELD_INLINE_BODY_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            DocLangToken.CONTENT.value,
            DocLangToken.BOLD.value,
            DocLangToken.ITALIC.value,
            DocLangToken.UNDERLINE.value,
            DocLangToken.STRIKETHROUGH.value,
            DocLangToken.SUBSCRIPT.value,
            DocLangToken.SUPERSCRIPT.value,
            DocLangToken.HANDWRITING.value,
            DocLangToken.RTL.value,
            DocLangToken.BR.value,
            DocLangToken.CHECKBOX.value,
            DocLangToken.FIELD_HINT.value,
        }
    )

    def _dispatch_body_nodes(
        self,
        *,
        doc: DoclingDocument,
        body_nodes: Sequence[Node],
        parent: NodeItem,
    ) -> None:
        """Dispatch element-body children under ``parent``."""
        for node in body_nodes:
            if isinstance(node, Element):
                self._dispatch_element(doc=doc, el=node, parent=parent)
            elif isinstance(node, Text) and node.data.strip():
                doc.add_text(label=DocItemLabel.TEXT, text=node.data.strip(), parent=parent)

    def _dispatch_field_inline_body_nodes(
        self,
        *,
        doc: DoclingDocument,
        body_nodes: Sequence[Node],
        parent: NodeItem,
    ) -> None:
        """Dispatch inline key/value body nodes, merging checkbox labels with following text."""
        meaningful = self._meaningful_body_nodes(body_nodes)
        idx = 0
        while idx < len(meaningful):
            node = meaningful[idx]
            if isinstance(node, Element) and node.tagName == DocLangToken.CHECKBOX.value:
                checkbox_class = node.getAttribute(DocLangAttributeKey.CLASS.value)
                if checkbox_class == DocLangAttributeValue.SELECTED.value:
                    label = DocItemLabel.CHECKBOX_SELECTED
                else:
                    label = DocItemLabel.CHECKBOX_UNSELECTED
                text = ""
                remaining = meaningful[idx + 1 :]
                if len(remaining) == 1:
                    nxt = remaining[0]
                    if isinstance(nxt, Text):
                        text = nxt.data.strip()
                        idx += 1
                    elif isinstance(nxt, Element) and nxt.tagName == DocLangToken.CONTENT.value:
                        text = self._get_text(nxt)
                        idx += 1
                doc.add_text(label=label, text=text, parent=parent)
                idx += 1
                continue
            if isinstance(node, Element):
                self._dispatch_element(doc=doc, el=node, parent=parent)
            elif isinstance(node, Text) and node.data.strip():
                doc.add_text(label=DocItemLabel.TEXT, text=node.data.strip(), parent=parent)
            idx += 1

    def _meaningful_body_nodes(self, body_nodes: Sequence[Node]) -> list[Node]:
        return [
            node for node in body_nodes if isinstance(node, Element) or (isinstance(node, Text) and node.data.strip())
        ]

    def _is_field_inline_body(self, body_nodes: Sequence[Node]) -> bool:
        meaningful = self._meaningful_body_nodes(body_nodes)
        if not meaningful:
            return False
        for node in meaningful:
            if isinstance(node, Text):
                continue
            if isinstance(node, Element) and node.tagName not in self._FIELD_INLINE_BODY_TAGS:
                return False
        return True

    @staticmethod
    def _field_value_kind(el: Element) -> Literal["read_only", "fillable"]:
        if el.getAttribute(DocLangAttributeKey.CLASS.value) == "fillable":
            return "fillable"
        return "read_only"

    def _parse_field_region(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        prov_list = self._extract_provenance(doc=doc, el=el)
        fri = doc.add_field_region(
            parent=parent,
            prov=(prov_list[0] if prov_list else None),
        )
        for prov in prov_list[1:]:
            fri.prov.append(prov)
        _, body_nodes = self._split_element_children_head_body(el)
        self._dispatch_body_nodes(doc=doc, body_nodes=body_nodes, parent=fri)

    def _parse_field_item(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        prov_list = self._extract_provenance(doc=doc, el=el)
        content_layer = self._extract_layer(el=el)
        fi = doc.add_field_item(
            parent=parent,
            prov=(prov_list[0] if prov_list else None),
            content_layer=content_layer,
        )
        for prov in prov_list[1:]:
            fi.prov.append(prov)
        _, body_nodes = self._split_element_children_head_body(el)
        self._dispatch_body_nodes(doc=doc, body_nodes=body_nodes, parent=fi)

    def _parse_field_key(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        self._parse_field_kv(doc=doc, el=el, parent=parent, is_value=False)

    def _parse_field_value(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        self._parse_field_kv(doc=doc, el=el, parent=parent, is_value=True)

    def _parse_checkbox(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        checkbox_class = el.getAttribute(DocLangAttributeKey.CLASS.value)
        if checkbox_class == DocLangAttributeValue.SELECTED.value:
            label = DocItemLabel.CHECKBOX_SELECTED
        else:
            label = DocItemLabel.CHECKBOX_UNSELECTED
        doc.add_text(label=label, text="", parent=parent)

    def _parse_field_hint(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        prov_list = self._extract_provenance(doc=doc, el=el)
        content_layer = self._extract_layer(el=el)
        text, formatting = self._extract_text_with_formatting(el)
        text_stripped = text.strip()
        if not text_stripped:
            return
        item = doc.add_field_hint(
            text=text_stripped,
            parent=parent,
            prov=(prov_list[0] if prov_list else None),
            content_layer=content_layer,
            formatting=formatting,
        )
        self._apply_initial_text_provenance(item, text=text_stripped, prov_list=prov_list)

    def _field_kv_needs_inline_container(self, body_nodes: Sequence[Node]) -> bool:
        """True when key/value body must become an inline group, not flat text."""
        meaningful = self._meaningful_body_nodes(body_nodes)
        element_children = [node for node in meaningful if isinstance(node, Element)]
        if not self._is_field_inline_body(body_nodes):
            return False
        if len(element_children) > 1:
            return True
        if any(node.tagName == DocLangToken.CHECKBOX.value for node in element_children):
            return True
        if element_children and any(isinstance(node, Text) for node in meaningful):
            return True
        return False

    def _parse_field_kv(
        self,
        *,
        doc: DoclingDocument,
        el: Element,
        parent: Optional[NodeItem],
        is_value: bool,
    ) -> None:
        """Parse ``<key>`` / ``<value>`` into field key or value items."""
        prov_list = self._extract_provenance(doc=doc, el=el)
        content_layer = self._extract_layer(el=el)
        kind = self._field_value_kind(el) if is_value else "read_only"
        _, body_nodes = self._split_element_children_head_body(el)
        simple_text = self._get_children_simple_text_block(el)
        needs_inline = self._field_kv_needs_inline_container(body_nodes)

        if simple_text is not None and not needs_inline:
            text, formatting = self._extract_text_with_formatting(el)
            if is_value:
                item = doc.add_field_value(
                    text=text,
                    parent=parent,
                    prov=(prov_list[0] if prov_list else None),
                    content_layer=content_layer,
                    formatting=formatting,
                    kind=kind,
                )
            else:
                item = doc.add_field_key(
                    text=text,
                    parent=parent,
                    prov=(prov_list[0] if prov_list else None),
                    content_layer=content_layer,
                    formatting=formatting,
                )
            self._apply_initial_text_provenance(item, text=text, prov_list=prov_list)
            self._apply_custom_meta_from_element(item=item, el=el)
            return

        if needs_inline:
            if is_value:
                item = doc.add_field_value(
                    text="",
                    parent=parent,
                    prov=(prov_list[0] if prov_list else None),
                    content_layer=content_layer,
                    kind=kind,
                )
            else:
                item = doc.add_field_key(
                    text="",
                    parent=parent,
                    prov=(prov_list[0] if prov_list else None),
                    content_layer=content_layer,
                )
            inline_group = doc.add_inline_group(parent=item)
            self._dispatch_field_inline_body_nodes(
                doc=doc,
                body_nodes=body_nodes,
                parent=inline_group,
            )
            self._apply_initial_text_provenance(item, text="", prov_list=prov_list)
            self._apply_custom_meta_from_element(item=item, el=el)
            return

        if is_value:
            item = doc.add_field_value(
                text="",
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                content_layer=content_layer,
                kind=kind,
            )
        else:
            item = doc.add_field_key(
                text="",
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                content_layer=content_layer,
            )
        for node in body_nodes:
            if isinstance(node, Element):
                self._dispatch_element(doc=doc, el=node, parent=item)
            elif isinstance(node, Text) and node.data.strip():
                doc.add_text(label=DocItemLabel.TEXT, text=node.data.strip(), parent=item)
        self._apply_initial_text_provenance(item, text="", prov_list=prov_list)
        self._apply_custom_meta_from_element(item=item, el=el)

    def _first_non_whitespace_node(self, nodes: Sequence[Node]) -> Optional[Node]:
        """Return the first node that is not whitespace-only text."""
        for node in nodes:
            if isinstance(node, Text) and not node.data.strip():
                continue
            return node
        return None

    def _token_category(self, tag: str) -> Optional[DocLangCategory]:
        try:
            return DocLangVocabulary._get_category(DocLangToken(tag))
        except ValueError:
            return None

    def _is_element_head_tag(self, el: Element) -> bool:
        """Return True when ``el`` is an element allowed in an element head."""
        return el.tagName in _ELEMENT_HEAD_TAGS

    def _is_ignorable_head_whitespace(self, node: Node) -> bool:
        return isinstance(node, Text) and not node.data.strip()

    def _split_element_children_head_body(
        self,
        el: Element,
        *,
        body_starts_at: Optional[Callable[[Node], bool]] = None,
    ) -> tuple[list[Node], list[Node]]:
        """Split immediate children into element-head prefix and body."""
        head_nodes: list[Node] = []
        body_nodes: list[Node] = []
        in_body = False

        for node in el.childNodes:
            if not in_body:
                if body_starts_at is not None and body_starts_at(node):
                    in_body = True
                    body_nodes.append(node)
                    continue
                if self._is_ignorable_head_whitespace(node):
                    head_nodes.append(node)
                    continue
                if isinstance(node, Element) and self._is_element_head_tag(node):
                    head_nodes.append(node)
                    continue
                in_body = True
                body_nodes.append(node)
            else:
                body_nodes.append(node)

        return head_nodes, body_nodes

    def _nodes_to_xml(self, nodes: Sequence[Node]) -> str:
        """Serialize a node sequence to XML/text (unwrap ``<content>`` elements)."""
        parts: list[str] = []
        for node in nodes:
            if isinstance(node, Text):
                parts.append(node.data)
            elif isinstance(node, Element):
                if node.tagName == DocLangToken.CONTENT.value:
                    parts.append(self._nodes_to_xml(node.childNodes))
                else:
                    parts.append(node.toxml())
        return "".join(parts)

    _LIST_ITEM_VIRTUAL_TEXT_CONTENT_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            DocLangToken.CONTENT.value,
            DocLangToken.BOLD.value,
            DocLangToken.ITALIC.value,
            DocLangToken.UNDERLINE.value,
            DocLangToken.STRIKETHROUGH.value,
            DocLangToken.SUPERSCRIPT.value,
            DocLangToken.SUBSCRIPT.value,
            DocLangToken.HANDWRITING.value,
            DocLangToken.RTL.value,
            DocLangToken.BR.value,
            DocLangToken.CHECKBOX.value,
        }
    )

    _LIST_ITEM_SEGMENT_SIBLING_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            DocLangToken.LIST.value,
            DocLangToken.PICTURE.value,
        }
    )

    def _is_list_item_virtual_text(self, nodes: Sequence[Node]) -> bool:
        """Decide whether list-item content after ``<ldiv>`` is virtual ``<text>``.

        Inspect the first non-whitespace node after the delimiter:
        - element head tokens, raw text, ``<content>``, or inline formatting → virtual text
        - semantic/grouping elements (``<text>``, ``<code>``, ``<list>``, …) → not virtual text
        """
        first = self._first_non_whitespace_node(nodes)
        if first is None:
            return False
        if isinstance(first, Text):
            return True
        if not isinstance(first, Element):
            return False
        if self._is_element_head_tag(first):
            return True
        if first.tagName in self._LIST_ITEM_VIRTUAL_TEXT_CONTENT_TAGS:
            return True
        category = self._token_category(first.tagName)
        if category in {DocLangCategory.FORMATTING, DocLangCategory.CONTENT}:
            return True
        return category not in {DocLangCategory.SEMANTIC, DocLangCategory.GROUPING}

    def _content_nodes_after_list_item_head(self, nodes: Sequence[Node]) -> list[Node]:
        """Drop leading property/head tokens; keep virtual-text body nodes."""
        content_nodes: list[Node] = []
        skipping_head = True
        for node in nodes:
            if skipping_head:
                if isinstance(node, Text) and not node.data.strip():
                    continue
                if isinstance(node, Element) and self._is_element_head_tag(node):
                    continue
                skipping_head = False
            content_nodes.append(node)
        return content_nodes

    def _split_virtual_text_leading_text(self, nodes: Sequence[Node]) -> tuple[str, list[Node]]:
        """Split virtual-text body into leading plain text and remaining nodes."""
        text_parts: list[str] = []
        rest_start = 0
        for i, node in enumerate(nodes):
            if isinstance(node, Text):
                text_parts.append(node.data)
                rest_start = i + 1
            elif isinstance(node, Element) and node.tagName == DocLangToken.CONTENT.value:
                text_parts.append(self._get_text(node))
                rest_start = i + 1
            else:
                break
        leading = "".join(text_parts).strip()
        rest = [node for node in nodes[rest_start:] if not (isinstance(node, Text) and not node.data.strip())]
        return leading, rest

    def _parse_list_item_virtual_text(
        self,
        *,
        doc: DoclingDocument,
        el: Element,
        li_group: ListGroup,
        ordered: bool,
        marker_text: str,
        all_content_nodes: Sequence[Node],
    ) -> None:
        """Parse list-item body emitted as virtual ``<text>`` after ``<ldiv>``."""
        prov_list = self._provenance_from_location_nodes(doc=doc, nodes=all_content_nodes)
        body_nodes = self._content_nodes_after_list_item_head(all_content_nodes)
        leading_text, rest_nodes = self._split_virtual_text_leading_text(body_nodes)
        rest_elements = [node for node in rest_nodes if isinstance(node, Element)]

        if (
            leading_text
            and rest_elements
            and all(el.tagName in self._LIST_ITEM_SEGMENT_SIBLING_TAGS for el in rest_elements)
        ):
            li = self._add_list_item_with_provenance(
                doc=doc,
                text=leading_text,
                parent=li_group,
                enumerated=ordered,
                marker=marker_text,
                prov_list=prov_list,
            )
            for content_el in rest_elements:
                self._dispatch_element(doc=doc, el=content_el, parent=li)
        elif not rest_nodes and leading_text:
            self._add_list_item_with_provenance(
                doc=doc,
                text=leading_text,
                parent=li_group,
                enumerated=ordered,
                marker=marker_text,
                prov_list=prov_list,
            )
        elif self._is_simple_virtual_text_nodes(body_nodes):
            text = self._get_text_from_nodes(body_nodes).strip()
            self._add_list_item_with_provenance(
                doc=doc,
                text=text,
                parent=li_group,
                enumerated=ordered,
                marker=marker_text,
                prov_list=prov_list,
            )
        else:
            li = self._add_list_item_with_provenance(
                doc=doc,
                text="",
                parent=li_group,
                enumerated=ordered,
                marker=marker_text,
                prov_list=prov_list,
            )
            self._parse_inline_group(doc=doc, el=el, parent=li, nodes=body_nodes)

    def _is_simple_virtual_text_nodes(self, nodes: Sequence[Node]) -> bool:
        """True when virtual-text body is plain text (optionally via ``<content>``)."""
        for node in nodes:
            if isinstance(node, Text):
                continue
            if isinstance(node, Element):
                if node.tagName == DocLangToken.CONTENT.value:
                    continue
                return False
        return any(
            (isinstance(node, Text) and node.data.strip())
            or (isinstance(node, Element) and node.tagName == DocLangToken.CONTENT.value)
            for node in nodes
        )

    def _get_text_from_nodes(self, nodes: Sequence[Node]) -> str:
        parts: list[str] = []
        for node in nodes:
            if isinstance(node, Text):
                parts.append(node.data)
            elif isinstance(node, Element) and node.tagName == DocLangToken.CONTENT.value:
                parts.append(self._get_text(node))
        return "".join(parts)

    def _provenance_from_location_nodes(self, *, doc: DoclingDocument, nodes: Sequence[Node]) -> list[ProvenanceItem]:
        """Collect ``<location>`` quartets from a flat node sequence (element head)."""
        values: list[int] = []
        res_for_group: Optional[int] = None
        provs: list[ProvenanceItem] = []

        for node in nodes:
            if not isinstance(node, Element) or node.tagName != DocLangToken.LOCATION.value:
                continue
            try:
                v = int(node.getAttribute(DocLangAttributeKey.VALUE.value) or "0")
            except Exception:
                v = 0
            try:
                r = int(node.getAttribute(DocLangAttributeKey.RESOLUTION.value) or str(self._default_resolution))
            except Exception:
                r = self._default_resolution
            values.append(v)
            res_for_group = r
            if len(values) == 4:
                self._ensure_page_exists(
                    doc=doc,
                    page_no=self._page_no,
                    resolution=res_for_group or self._default_resolution,
                )
                l = float(min(values[0], values[2]))
                t = float(min(values[1], values[3]))
                rgt = float(max(values[0], values[2]))
                btm = float(max(values[1], values[3]))
                bbox = BoundingBox.from_tuple((l, t, rgt, btm), origin=CoordOrigin.TOPLEFT)
                provs.append(ProvenanceItem(page_no=self._page_no, bbox=bbox, charspan=(0, 0)))
                values = []
                res_for_group = None

        return provs

    def _add_list_item_with_provenance(
        self,
        *,
        doc: DoclingDocument,
        text: str,
        parent: NodeItem,
        enumerated: bool,
        marker: str,
        prov_list: list[ProvenanceItem],
    ) -> ListItem:
        item = doc.add_list_item(
            text=text,
            parent=parent,
            enumerated=enumerated,
            marker=marker,
            prov=(prov_list[0] if prov_list else None),
        )
        self._apply_initial_text_provenance(item, text=text, prov_list=prov_list)
        return item

    def _parse_list(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        ordered = el.getAttribute(DocLangAttributeKey.CLASS.value) == DocLangAttributeValue.ORDERED.value
        list_head_nodes = [node for node in el.childNodes if isinstance(node, Element)]
        thread_id = self._extract_thread_id_from_nodes(list_head_nodes)
        if (
            thread_id
            and (existing := self._get_thread_item(thread_id, host=DocLangToken.LIST.value)) is not None
            and isinstance(existing, ListGroup)
        ):
            li_group = existing
        else:
            li_group = doc.add_list_group(parent=parent)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=DocLangToken.LIST.value, item=li_group)
        actual_children = [
            ch for ch in el.childNodes if isinstance(ch, Element) and ch.tagName not in {DocLangToken.LOCATION.value}
        ]

        # Find all ldiv boundaries (delimiters)
        boundaries = [
            i for i, n in enumerate(actual_children) if isinstance(n, Element) and n.tagName == DocLangToken.LDIV.value
        ]

        # Create ranges: each range is from an ldiv to the next ldiv (or end)
        ranges = [
            (
                boundaries[i],
                (boundaries[i + 1] if i < len(boundaries) - 1 else len(actual_children)),
            )
            for i in range(len(boundaries))
        ]

        for start, end in ranges:
            # The ldiv element itself
            ldiv_el = actual_children[start]

            # Extract marker if present within the ldiv
            marker_text = ""
            for ch in ldiv_el.childNodes:
                if isinstance(ch, Element) and ch.tagName == DocLangToken.MARKER.value:
                    marker_text = self._get_text(ch).strip()
                    break

            # Get ALL nodes (including Text nodes) between this ldiv and the next
            # We need to check the original childNodes, not just actual_children
            ldiv_index_in_all = list(el.childNodes).index(ldiv_el)
            next_ldiv_index = None
            if end < len(actual_children):
                next_ldiv_index = list(el.childNodes).index(actual_children[end])

            # Get all nodes between ldivs (including Text nodes and head tokens)
            if next_ldiv_index is not None:
                all_content_nodes = list(el.childNodes)[ldiv_index_in_all + 1 : next_ldiv_index]
            else:
                all_content_nodes = list(el.childNodes)[ldiv_index_in_all + 1 :]

            # Content elements come after the ldiv (start+1 to end) - only Element nodes,
            # excluding property tokens that may appear between ldiv and body content.
            content_elements = [
                node
                for node in actual_children[start + 1 : end]
                if not (isinstance(node, Element) and self._is_element_head_tag(node))
            ]

            is_virtual_text = self._is_list_item_virtual_text(all_content_nodes)

            if not all_content_nodes:
                # Empty list item (just ldiv, no content)
                doc.add_list_item(
                    text="",
                    parent=li_group,
                    enumerated=ordered,
                    marker=marker_text,
                )
            elif is_virtual_text:
                self._parse_list_item_virtual_text(
                    doc=doc,
                    el=el,
                    li_group=li_group,
                    ordered=ordered,
                    marker_text=marker_text,
                    all_content_nodes=all_content_nodes,
                )
            elif len(content_elements) == 1 and isinstance(content_elements[0], Element):
                # Single element after ldiv
                content_el = content_elements[0]
                if content_el.tagName == DocLangToken.TEXT.value:
                    # Check if it's a simple text item or has complex content (code, formula, etc.)
                    element_children = [
                        node
                        for node in content_el.childNodes
                        if isinstance(node, Element)
                        and node.tagName not in {DocLangToken.LOCATION.value, DocLangToken.LAYER.value}
                    ]

                    # If it has complex content (multiple elements or non-simple content), dispatch it
                    if len(element_children) > 1 or self._get_children_simple_text_block(content_el) is None:
                        # Complex content - create empty list item and dispatch the text element
                        li = doc.add_list_item(
                            text="",
                            parent=li_group,
                            enumerated=ordered,
                            marker=marker_text,
                        )
                        self._dispatch_element(doc=doc, el=content_el, parent=li)
                    else:
                        # Simple text item
                        prov_list = self._extract_provenance(doc=doc, el=content_el)
                        text = self._get_text(content_el).strip()
                        doc.add_list_item(
                            text=text,
                            parent=li_group,
                            enumerated=ordered,
                            marker=marker_text,
                            prov=(prov_list[0] if prov_list else None),
                        )
                else:
                    # Other single element (heading, code, nested list, etc.)
                    li = doc.add_list_item(
                        text="",
                        parent=li_group,
                        enumerated=ordered,
                        marker=marker_text,
                    )
                    self._dispatch_element(doc=doc, el=content_el, parent=li)
            else:
                # Multiple content elements after ldiv
                # Special case: if first element is a simple <text> and remaining are <list> elements,
                # treat the text as the ListItem's text (collapsed representation)
                first_el = content_elements[0]
                remaining_els = content_elements[1:]

                if (
                    isinstance(first_el, Element)
                    and first_el.tagName == DocLangToken.TEXT.value
                    and all(
                        isinstance(el, Element) and el.tagName in self._LIST_ITEM_SEGMENT_SIBLING_TAGS
                        for el in remaining_els
                    )
                ):
                    # Check if the text element is simple (no complex content)
                    element_children = [
                        node
                        for node in first_el.childNodes
                        if isinstance(node, Element)
                        and node.tagName not in {DocLangToken.LOCATION.value, DocLangToken.LAYER.value}
                    ]

                    if len(element_children) <= 1 and self._get_children_simple_text_block(first_el) is not None:
                        # Simple text - use it as the ListItem's text
                        prov_list = self._extract_provenance(doc=doc, el=first_el)
                        text = self._get_text(first_el).strip()
                        li = doc.add_list_item(
                            text=text,
                            parent=li_group,
                            enumerated=ordered,
                            marker=marker_text,
                            prov=(prov_list[0] if prov_list else None),
                        )
                        # Dispatch the nested list(s) as children
                        for content_el in remaining_els:
                            self._dispatch_element(doc=doc, el=content_el, parent=li)
                    else:
                        # Complex text content - dispatch all elements
                        li = doc.add_list_item(
                            text="",
                            parent=li_group,
                            enumerated=ordered,
                            marker=marker_text,
                        )
                        for content_el in content_elements:
                            self._dispatch_element(doc=doc, el=content_el, parent=li)
                else:
                    # General case: dispatch all elements as children
                    li = doc.add_list_item(
                        text="",
                        parent=li_group,
                        enumerated=ordered,
                        marker=marker_text,
                    )
                    for content_el in content_elements:
                        self._dispatch_element(doc=doc, el=content_el, parent=li)

    # ------------- Inline groups -------------
    def _parse_inline_group(
        self,
        *,
        doc: DoclingDocument,
        el: Element,
        parent: Optional[NodeItem],
        nodes: Optional[Sequence[Node]] = None,
    ) -> None:
        """Parse <inline> elements into InlineGroup objects."""
        # Create the inline group
        inline_group = doc.add_inline_group(parent=parent)

        # Process all child elements, adding them as children of the inline group
        my_nodes = nodes or el.childNodes
        for node in my_nodes:
            if isinstance(node, Element):
                # Recursively dispatch child elements with the inline group as parent
                self._dispatch_element(doc=doc, el=node, parent=inline_group)
            elif isinstance(node, Text):
                # Handle direct text content
                text_content = node.data.strip()
                if text_content:
                    doc.add_text(
                        label=DocItemLabel.TEXT,
                        text=text_content,
                        parent=inline_group,
                    )

    # ------------- Floating items (table / picture) -------------

    @staticmethod
    def _table_label_from_otsl_element(el: Element) -> DocItemLabel:
        """Resolve table label from ``<table>`` or ``<index>`` (v0.5: no ``class`` on ``<table>``)."""
        if el.tagName == DocLangToken.INDEX.value:
            return DocItemLabel.DOCUMENT_INDEX
        if el.tagName != DocLangToken.TABLE.value:
            raise ValueError(f"Expected table or index element, got '{el.tagName}'.")
        if el.getAttribute(DocLangAttributeKey.CLASS.value):
            raise ValueError("table element must not have a class attribute.")
        return DocItemLabel.TABLE

    def _parse_table(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        """Parse ``<table>``, ``<index>``, or a ``<group>`` wrapping them (with footnotes)."""
        otsl_el: Optional[Element]
        footnotes: list[TextItem] = []
        if el.tagName in {DocLangToken.TABLE.value, DocLangToken.INDEX.value}:
            caption = self._extract_caption(doc=doc, el=el)
            otsl_el = el
            table_label = self._table_label_from_otsl_element(el)
        else:
            footnotes = self._extract_footnotes(doc=doc, el=el)
            otsl_el = self._first_child(el, DocLangToken.TABLE.value) or self._first_child(el, DocLangToken.INDEX.value)
            caption = self._extract_caption(doc=doc, el=el)
            if caption is None and otsl_el is not None:
                caption = self._extract_caption(doc=doc, el=otsl_el)
            if otsl_el is None:
                tbl = doc.add_table(data=TableData(), caption=caption, parent=parent)
                for ftn in footnotes:
                    tbl.footnotes.append(ftn.get_ref())
                return
            table_label = self._table_label_from_otsl_element(otsl_el)

        head_nodes, body_nodes = self._split_element_children_head_body(otsl_el)
        tbl_provs = self._provenance_from_location_nodes(doc=doc, nodes=head_nodes)
        content_layer = self._layer_from_nodes(head_nodes)
        thread_id = self._extract_thread_id_from_nodes(head_nodes)
        table_host = otsl_el.tagName
        if (
            thread_id
            and (existing := self._get_thread_item(thread_id, host=table_host)) is not None
            and isinstance(existing, TableItem)
        ):
            row_offset, col_offset = _thread_table_merge_offset(existing, tbl_provs[0]) if tbl_provs else (0, 0)
            for prov in tbl_provs:
                existing.prov.append(prov)
            inner = self._nodes_to_xml(body_nodes)
            if inner.strip():
                tbl_content = _wrap(text=inner, wrap_tag=DocLangToken.TABLE.value)
                fragment_td = self._parse_otsl_table_content(
                    otsl_content=tbl_content,
                    doc=doc,
                    parent=existing,
                    row_offset=row_offset,
                    col_offset=col_offset,
                )
                if existing.data is None:
                    existing.data = fragment_td
                else:
                    _merge_table_data(
                        base=existing.data,
                        fragment=fragment_td,
                        row_offset=row_offset,
                        col_offset=col_offset,
                    )
            return
        inner = self._nodes_to_xml(body_nodes)
        tbl = doc.add_table(
            data=TableData(),
            caption=caption,
            parent=parent,
            prov=(tbl_provs[0] if tbl_provs else None),
            content_layer=content_layer,
            label=table_label,
        )
        tbl_content = _wrap(text=inner, wrap_tag=DocLangToken.TABLE.value)
        td = self._parse_otsl_table_content(otsl_content=tbl_content, doc=doc, parent=tbl)
        tbl.data = td
        for p in tbl_provs[1:]:
            tbl.prov.append(p)
        if thread_id:
            self._register_thread(thread_id=thread_id, host=table_host, item=tbl)
        for ftn in footnotes:
            tbl.footnotes.append(ftn.get_ref())

    def _parse_picture(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        """Parse ``<picture>`` or a ``<group>`` wrapping it (with footnotes)."""
        picture_el: Optional[Element]
        footnotes: list[TextItem] = []
        if el.tagName == DocLangToken.PICTURE.value:
            caption = self._extract_caption(doc=doc, el=el)
            picture_el = el
        else:
            footnotes = self._extract_footnotes(doc=doc, el=el)
            picture_el = self._first_child(el, DocLangToken.PICTURE.value)
            caption = self._extract_caption(doc=doc, el=el)
            if caption is None and picture_el is not None:
                caption = self._extract_caption(doc=doc, el=picture_el)

        prov_list: list[ProvenanceItem] = []
        content_layer: Optional[ContentLayer] = None
        if picture_el is not None:
            prov_list = self._extract_provenance(doc=doc, el=picture_el)
            content_layer = self._extract_layer(el=picture_el)

        pic = doc.add_picture(
            caption=caption,
            parent=parent,
            prov=(prov_list[0] if prov_list else None),
            content_layer=content_layer,
        )
        for p in prov_list[1:]:
            pic.prov.append(p)
        for ftn in footnotes:
            pic.footnotes.append(ftn.get_ref())

        if picture_el is not None:
            if label_val := self._extract_label_value(el=picture_el):
                if class_name := _picture_classification_label_from_doclang(label_val):
                    if pic.meta is None:
                        pic.meta = PictureMeta()
                    pic.meta.classification = PictureClassificationMetaField(
                        predictions=[
                            PictureClassificationPrediction(
                                class_name=class_name,
                                confidence=1.0,
                            )
                        ]
                    )
            self._apply_custom_meta_from_element(item=pic, el=picture_el)
            self._parse_picture_body(doc=doc, picture_el=picture_el, pic=pic)

    def _parse_picture_body(self, *, doc: DoclingDocument, picture_el: Element, pic: PictureItem) -> None:
        """Parse v0.6 picture body.

        Layout after the element head:
        - Preamble: optional ``<src>``, then optional ``<tabular>`` (chart data only)
        - Content: any semantic elements (e.g. nested ``<table>``), as picture children
        """
        _, body_nodes = self._split_element_children_head_body(picture_el)
        idx = 0
        while idx < len(body_nodes):
            node = body_nodes[idx]
            if not isinstance(node, Element):
                idx += 1
                continue
            if node.tagName == DocLangToken.SRC.value:
                # image URI restoration is not implemented yet
                idx += 1
                continue
            if node.tagName == DocLangToken.TABULAR.value:
                _, otsl_body_nodes = self._split_element_children_head_body(node)
                inner = self._nodes_to_xml(otsl_body_nodes)
                td = self._parse_otsl_table_content(_wrap(inner, DocLangToken.TABULAR.value))
                if pic.meta is None:
                    pic.meta = PictureMeta()
                pic.meta.tabular_chart = TabularChartMetaField(chart_data=td)
                idx += 1
                continue
            break

        for node in body_nodes[idx:]:
            if isinstance(node, Element):
                self._dispatch_element(doc=doc, el=node, parent=pic)

    def _apply_custom_meta_from_element(self, *, item: NodeItem, el: Element) -> None:
        """Restore item meta from ``<custom>`` children in the element head."""
        head_nodes, _ = self._split_element_children_head_body(el)
        self._apply_custom_meta_from_head_nodes(item=item, head_nodes=head_nodes)

    def _ensure_item_meta(self, item: DocItem) -> BaseMeta:
        """Return ``item.meta``, creating the appropriate meta model when absent."""
        if item.meta is None:
            if isinstance(item, PictureItem):
                item.meta = PictureMeta()
            elif isinstance(item, FloatingItem):
                item.meta = FloatingMeta()
            else:
                item.meta = BaseMeta()
        return item.meta

    @staticmethod
    def _split_namespace_field_tag(tag: str) -> Optional[tuple[str, str]]:
        """Parse a ``namespace__field`` custom-vocabulary tag."""
        if MetaUtils._META_FIELD_NAMESPACE_DELIMITER not in tag:
            return None
        namespace, name = tag.split(MetaUtils._META_FIELD_NAMESPACE_DELIMITER, maxsplit=1)
        if namespace and name:
            return namespace, name
        return None

    def _apply_custom_meta_field_element(self, *, item: DocItem, field_el: Element) -> None:
        """Map one ``<custom>`` child element onto ``item.meta``."""
        tag = field_el.tagName
        value = self._get_text(field_el)
        meta = self._ensure_item_meta(item)

        if tag == _DOCLANG_META_TAG_SUMMARY:
            if text := value.strip():
                meta.summary = SummaryMetaField(text=text)
        elif tag == _DOCLANG_META_TAG_DESCRIPTION:
            if text := value.strip():
                meta.description = DescriptionMetaField(text=text)
        elif tag == _DOCLANG_META_TAG_SMILES:
            if isinstance(item, PictureItem) and (smi := value.strip()):
                picture_meta = cast(PictureMeta, self._ensure_item_meta(item))
                picture_meta.molecule = MoleculeMetaField(smi=smi)
        elif parsed := self._split_namespace_field_tag(tag):
            namespace, name = parsed
            meta.set_custom_field(namespace=namespace, name=name, value=value)

    def _apply_custom_meta_from_head_nodes(self, *, item: NodeItem, head_nodes: Sequence[Node]) -> None:
        """Restore item meta from ``<custom>`` children in the element head."""
        if not isinstance(item, DocItem):
            return
        for node in head_nodes:
            if not isinstance(node, Element) or node.tagName != DocLangToken.CUSTOM.value:
                continue
            for child in node.childNodes:
                if isinstance(child, Element):
                    self._apply_custom_meta_field_element(item=item, field_el=child)

    # ------------- Helpers -------------
    def _extract_caption(self, *, doc: DoclingDocument, el: Element) -> Optional[TextItem]:
        """Extract caption from element head or from a ``<group>`` wrapper around a float."""
        cap_el = self._first_child(el, DocLangToken.CAPTION.value)
        if cap_el is None:
            return None
        text = self._get_text(cap_el).strip()
        if not text:
            return None
        prov_list = self._extract_provenance(doc=doc, el=cap_el)
        item = doc.add_text(
            label=DocItemLabel.CAPTION,
            text=text,
            prov=(prov_list[0] if prov_list else None),
        )
        for p in prov_list[1:]:
            item.prov.append(p)
        return item

    def _extract_footnotes(self, *, doc: DoclingDocument, el: Element) -> list[TextItem]:
        footnotes: list[TextItem] = []
        for node in el.childNodes:
            if isinstance(node, Element) and node.tagName == DocLangToken.FOOTNOTE.value:
                text = self._get_text(node).strip()
                if text:
                    prov_list = self._extract_provenance(doc=doc, el=node)
                    item = doc.add_text(
                        label=DocItemLabel.FOOTNOTE,
                        text=text,
                        prov=(prov_list[0] if prov_list else None),
                    )
                    for p in prov_list[1:]:
                        item.prov.append(p)
                    footnotes.append(item)
        return footnotes

    def _first_child(self, el: Element, tag_name: str) -> Optional[Element]:
        for node in el.childNodes:
            if isinstance(node, Element) and node.tagName == tag_name:
                return node
        return None

    def _inner_xml(self, el: Element, exclude_tags: Optional[set[str]] = None) -> str:
        """Extract inner XML content, optionally excluding specific element tags.

        Args:
            el: The element to extract content from
            exclude_tags: Optional set of tag names to exclude from the output
        """
        parts: list[str] = []
        exclude_tags = exclude_tags or set()
        for node in el.childNodes:
            if isinstance(node, Text):
                parts.append(node.data)
            elif isinstance(node, Element):
                if node.tagName == DocLangToken.CONTENT.value:
                    res = self._inner_xml(node, exclude_tags=exclude_tags)
                    parts.append(res)
                elif node.tagName not in exclude_tags:
                    parts.append(node.toxml())
        return "".join(parts)

    def _layer_from_nodes(self, nodes: Sequence[Node]) -> Optional[ContentLayer]:
        r"""Extract content layer from ``<layer value=\"...\"/>`` in element head nodes."""
        for node in nodes:
            if isinstance(node, Element) and node.tagName == DocLangToken.LAYER.value:
                if layer_value := node.getAttribute(DocLangAttributeKey.VALUE.value):
                    try:
                        return ContentLayer(layer_value)
                    except ValueError:
                        pass
        return None

    def _label_value_from_nodes(self, nodes: Sequence[Node]) -> Optional[str]:
        r"""Extract ``<label value=\"...\"/>`` from element head nodes."""
        for node in nodes:
            if isinstance(node, Element) and node.tagName == DocLangToken.LABEL.value:
                if label_val := node.getAttribute(DocLangAttributeKey.VALUE.value):
                    return label_val
        return None

    # --------- OTSL table parsing (inlined) ---------
    _OTSL_STRUCTURAL_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            DocLangToken.FCEL.value,
            DocLangToken.ECEL.value,
            DocLangToken.LCEL.value,
            DocLangToken.UCEL.value,
            DocLangToken.XCEL.value,
            DocLangToken.NL.value,
            DocLangToken.CHED.value,
            DocLangToken.RHED.value,
            DocLangToken.SROW.value,
            DocLangToken.CORN.value,
        }
    )

    def _bbox_from_location_text_fragments(
        self, *, doc: DoclingDocument, fragments: list[str]
    ) -> Optional[BoundingBox]:
        r"""Build a TOPLEFT bbox from four ``<location value=\"...\"/>`` XML fragments."""
        if len(fragments) != 4:
            return None
        values: list[int] = []
        res_for_group: Optional[int] = None
        for fragment in fragments:
            frag_dom = parseString(fragment)
            loc_el = frag_dom.documentElement
            if loc_el is None or loc_el.tagName != DocLangToken.LOCATION.value:
                return None
            try:
                v = int(loc_el.getAttribute(DocLangAttributeKey.VALUE.value) or "0")
            except Exception:
                v = 0
            try:
                r = int(loc_el.getAttribute(DocLangAttributeKey.RESOLUTION.value) or str(self._default_resolution))
            except Exception:
                r = self._default_resolution
            values.append(v)
            res_for_group = r
        self._ensure_page_exists(
            doc=doc,
            page_no=self._page_no,
            resolution=res_for_group or self._default_resolution,
        )
        l = float(min(values[0], values[2]))
        t = float(min(values[1], values[3]))
        rgt = float(max(values[0], values[2]))
        btm = float(max(values[1], values[3]))
        return BoundingBox.from_tuple((l, t, rgt, btm), origin=CoordOrigin.TOPLEFT)

    def _consume_leading_location_fragments(
        self,
        *,
        doc: Optional[DoclingDocument],
        texts: list[str],
        start: int,
    ) -> tuple[int, Optional[BoundingBox]]:
        """Consume a leading quartet of location fragments; return next index and bbox."""
        frags: list[str] = []
        idx = start
        loc_tag = f"<{DocLangToken.LOCATION.value}"
        while idx < len(texts) and texts[idx].strip().startswith(loc_tag):
            frags.append(texts[idx])
            idx += 1
            if len(frags) == 4:
                bbox = self._bbox_from_location_text_fragments(doc=doc, fragments=frags) if doc is not None else None
                return idx, bbox
        return start, None

    def _consume_otsl_cell_body_parts(self, texts: list[str], start: int) -> tuple[int, list[str]]:
        """Collect OTSL cell body fragments until the next structural token."""
        structural = {
            DocLangVocabulary._create_selfclosing_token(token=DocLangToken.FCEL),
            DocLangVocabulary._create_selfclosing_token(token=DocLangToken.ECEL),
            DocLangVocabulary._create_selfclosing_token(token=DocLangToken.LCEL),
            DocLangVocabulary._create_selfclosing_token(token=DocLangToken.UCEL),
            DocLangVocabulary._create_selfclosing_token(token=DocLangToken.XCEL),
            DocLangVocabulary._create_selfclosing_token(token=DocLangToken.NL),
            DocLangVocabulary._create_selfclosing_token(token=DocLangToken.CHED),
            DocLangVocabulary._create_selfclosing_token(token=DocLangToken.RHED),
            DocLangVocabulary._create_selfclosing_token(token=DocLangToken.SROW),
        }
        parts: list[str] = []
        idx = start
        while idx < len(texts) and texts[idx] not in structural:
            parts.append(texts[idx])
            idx += 1
        return idx, parts

    def _otsl_extract_tokens_and_text(self, s: str) -> tuple[list[str], list[str]]:
        """Extract OTSL structural tokens and interleaved text.

        Strips the outer wrapper and preserves OTSL body content including per-cell
        ``<location>`` tokens. Handles nested XML elements (like
        ``<text><italic>...</italic></text>``) by keeping them as single units.
        """
        tokens: list[str] = []
        parts: list[str] = []

        dom = parseString(s)
        otsl_el = dom.documentElement
        if otsl_el is None:
            raise ValueError("No document element found")

        otsl_tokens = {
            DocLangToken.FCEL.value,
            DocLangToken.ECEL.value,
            DocLangToken.LCEL.value,
            DocLangToken.UCEL.value,
            DocLangToken.XCEL.value,
            DocLangToken.NL.value,
            DocLangToken.CHED.value,
            DocLangToken.RHED.value,
            DocLangToken.SROW.value,
            DocLangToken.CORN.value,
        }

        for node in otsl_el.childNodes:
            if isinstance(node, Text):
                text = node.data.strip()
                if text:
                    parts.append(text)
            elif isinstance(node, Element):
                tag_name = node.tagName
                if tag_name in otsl_tokens:
                    token_str = f"<{tag_name}/>"
                    tokens.append(token_str)
                    parts.append(token_str)
                else:
                    # This is a nested element (like <text>, <italic>, etc.)
                    # Keep it as a complete XML string
                    xml_str = node.toxml()
                    parts.append(xml_str)

        return tokens, parts

    def _otsl_parse_texts(
        self,
        texts: list[str],
        tokens: list[str],
        doc: Optional["DoclingDocument"] = None,
        parent: Optional[NodeItem] = None,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> tuple[list[TableCell], list[list[str]]]:
        """Parse OTSL interleaved texts+tokens into TableCell list and row tokens."""
        # Token strings used in the stream (normalized to <name>)

        fcel = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.FCEL)
        ecel = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.ECEL)
        lcel = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.LCEL)
        ucel = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.UCEL)
        xcel = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.XCEL)
        nl = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.NL)
        ched = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.CHED)
        rhed = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.RHED)
        srow = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.SROW)
        corn = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.CORN)

        # Clean tokens to only structural OTSL markers
        clean_tokens: list[str] = []
        for t in tokens:
            if t in [ecel, fcel, lcel, ucel, xcel, nl, ched, rhed, srow, corn]:
                clean_tokens.append(t)
        tokens = clean_tokens

        # Split into rows by NL markers while keeping segments
        split_row_tokens = [list(group) for is_sep, group in groupby(tokens, key=lambda z: z == nl) if not is_sep]

        table_cells: list[TableCell] = []
        r_idx = 0
        c_idx = 0

        def count_right(rows: list[list[str]], c: int, r: int, which: list[str]) -> int:
            span = 0
            j = c
            while j < len(rows[r]) and rows[r][j] in which:
                j += 1
                span += 1
            return span

        def count_down(rows: list[list[str]], c: int, r: int, which: list[str]) -> int:
            span = 0
            i = r
            while i < len(rows) and c < len(rows[i]) and rows[i][c] in which:
                i += 1
                span += 1
            return span

        origin_tokens = [fcel, ecel, ched, rhed, srow, corn]
        continuation_origin_tokens = [lcel, ucel, xcel]

        for i, t in enumerate(texts):
            cell_text = ""
            if t in origin_tokens + continuation_origin_tokens:
                row_span = 1
                col_span = 1
                cell_bbox: Optional[BoundingBox] = None
                content_idx = i + 1
                cell_parts: list[str] = []
                if t != ecel and content_idx < len(texts):
                    content_idx, cell_bbox = self._consume_leading_location_fragments(
                        doc=doc,
                        texts=texts,
                        start=content_idx,
                    )
                    content_idx, cell_parts = self._consume_otsl_cell_body_parts(texts, content_idx)
                    cell_text = "".join(cell_parts)

                is_continuation_origin = t in continuation_origin_tokens
                if is_continuation_origin and not cell_text.strip() and not cell_parts:
                    pass
                else:
                    next_right = texts[content_idx] if content_idx < len(texts) else ""
                    next_bottom = (
                        split_row_tokens[r_idx + 1][c_idx]
                        if (r_idx + 1) < len(split_row_tokens) and c_idx < len(split_row_tokens[r_idx + 1])
                        else ""
                    )

                    if next_right in [lcel, xcel]:
                        col_span += count_right(split_row_tokens, c_idx + 1, r_idx, [lcel, xcel])
                    if next_bottom in [ucel, xcel]:
                        row_span += count_down(split_row_tokens, c_idx, r_idx + 1, [ucel, xcel])

                    cell_text_stripped = cell_text.strip()
                    xml_parts = [
                        part.strip()
                        for part in cell_parts
                        if part.strip().startswith("<") and part.strip().endswith(">")
                    ]
                    cell_added = False
                    if xml_parts and doc is not None and parent is not None:
                        cell_group = doc.add_group(parent=parent, label=GroupLabel.UNSPECIFIED)
                        text_parts: list[str] = []
                        for part in xml_parts:
                            wrapped_xml = f"<root>{part}</root>"
                            dom = parseString(wrapped_xml)
                            root_el = dom.documentElement
                            if root_el is None:
                                raise ValueError("No document element found")
                            for child_node in root_el.childNodes:
                                if isinstance(child_node, Element):
                                    self._dispatch_element(doc=doc, el=child_node, parent=cell_group)
                                    text_parts.append(self._get_text(child_node))
                        actual_text = "".join(text_parts).strip() or cell_text_stripped
                        table_cells.append(
                            RichTableCell(
                                text=actual_text,
                                row_span=row_span,
                                col_span=col_span,
                                start_row_offset_idx=r_idx + row_offset,
                                end_row_offset_idx=r_idx + row_span + row_offset,
                                start_col_offset_idx=c_idx + col_offset,
                                end_col_offset_idx=c_idx + col_span + col_offset,
                                ref=cell_group.get_ref(),
                                bbox=cell_bbox,
                            )
                        )
                        cell_added = True

                    if not cell_added:
                        table_cells.append(
                            TableCell(
                                text=cell_text_stripped,
                                row_span=row_span,
                                col_span=col_span,
                                start_row_offset_idx=r_idx + row_offset,
                                end_row_offset_idx=r_idx + row_span + row_offset,
                                start_col_offset_idx=c_idx + col_offset,
                                end_col_offset_idx=c_idx + col_span + col_offset,
                                column_header=t in [ched, corn],
                                row_header=t in [rhed, corn],
                                row_section=t == srow,
                                bbox=cell_bbox,
                            )
                        )

            if t in origin_tokens + continuation_origin_tokens:
                c_idx += 1
            if t == nl:
                r_idx += 1
                c_idx = 0

        return table_cells, split_row_tokens

    def _parse_otsl_table_content(
        self,
        otsl_content: str,
        doc: Optional["DoclingDocument"] = None,
        parent: Optional[NodeItem] = None,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> TableData:
        """Parse OTSL content into TableData (inlined from utils)."""
        tokens, mixed = self._otsl_extract_tokens_and_text(otsl_content)
        table_cells, split_rows = self._otsl_parse_texts(
            mixed,
            tokens,
            doc=doc,
            parent=parent,
            row_offset=row_offset,
            col_offset=col_offset,
        )
        return TableData(
            num_rows=len(split_rows),
            num_cols=(max(len(r) for r in split_rows) if split_rows else 0),
            table_cells=table_cells,
        )

    def _extract_text_with_formatting(self, el: Element) -> tuple[str, Optional[Formatting]]:
        """Extract text content and formatting from an element.

        If the element contains a single formatting child (bold, italic, etc.),
        recursively extract the text and build up the Formatting object.

        Returns:
            Tuple of (text_content, formatting_object or None)
        """
        # Get non-whitespace, non-location child elements
        child_elements = [
            node
            for node in el.childNodes
            if isinstance(node, Element) and node.tagName not in {DocLangToken.LOCATION.value}
        ]

        # Check if we have a single child that is a formatting tag
        if len(child_elements) == 1:
            child = child_elements[0]
            tag_name = child.tagName

            # Mapping of format tags to Formatting attributes
            format_tags = {
                DocLangToken.BOLD,
                DocLangToken.ITALIC,
                DocLangToken.STRIKETHROUGH,
                DocLangToken.UNDERLINE,
                DocLangToken.SUPERSCRIPT,
                DocLangToken.SUBSCRIPT,
                DocLangToken.RTL,
            }

            if tag_name in format_tags:
                # Recursively extract text and formatting from the child
                text, child_formatting = self._extract_text_with_formatting(child)

                # Build up the formatting object
                if child_formatting is None:
                    child_formatting = Formatting()

                # Apply the current formatting tag
                if tag_name == DocLangToken.BOLD.value:
                    child_formatting.bold = True
                elif tag_name == DocLangToken.ITALIC.value:
                    child_formatting.italic = True
                elif tag_name == DocLangToken.STRIKETHROUGH.value:
                    child_formatting.strikethrough = True
                elif tag_name == DocLangToken.UNDERLINE.value:
                    child_formatting.underline = True
                elif tag_name == DocLangToken.SUPERSCRIPT.value:
                    child_formatting.script = Script.SUPER
                elif tag_name == DocLangToken.SUBSCRIPT.value:
                    child_formatting.script = Script.SUB

                return text, child_formatting

        # No formatting found, just extract plain text
        return self._get_text(el), None

    def _get_text(self, el: Element) -> str:
        out: list[str] = []
        for node in el.childNodes:
            if isinstance(node, Text):
                # Skip pure indentation/pretty-print whitespace
                if node.data.strip():
                    out.append(node.data if el.tagName == DocLangToken.CONTENT.value else node.data.strip())
            elif isinstance(node, Element):
                nm = node.tagName
                if nm in {DocLangToken.LOCATION.value}:
                    continue
                if nm == DocLangToken.BR.value:
                    out.append("\n")
                else:
                    out.append(self._get_text(node))
        return "".join(out)

    # --------- Location helpers ---------
    def _ensure_page_exists(self, *, doc: DoclingDocument, page_no: int, resolution: int) -> None:
        # If the page already exists, do nothing; otherwise add with a square size based on resolution
        if page_no not in doc.pages:
            doc.add_page(page_no=page_no, size=Size(width=resolution, height=resolution))

    def _extract_provenance(self, *, doc: DoclingDocument, el: Element) -> list[ProvenanceItem]:
        head_nodes, _ = self._split_element_children_head_body(el)
        return self._provenance_from_location_nodes(doc=doc, nodes=head_nodes)

    def _extract_layer(self, *, el: Element) -> Optional[ContentLayer]:
        r"""Extract content layer from element-head ``<layer value=\"...\"/>``."""
        head_nodes, _ = self._split_element_children_head_body(el)
        return self._layer_from_nodes(head_nodes)

    def _extract_label_value(self, *, el: Element) -> Optional[str]:
        r"""Extract ``<label value=\"...\"/>`` from element head."""
        head_nodes, _ = self._split_element_children_head_body(el)
        return self._label_value_from_nodes(head_nodes)
