#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define classes for HTML serialization."""
import base64
import html
import logging
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote
from xml.etree.cElementTree import SubElement, tostring
from xml.sax.saxutils import unescape

import latex2mathml.converter
from pydantic import AnyUrl, BaseModel
from typing_extensions import override

from docling_core.experimental.serializer.base import (
    BaseDocSerializer,
    BaseFallbackSerializer,
    BaseFormSerializer,
    BaseInlineSerializer,
    BaseKeyValueSerializer,
    BaseListSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.experimental.serializer.common import (
    CommonParams,
    DocSerializer,
    create_ser_result,
)
from docling_core.experimental.serializer.html_styles import (
    _get_css_for_single_column,
    _get_css_for_split_page,
)
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    CodeItem,
    ContentLayer,
    DocItem,
    DoclingDocument,
    FloatingItem,
    FormItem,
    FormulaItem,
    GraphData,
    ImageRef,
    InlineGroup,
    KeyValueItem,
    ListItem,
    NodeItem,
    OrderedList,
    PictureItem,
    SectionHeaderItem,
    TableCell,
    TableItem,
    TextItem,
    TitleItem,
    UnorderedList,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.utils import (
    get_html_tag_with_text_direction,
    get_text_direction,
)

_logger = logging.getLogger(__name__)


class HTMLOutputStyle(str, Enum):
    """HTML output style."""

    SINGLE_COLUMN = "single_column"
    SPLIT_PAGE = "split_page"


class HTMLParams(CommonParams):
    """HTML-specific serialization parameters."""

    # Default layers to use for HTML export
    layers: set[ContentLayer] = {ContentLayer.BODY}

    # How to handle images
    image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER

    # HTML document properties
    html_lang: str = "en"
    html_head: Optional[str] = None

    css_styles: Optional[str] = None

    add_document_metadata: bool = True
    prettify: bool = True  # Add indentation and line breaks

    # Formula rendering options
    formula_to_mathml: bool = True

    # Allow for different output styles
    output_style: HTMLOutputStyle = HTMLOutputStyle.SINGLE_COLUMN


class HTMLTextSerializer(BaseModel, BaseTextSerializer):
    """HTML-specific text item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        is_inline_scope: bool = False,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed text item to HTML."""
        params = HTMLParams(**kwargs)
        res_parts: list[SerializationResult] = []

        # Prepare the HTML based on item type
        if isinstance(item, TitleItem):
            text_inner = self._prepare_content(item.text)
            text = get_html_tag_with_text_direction(html_tag="h1", text=text_inner)

        elif isinstance(item, SectionHeaderItem):
            section_level = min(item.level + 1, 6)
            text_inner = self._prepare_content(item.text)
            text = get_html_tag_with_text_direction(
                html_tag=f"h{section_level}", text=text_inner
            )

        elif isinstance(item, FormulaItem):
            text = self._process_formula(
                item=item,
                doc=doc,
                image_mode=params.image_mode,
                formula_to_mathml=params.formula_to_mathml,
                is_inline_scope=is_inline_scope,
            )

        elif isinstance(item, CodeItem):
            text = self._process_code(item=item, is_inline_scope=is_inline_scope)

        elif isinstance(item, ListItem):
            # List items are handled by list serializer
            text_inner = self._prepare_content(item.text)
            text = get_html_tag_with_text_direction(html_tag="li", text=text_inner)

        elif is_inline_scope:
            text = self._prepare_content(item.text)
        else:
            # Regular text item
            text_inner = self._prepare_content(item.text)
            text = get_html_tag_with_text_direction(html_tag="p", text=text_inner)

        # Apply formatting and hyperlinks
        text = doc_serializer.post_process(
            text=text,
            formatting=item.formatting,
            hyperlink=item.hyperlink,
        )

        if text:
            text_res = create_ser_result(text=text, span_source=item)
            res_parts.append(text_res)

        if isinstance(item, FloatingItem):
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                res_parts.append(cap_res)

        return create_ser_result(text=text, span_source=res_parts)

    def _prepare_content(
        self, text: str, do_escape_html=True, do_replace_newline=True
    ) -> str:
        """Prepare text content for HTML inclusion."""
        if do_escape_html:
            text = html.escape(text, quote=False)
        if do_replace_newline:
            text = text.replace("\n", "<br>")
        return text

    def _process_code(
        self,
        item: CodeItem,
        is_inline_scope: bool,
    ) -> str:
        code_text = self._prepare_content(
            item.text, do_escape_html=False, do_replace_newline=False
        )
        if is_inline_scope:
            text = f"<code>{code_text}</code>"
        else:
            text = f"<pre><code>{code_text}</code></pre>"

        return text

    def _process_formula(
        self,
        item: FormulaItem,
        doc: DoclingDocument,
        image_mode: ImageRefMode,
        formula_to_mathml: bool,
        is_inline_scope: bool,
    ) -> str:
        """Process a formula item to HTML/MathML."""
        math_formula = self._prepare_content(
            item.text, do_escape_html=False, do_replace_newline=False
        )

        # If formula is empty, try to use an image fallback
        if item.text == "" and item.orig != "":
            img_fallback = self._get_formula_image_fallback(item, doc)
            if (
                image_mode == ImageRefMode.EMBEDDED
                and len(item.prov) > 0
                and img_fallback
            ):
                return img_fallback

        # Try to generate MathML
        if formula_to_mathml and math_formula:
            try:
                # Set display mode based on context
                display_mode = "inline" if is_inline_scope else "block"
                mathml_element = latex2mathml.converter.convert_to_element(
                    math_formula, display=display_mode
                )
                annotation = SubElement(
                    mathml_element, "annotation", dict(encoding="TeX")
                )
                annotation.text = math_formula
                mathml = unescape(tostring(mathml_element, encoding="unicode"))

                # Don't wrap in div for inline formulas
                if is_inline_scope:
                    return mathml
                else:
                    return f"<div>{mathml}</div>"

            except Exception:
                img_fallback = self._get_formula_image_fallback(item, doc)
                if (
                    image_mode == ImageRefMode.EMBEDDED
                    and len(item.prov) > 0
                    and img_fallback
                ):
                    return img_fallback
                elif math_formula:
                    return f"<pre>{math_formula}</pre>"
                else:
                    return "<pre>Formula not decoded</pre>"

        _logger.warning("Could not parse formula with MathML")

        # Fallback options if we got here
        if math_formula and is_inline_scope:
            return f"<code>{math_formula}</code>"
        elif math_formula and (not is_inline_scope):
            f"<pre>{math_formula}</pre>"
        elif is_inline_scope:
            return '<span class="formula-not-decoded">Formula not decoded</span>'

        return '<div class="formula-not-decoded">Formula not decoded</div>'

    def _get_formula_image_fallback(
        self, item: TextItem, doc: DoclingDocument
    ) -> Optional[str]:
        """Try to get an image fallback for a formula."""
        item_image = item.get_image(doc=doc)
        if item_image is not None:
            img_ref = ImageRef.from_pil(item_image, dpi=72)
            return (
                "<figure>" f'<img src="{img_ref.uri}" alt="{item.orig}" />' "</figure>"
            )
        return None


class HTMLTableSerializer(BaseTableSerializer):
    """HTML-specific table item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed table item to HTML."""
        nrows = item.data.num_rows
        ncols = item.data.num_cols

        res_parts: list[SerializationResult] = []
        cap_res = doc_serializer.serialize_captions(item=item, tag="caption", **kwargs)
        if cap_res.text:
            res_parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            body = ""

            for i in range(nrows):
                body += "<tr>"
                for j in range(ncols):
                    cell: TableCell = item.data.grid[i][j]

                    rowspan, rowstart = (
                        cell.row_span,
                        cell.start_row_offset_idx,
                    )
                    colspan, colstart = (
                        cell.col_span,
                        cell.start_col_offset_idx,
                    )

                    if rowstart != i:
                        continue
                    if colstart != j:
                        continue

                    content = html.escape(cell.text.strip())
                    celltag = "td"
                    if cell.column_header:
                        celltag = "th"

                    opening_tag = f"{celltag}"
                    if rowspan > 1:
                        opening_tag += f' rowspan="{rowspan}"'
                    if colspan > 1:
                        opening_tag += f' colspan="{colspan}"'

                    text_dir = get_text_direction(content)
                    if text_dir == "rtl":
                        opening_tag += f' dir="{dir}"'

                    body += f"<{opening_tag}>{content}</{celltag}>"
                body += "</tr>"

            if body:
                body = f"<tbody>{body}</tbody>"
                res_parts.append(create_ser_result(text=body, span_source=item))

        text_res = "".join([r.text for r in res_parts])
        text_res = f"<table>{text_res}</table>" if text_res else ""

        return create_ser_result(text=text_res, span_source=res_parts)


class HTMLPictureSerializer(BasePictureSerializer):
    """HTML-specific picture item serializer."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Export picture to HTML format."""
        params = HTMLParams(**kwargs)

        res_parts: list[SerializationResult] = []

        cap_res = doc_serializer.serialize_captions(
            item=item,
            tag="figcaption",
            **kwargs,
        )
        if cap_res.text:
            res_parts.append(cap_res)

        img_text = ""
        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):

            if params.image_mode == ImageRefMode.EMBEDDED:
                # short-cut: we already have the image in base64
                if (
                    isinstance(item.image, ImageRef)
                    and isinstance(item.image.uri, AnyUrl)
                    and item.image.uri.scheme == "data"
                ):
                    img_text = f'<img src="{item.image.uri}">'
                else:
                    # get the item.image._pil or crop it out of the page-image
                    img = item.get_image(doc)

                    if img is not None:
                        imgb64 = item._image_to_base64(img)
                        img_text = f'<img src="data:image/png;base64,{imgb64}">'
            elif params.image_mode == ImageRefMode.REFERENCED:
                if isinstance(item.image, ImageRef) and not (
                    isinstance(item.image.uri, AnyUrl)
                    and item.image.uri.scheme == "data"
                ):
                    img_text = f'<img src="{quote(str(item.image.uri))}">'
        if img_text:
            res_parts.append(create_ser_result(text=img_text, span_source=item))

        text_res = "".join([r.text for r in res_parts])
        if text_res:
            text_res = f"<figure>{text_res}</figure>"

        return create_ser_result(text=text_res, span_source=res_parts)


class _HTMLGraphDataSerializer:
    """HTML-specific graph-data item serializer."""

    def serialize(
        self,
        *,
        item: Union[FormItem, KeyValueItem],
        graph_data: GraphData,
        class_name: str,
    ) -> SerializationResult:
        """Serialize the graph-data to HTML."""
        # Build cell lookup by ID
        cell_map = {cell.cell_id: cell for cell in graph_data.cells}

        # Build relationship maps
        child_links: dict[int, list[int]] = (
            {}
        )  # source_id -> list of child_ids (to_child)
        value_links: dict[int, list[int]] = {}  # key_id -> list of value_ids (to_value)
        parents: set[int] = (
            set()
        )  # Set of all IDs that are targets of to_child (to find roots)

        for link in graph_data.links:
            if (
                link.source_cell_id not in cell_map
                or link.target_cell_id not in cell_map
            ):
                continue

            if link.label.value == "to_child":
                child_links.setdefault(link.source_cell_id, []).append(
                    link.target_cell_id
                )
                parents.add(link.target_cell_id)
            elif link.label.value == "to_value":
                value_links.setdefault(link.source_cell_id, []).append(
                    link.target_cell_id
                )

        # Find root cells (cells with no parent)
        root_ids = [cell_id for cell_id in cell_map.keys() if cell_id not in parents]

        # Generate the HTML
        parts = [f'<div class="{class_name}">']

        # If we have roots, make a list structure
        if root_ids:
            parts.append(f'<ul class="{class_name}">')
            for root_id in root_ids:
                parts.append(
                    self._render_cell_tree(
                        cell_id=root_id,
                        cell_map=cell_map,
                        child_links=child_links,
                        value_links=value_links,
                        level=0,
                    )
                )
            parts.append("</ul>")

        # If no hierarchy, fall back to definition list
        else:
            parts.append(f'<dl class="{class_name}">')
            for key_id, value_ids in value_links.items():
                key_cell = cell_map[key_id]
                key_text = html.escape(key_cell.text)
                parts.append(f"<dt>{key_text}</dt>")

                for value_id in value_ids:
                    value_cell = cell_map[value_id]
                    value_text = html.escape(value_cell.text)
                    parts.append(f"<dd>{value_text}</dd>")
            parts.append("</dl>")

        parts.append("</div>")

        return create_ser_result(text="\n".join(parts), span_source=item)

    def _render_cell_tree(
        self,
        cell_id: int,
        cell_map: dict,
        child_links: dict,
        value_links: dict,
        level: int,
    ) -> str:
        """Recursively render a cell and its children as a nested list."""
        cell = cell_map[cell_id]
        cell_text = html.escape(cell.text)

        # Format key-value pairs if this cell has values linked
        if cell_id in value_links:
            value_texts = []
            for value_id in value_links[cell_id]:
                if value_id in cell_map:
                    value_cell = cell_map[value_id]
                    value_texts.append(html.escape(value_cell.text))

            cell_text = f"<strong>{cell_text}</strong>: {', '.join(value_texts)}"

        # If this cell has children, create a nested list
        if cell_id in child_links and child_links[cell_id]:
            children_html = []
            children_html.append(f"<li>{cell_text}</li>")
            children_html.append("<ul>")

            for child_id in child_links[cell_id]:
                children_html.append(
                    self._render_cell_tree(
                        cell_id=child_id,
                        cell_map=cell_map,
                        child_links=child_links,
                        value_links=value_links,
                        level=level + 1,
                    )
                )

            children_html.append("</ul>")
            return "\n".join(children_html)

        elif cell_id in value_links:
            return f"<li>{cell_text}</li>"
        else:
            # Leaf node - just render the cell
            # return f'<li>{cell_text}</li>'
            return ""


class HTMLKeyValueSerializer(BaseKeyValueSerializer):
    """HTML-specific key-value item serializer."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed key-value item to HTML."""
        res_parts: list[SerializationResult] = []

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            graph_serializer = _HTMLGraphDataSerializer()

            # Add key-value if available
            kv_res = graph_serializer.serialize(
                item=item,
                graph_data=item.graph,
                class_name="key-value-region",
            )
            if kv_res.text:
                res_parts.append(kv_res)

        # Add caption if available
        cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
        if cap_res.text:
            res_parts.append(cap_res)

        text_res = "\n".join([r.text for r in res_parts])

        return create_ser_result(text=text_res, span_source=res_parts)


class HTMLFormSerializer(BaseFormSerializer):
    """HTML-specific form item serializer."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed form item to HTML."""
        res_parts: list[SerializationResult] = []

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            graph_serializer = _HTMLGraphDataSerializer()

            # Add form if available
            form_res = graph_serializer.serialize(
                item=item,
                graph_data=item.graph,
                class_name="form-container",
            )
            if form_res.text:
                res_parts.append(form_res)

        # Add caption if available
        cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
        if cap_res.text:
            res_parts.append(cap_res)

        text_res = "\n".join([r.text for r in res_parts])

        return create_ser_result(text=text_res, span_source=res_parts)


class HTMLListSerializer(BaseModel, BaseListSerializer):
    """HTML-specific list serializer."""

    @override
    def serialize(
        self,
        *,
        item: Union[UnorderedList, OrderedList],
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs,
    ) -> SerializationResult:
        """Serializes a list to HTML."""
        my_visited: set[str] = visited if visited is not None else set()

        # Get all child parts
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
            **kwargs,
        )

        # Add all child parts
        text_res = "\n".join(
            [
                (
                    p.text
                    if (
                        (p.text.startswith("<li>") and p.text.endswith("</li>"))
                        or (p.text.startswith("<ol>") and p.text.endswith("</ol>"))
                        or (p.text.startswith("<ul>") and p.text.endswith("</ul>"))
                    )
                    else f"<li>{p.text}</li>"
                )
                for p in parts
            ]
        )
        if text_res:
            tag = "ol" if isinstance(item, OrderedList) else "ul"
            text_res = f"<{tag}>\n{text_res}\n</{tag}>"

        return create_ser_result(text=text_res, span_source=parts)


class HTMLInlineSerializer(BaseInlineSerializer):
    """HTML-specific inline group serializer."""

    @override
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs,
    ) -> SerializationResult:
        """Serializes an inline group to HTML."""
        my_visited: set[str] = visited if visited is not None else set()

        # Get all parts with inline scope
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level,
            is_inline_scope=True,
            visited=my_visited,
            **kwargs,
        )

        # Join all parts without separators
        inline_html = " ".join([p.text for p in parts if p.text])

        # Wrap in span if needed
        if inline_html:
            inline_html = f"<span class='inline-group'>{inline_html}</span>"

        return create_ser_result(text=inline_html, span_source=parts)


class HTMLFallbackSerializer(BaseFallbackSerializer):
    """HTML-specific fallback serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Fallback serializer for items not handled by other serializers."""
        if isinstance(item, DocItem):
            return create_ser_result(
                text=f"<!-- Unhandled item type: {item.__class__.__name__} -->",
                span_source=item,
            )
        else:
            # For group items, we don't generate any markup
            return create_ser_result()


class HTMLDocSerializer(DocSerializer):
    """HTML-specific document serializer."""

    text_serializer: BaseTextSerializer = HTMLTextSerializer()
    table_serializer: BaseTableSerializer = HTMLTableSerializer()
    picture_serializer: BasePictureSerializer = HTMLPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = HTMLKeyValueSerializer()
    form_serializer: BaseFormSerializer = HTMLFormSerializer()
    fallback_serializer: BaseFallbackSerializer = HTMLFallbackSerializer()

    list_serializer: BaseListSerializer = HTMLListSerializer()
    inline_serializer: BaseInlineSerializer = HTMLInlineSerializer()

    params: HTMLParams = HTMLParams()

    @override
    def serialize_bold(self, text: str, **kwargs) -> str:
        """Apply HTML-specific bold serialization."""
        return f"<strong>{text}</strong>"

    @override
    def serialize_italic(self, text: str, **kwargs) -> str:
        """Apply HTML-specific italic serialization."""
        return f"<em>{text}</em>"

    @override
    def serialize_underline(self, text: str, **kwargs) -> str:
        """Apply HTML-specific underline serialization."""
        return f"<u>{text}</u>"

    @override
    def serialize_strikethrough(self, text: str, **kwargs) -> str:
        """Apply HTML-specific strikethrough serialization."""
        return f"<del>{text}</del>"

    @override
    def serialize_hyperlink(
        self, text: str, hyperlink: Union[AnyUrl, Path], **kwargs
    ) -> str:
        """Apply HTML-specific hyperlink serialization."""
        return f'<a href="{str(hyperlink)}">{text}</a>'

    @override
    def serialize_doc(
        self, parts: list[SerializationResult], **kwargs
    ) -> SerializationResult:
        """Serialize a document out of its pages."""
        # Create HTML structure
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            self._generate_head(),
            "<body>",
        ]

        if self.params.output_style == HTMLOutputStyle.SPLIT_PAGE:
            html_content = "\n".join([p.text for p in parts if p.text])
            next_page: Optional[int] = None
            prev_full_match_end = 0
            pages = {}
            for full_match, prev_page, next_page in self._get_page_breaks(html_content):
                this_match_start = html_content.find(full_match)
                pages[prev_page] = html_content[prev_full_match_end:this_match_start]
                prev_full_match_end = this_match_start + len(full_match)

            # capture last page
            if next_page is not None:
                pages[next_page] = html_content[prev_full_match_end:]

            html_parts.append("<table>")
            html_parts.append("<tbody>")

            applicable_pages = self._get_applicable_pages()
            for page_no, page in pages.items():

                if isinstance(page_no, int):
                    if applicable_pages is not None and page_no not in applicable_pages:
                        continue
                    page_img = self.doc.pages[page_no].image

                    html_parts.append("<tr>")

                    html_parts.append("<td>")

                    # short-cut: we already have the image in base64
                    if (
                        (page_img is not None)
                        and isinstance(page_img, ImageRef)
                        and isinstance(page_img.uri, AnyUrl)
                        and page_img.uri.scheme == "data"
                    ):
                        img_text = f'<img src="{page_img.uri}">'
                        html_parts.append(f"<figure>{img_text}</figure>")

                    elif (page_img is not None) and (page_img._pil is not None):

                        buffered = BytesIO()
                        page_img._pil.save(
                            buffered, format="PNG"
                        )  # Save the image to the byte stream
                        img_bytes = buffered.getvalue()  # Get the byte data

                        # Encode to Base64 and decode to string
                        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                        img_text = f'<img src="data:image/png;base64,{img_base64}">'

                        html_parts.append(f"<figure>{img_text}</figure>")
                    else:
                        html_parts.append("<figure>no page-image found</figure>")

                    html_parts.append("</td>")

                    html_parts.append("<td>")
                    html_parts.append(f"<div class='page'>\n{page}\n</div>")
                    html_parts.append("</td>")

                    html_parts.append("</tr>")
                else:
                    raise ValueError(
                        "We need page-indices to leverage `split_page_view`"
                    )

            html_parts.append("</tbody>")
            html_parts.append("</table>")

        elif self.params.output_style == HTMLOutputStyle.SINGLE_COLUMN:
            # Add all pages
            html_content = "\n".join([p.text for p in parts if p.text])
            html_content = f"<div class='page'>\n{html_content}\n</div>"
            html_parts.append(html_content)
        else:
            raise ValueError(f"unknown output-style: {self.params.output_style}")

        # Close HTML structure
        html_parts.extend(["</body>", "</html>"])

        # Join with newlines
        html_content = "\n".join(html_parts)

        return create_ser_result(text=html_content, span_source=parts)

    @override
    def serialize_captions(
        self,
        item: FloatingItem,
        tag: str = "figcaption",
        **kwargs,
    ) -> SerializationResult:
        """Serialize the item's captions."""
        params = self.params.merge_with_patch(patch=kwargs)
        results: list[SerializationResult] = []
        text_res = ""
        if DocItemLabel.CAPTION in params.labels:
            results = [
                create_ser_result(text=it.text, span_source=it)
                for cap in item.captions
                if isinstance(it := cap.resolve(self.doc), TextItem)
                and it.self_ref not in self.get_excluded_refs(**kwargs)
            ]
            text_res = params.caption_delim.join([r.text for r in results])
            if text_res:
                text_dir = get_text_direction(text_res)
                dir_str = f' dir="{text_dir}"' if text_dir == "rtl" else ""
                text_res = f"<{tag}{dir_str}>{html.escape(text_res)}</{tag}>"
        return create_ser_result(text=text_res, span_source=results)

    def _generate_head(self) -> str:
        """Generate the HTML head section with metadata and styles."""
        params = self.params

        if self.params.html_head is not None:
            return self.params.html_head

        head_parts = ["<head>", '<meta charset="UTF-8">']

        # Add metadata if requested
        if params.add_document_metadata:
            if self.doc.name:
                head_parts.append(f"<title>{html.escape(self.doc.name)}</title>")
            else:
                head_parts.append("<title>Docling Document</title>")

            head_parts.append(
                '<meta name="generator" content="Docling HTML Serializer">'
            )

        # Add default styles or custom CSS
        if params.css_styles:
            if params.css_styles.startswith("<style>") and params.css_styles.endswith(
                "</style>"
            ):
                head_parts.append(f"\n{params.css_styles}\n")
            else:
                head_parts.append(f"<style>\n{params.css_styles}\n</style>")
        elif self.params.output_style == HTMLOutputStyle.SPLIT_PAGE:
            head_parts.append(_get_css_for_split_page())
        elif self.params.output_style == HTMLOutputStyle.SINGLE_COLUMN:
            head_parts.append(_get_css_for_single_column())
        else:
            raise ValueError(f"unknown output-style: {self.params.output_style}")

        head_parts.append("</head>")

        if params.prettify:
            return "\n".join(head_parts)
        else:
            return "".join(head_parts)

    def _get_default_css(self) -> str:
        """Return default CSS styles for the HTML document."""
        return "<style></style>"

    @override
    def requires_page_break(self):
        """Whether to add page breaks."""
        return self.params.output_style == HTMLOutputStyle.SPLIT_PAGE
