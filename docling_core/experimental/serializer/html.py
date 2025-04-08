#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define classes for HTML serialization."""
import html
import logging
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote
from xml.etree.cElementTree import SubElement, tostring
from xml.sax.saxutils import unescape

import latex2mathml.converter
import latex2mathml.exceptions
from pydantic import AnyUrl, BaseModel
from typing_extensions import override

from docling_core.experimental.serializer.base import (
    BaseDocSerializer,
    BaseFallbackSerializer,
    BaseFormSerializer,
    BaseGraphDataSerializer,
    BaseInlineSerializer,
    BaseKeyValueSerializer,
    BaseListSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.experimental.serializer.common import CommonParams, DocSerializer
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    CodeItem,
    ContentLayer,
    DoclingDocument,
    FloatingItem,
    FormItem,
    FormulaItem,
    GraphData,
    GroupItem,
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
from docling_core.types.doc.utils import (
    get_html_tag_with_text_direction,
    get_text_direction,
)

_logger = logging.getLogger(__name__)


class HTMLParams(CommonParams):
    """HTML-specific serialization parameters."""

    # Default layers to use for HTML export
    layers: set[ContentLayer] = {ContentLayer.BODY}

    # How to handle images
    image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER

    # HTML document properties
    html_lang: str = "en"
    css_styles: Optional[str] = None
    add_document_metadata: bool = True
    prettify: bool = True  # Add indentation and line breaks

    # Formula rendering options
    formula_to_mathml: bool = True

    # Allow for split page view (only possible if page-images are present)
    split_page_view: bool = False

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

        print(f"HTMLTextSerializer {item.get_ref().cref}: {item.label} -> {item.text[0:64]}")
        
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

        return SerializationResult(text=text)

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
            item.text, do_escape_html=True, do_replace_newline=False
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
        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return SerializationResult(text="")

        print(f"HTMLTableSerializer {item.get_ref().cref}: {item.label}")
        
        text = self._serialize_table(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            add_caption=True,
            add_footnotes=True,
        )
        return SerializationResult(text=text)
    
    def _serialize_table(
        self,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        add_caption: bool = True,
        add_footnotes: bool = True,
    ) -> str:
        """Export the table as html."""
        nrows = item.data.num_rows
        ncols = item.data.num_cols

        caption_text = doc_serializer.serialize_captions(item=item, tag="caption")
        print(caption_text)
        
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

        if len(caption_text.text) > 0 and len(body) > 0:
            body = f"<table>{caption_text.text}<tbody>{body}</tbody></table>"
        elif len(caption_text.text) == 0 and len(body) > 0:
            body = f"<table><tbody>{body}</tbody></table>"
        elif len(caption_text.text) > 0 and len(body) == 0:
            body = f"<table>{caption_text.text}</table>"
        else:
            body = "<table></table>"

        return body


class HTMLPictureSerializer(BasePictureSerializer):
    """HTML-specific picture item serializer."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        visited: Optional[set[str]] = None,
        add_caption: bool = True,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        **kwargs,
    ) -> SerializationResult:
        """Export picture to HTML format."""
        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return SerializationResult(text="")
        
        print(f"HTMLPictureSerializer {item.get_ref().cref}: {item.label}")
        
        caption = doc_serializer.serialize_captions(
            item=item, doc_serializer=doc_serializer, doc=doc, tag="figcaption"
        )
        
        result = ""

        if image_mode == ImageRefMode.PLACEHOLDER:
            result = f"<figure>{caption.text}</figure>"

        elif image_mode == ImageRefMode.EMBEDDED:
            # short-cut: we already have the image in base64
            if (
                isinstance(item.image, ImageRef)
                and isinstance(item.image.uri, AnyUrl)
                and item.image.uri.scheme == "data"
            ):
                img_text = f'<img src="{item.image.uri}">'
                result = f"<figure>{caption.text}{img_text}</figure>"
            else:
                # get the item.image._pil or crop it out of the page-image
                img = item.get_image(doc)

                if img is not None:
                    imgb64 = item._image_to_base64(img)
                    img_text = f'<img src="data:image/png;base64,{imgb64}">'

                    result = f"<figure>{caption.text}{img_text}</figure>"
                else:
                    result = f"<figure>{caption.text}</figure>"

        elif image_mode == ImageRefMode.REFERENCED:

            if not isinstance(item.image, ImageRef) or (
                isinstance(item.image.uri, AnyUrl) and item.image.uri.scheme == "data"
            ):
                result = f"<figure>{caption.text}</figure>"

            else:
                img_text = f'<img src="{quote(str(item.image.uri))}">'
                result = f"<figure>{caption.text}{img_text}</figure>"
        else:
            result = f"<figure>{caption.text}</figure>"
        
        return SerializationResult(text=result)


class HTMLGraphDataSerializer(BaseGraphDataSerializer):
    """HTML-specific graph-data item serializer."""

    @override
    def serialize(
        self,
        *,
        item: GraphData,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        tag: str,
        **kwargs,
    ) -> SerializationResult:
        print("HTMLGraphDataSerializer")
        
        """Serialize the graph-data to HTML."""
        # Build cell lookup by ID
        cell_map = {cell.cell_id: cell for cell in item.cells}

        # Build relationship maps
        child_links: dict[int, list[int]] = (
            {}
        )  # source_id -> list of child_ids (to_child)
        value_links: dict[int, list[int]] = {}  # key_id -> list of value_ids (to_value)
        parents: set[int] = (
            set()
        )  # Set of all IDs that are targets of to_child (to find roots)

        for link in item.links:
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
        parts = [f'<div class="{tag}">']

        # If we have roots, make a list structure
        if root_ids:
            parts.append(f'<ul class="{tag}">')
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
            parts.append(f'<dl class="{tag}">')
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

        return SerializationResult(text="\n".join(parts))

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
        print(f"HTMLKeyValueSerializer {item.get_ref().cref}: {item.label}")
        
        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return SerializationResult(text="")

        graph_serializer = HTMLGraphDataSerializer()

        # Add key-value if available
        key_value = graph_serializer.serialize(
            item=item.graph,
            doc_serializer=doc_serializer,
            doc=doc,
            tag="key-value-region",
        )

        # Add caption if available
        caption = doc_serializer.serialize_captions(item=item, **kwargs)

        return SerializationResult(text="\n".join([key_value.text, caption.text]))


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
        print(f"HTMLFormSerializer {item.get_ref().cref}: {item.label}")
        
        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return SerializationResult(text="")

        graph_serializer = HTMLGraphDataSerializer()

        # Add key-value if available
        key_value = graph_serializer.serialize(
            item=item.graph,
            doc_serializer=doc_serializer,
            doc=doc,
            tag="form-container",
        )

        # Add caption if available
        caption = doc_serializer.serialize_captions(item=item, **kwargs)

        return SerializationResult(text="\n".join([key_value.text, caption.text]))


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
        print(f"HTMLListSerializer {item.get_ref().cref}: {item.label}")
                
        my_visited: set[str] = visited if visited is not None else set()

        # Get all child parts
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
            **kwargs,
        )

        if len(parts)==0:
            print(f" => no list-items found for {item.get_ref().cref}")
            return SerializationResult(text="")            
        
        # Start the appropriate list type
        tag = "ol" if isinstance(item, OrderedList) else "ul"
        list_html = [f"<{tag}>"]

        # Add all child parts
        for part in parts:
            if part.text.startswith("<li>") and part.text.endswith("</li>"):
                list_html.append(part.text)
            elif part.text.startswith("<ol>") and part.text.endswith("</ol>"):
                list_html.append(part.text)
            elif part.text.startswith("<ul>") and part.text.endswith("</ul>"):
                list_html.append(part.text)
            else:
                _logger.info(f"no <li>, <ol> or <ul> for {part.text}")
                list_html.append(f"<li>{part.text}</li>")

        # Close the list
        list_html.append(f"</{tag}>")

        return SerializationResult(text="\n".join(list_html))


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
        print(f"HTMLInlineSerializer: {item.label}: {visited}")
        
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
        inline_html = " ".join([p.text for p in parts])

        # Wrap in span if needed
        if inline_html:
            inline_html = f"<span class='inline-group'>{inline_html}</span>"

        return SerializationResult(text=inline_html)


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
        print(f"HTMLFallbackSerializer {item.get_ref().cref}: {item.label}")
        
        # For group items, we don't generate any markup
        if isinstance(item, GroupItem):
            return SerializationResult(text="")

        # For other doc items, add a comment
        return SerializationResult(
            text=f"<!-- Unhandled item type: {item.__class__.__name__} -->"
        )


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
    def serialize_page(self, parts: list[SerializationResult]) -> SerializationResult:
        """Serialize a page out of its parts."""
        # Join all parts with newlines
        body_content = "\n".join([p.text for p in parts if p.text])
        return SerializationResult(text=f"<div class='page'>\n{body_content}\n</div>")

    @override
    def serialize_doc(self, pages: list[SerializationResult]) -> SerializationResult:
        """Serialize a document out of its pages."""
        # Create HTML structure
        html_parts = [
            "<!DOCTYPE html>",
            self._generate_head(),
            "<body>",
        ]

        # Add all pages
        for page in pages:
            if page.text:
                html_parts.append(page.text)

        # Close HTML structure
        html_parts.extend(["</body>", "</html>"])

        # Join with newlines
        html_content = "\n".join(html_parts)

        return SerializationResult(text=html_content)

    @override
    def serialize_captions(
        self,
        item: FloatingItem,
        tag: str = "figcaption",
        **kwargs,
    ) -> SerializationResult:
        """Serialize the item's captions."""
        print(f"serialize_captions: {item.label}")

        caption_parts = []

        # Extract caption text from all caption items
        for cap in item.captions:
            caption_item = cap.resolve(self.doc)
            if isinstance(caption_item, TextItem):
                caption_parts.append(caption_item.text)

        # Join all captions with a space
        if len(caption_parts)>0:
            caption_text = " ".join(caption_parts)
            text_dir = get_text_direction(caption_text)

            # Create proper HTML
            if text_dir == "rtl":
                return SerializationResult(
                    text=f'<{tag} dir="{text_dir}">{html.escape(caption_text)}</{tag}>'
                )
            else:
                return SerializationResult(
                    text=f"<{tag}>{html.escape(caption_text)}</{tag}>"
                )
        
        return SerializationResult(text="")

    def _generate_head(self) -> str:
        """Generate the HTML head section with metadata and styles."""
        params = self.params

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
            head_parts.append(f"<style>\n{params.css_styles}\n</style>")
        else:
            head_parts.append(self._get_default_css())

        head_parts.append("</head>")

        if params.prettify:
            return "\n".join(head_parts)
        else:
            return "".join(head_parts)

    def _get_default_css(self) -> str:
        """Return default CSS styles for the HTML document."""
        return """<style>
    html {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
        line-height: 1.6;
    }
    body {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        background-color: white;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    h1 {
        font-size: 2em;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.3em;
    }
    table {
        border-collapse: collapse;
        margin: 1em 0;
        width: 100%;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    figure {
        margin: 1.5em 0;
        text-align: center;
    }
    figcaption {
        color: #666;
        font-style: italic;
        margin-top: 0.5em;
    }
    img {
        max-width: 100%;
        height: auto;
    }
    pre {
        background-color: #f6f8fa;
        border-radius: 3px;
        padding: 1em;
        overflow: auto;
    }
    code {
        font-family: monospace;
        background-color: #f6f8fa;
        padding: 0.2em 0.4em;
        border-radius: 3px;
    }
    pre code {
        background-color: transparent;
        padding: 0;
    }
    .formula {
        text-align: center;
        padding: 0.5em;
        margin: 1em 0;
        background-color: #f9f9f9;
    }
    .formula-not-decoded {
        text-align: center;
        padding: 0.5em;
        margin: 1em 0;
        background: repeating-linear-gradient(
            45deg,
            #f0f0f0,
            #f0f0f0 10px,
            #f9f9f9 10px,
            #f9f9f9 20px
        );
    }
    .page-break {
        page-break-after: always;
        border-top: 1px dashed #ccc;
        margin: 2em 0;
    }
    .key-value-region {
        background-color: #f9f9f9;
        padding: 1em;
        border-radius: 4px;
        margin: 1em 0;
    }
    .key-value-region dt {
        font-weight: bold;
    }
    .key-value-region dd {
        margin-left: 1em;
        margin-bottom: 0.5em;
    }
    .form-container {
        border: 1px solid #ddd;
        padding: 1em;
        border-radius: 4px;
        margin: 1em 0;
    }
    .form-item {
        margin-bottom: 0.5em;
    }
    .image-classification {
        font-size: 0.9em;
        color: #666;
        margin-top: 0.5em;
    }
</style>"""
