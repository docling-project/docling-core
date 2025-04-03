#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define classes for HTML serialization."""
import html
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from docling_core.experimental.serializer.common import CommonParams, DocSerializer
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    CodeItem,
    ContentLayer,
    DocItem,
    DoclingDocument,
    FloatingItem,
    Formatting,
    FormItem,
    FormulaItem,
    ImageRef,
    InlineGroup,
    KeyValueItem,
    NodeItem,
    OrderedList,
    PictureClassificationData,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
    UnorderedList,
)
from docling_core.types.doc.utils import get_html_tag_with_text_direction, get_text_direction


class HTMLParams(CommonParams):
    """HTML-specific serialization parameters."""

    layers: set[ContentLayer] = {ContentLayer.BODY}
    image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER
    image_placeholder: str = "<!-- image -->"
    add_page_break: bool = True
    page_break_placeholder: str = '<div class="page-break"></div>'
    css_styles: Optional[str] = None
    html_lang: str = "en"
    formula_to_mathml: bool = True
    add_document_metadata: bool = True
    prettify: bool = True  # Add indentation and line breaks
    add_image_dimensions: bool = True  # Add width and height attributes to images


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
        """Serializes the passed item."""
        params = HTMLParams(**kwargs)
        parts: List[str] = []
        
        text_content = html.escape(item.text, quote=False)
        
        # Replace newlines with <br> tags if not in code or formula
        if not isinstance(item, (CodeItem, FormulaItem)):
            text_content = text_content.replace("\n", "<br>")
            
        if isinstance(item, TitleItem):
            text = get_html_tag_with_text_direction(html_tag="h1", text=text_content)
        elif isinstance(item, SectionHeaderItem):
            section_level = min(item.level + 1, 6)  # h1-h6 are valid in HTML
            text = get_html_tag_with_text_direction(
                html_tag=f"h{section_level}", text=text_content
            )
        elif isinstance(item, CodeItem):
            language_attr = ""
            if item.code_language.value != "unknown":
                language_attr = f' class="language-{item.code_language.value.lower()}"'
            
            if is_inline_scope:
                text = f'<code{language_attr}>{text_content}</code>'
            else:
                text = f'<pre><code{language_attr}>{text_content}</code></pre>'
        elif isinstance(item, FormulaItem):
            if params.formula_to_mathml and item.text:
                # Simplified formula handling - full implementation would use latex2mathml
                text = f'<div class="formula">{text_content}</div>'
            elif item.text:
                text = f'<div class="formula">{text_content}</div>'
            elif item.orig:
                text = '<div class="formula-not-decoded">Formula not decoded</div>'
            else:
                text = ''
        else:
            # Regular text
            text = get_html_tag_with_text_direction(html_tag="p", text=text_content)
        
        parts.append(text)

        # Handle captions for floating items
        if isinstance(item, FloatingItem):
            cap_text = doc_serializer.serialize_captions(item=item, **kwargs).text
            if cap_text:
                parts.append(cap_text)

        text_res = "\n".join(parts)
        text_res = doc_serializer.post_process(
            text=text_res,
            formatting=item.formatting,
            hyperlink=item.hyperlink,
        )
        
        return SerializationResult(text=text_res)


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
        """Serializes the passed item."""
        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return SerializationResult(text="")
            
        # Process captions first
        cap_text = doc_serializer.serialize_captions(item=item, **kwargs).text
        
        # Start building the table
        rows = []
        
        for i, row in enumerate(item.data.grid):
            row_cells = []
            for j, cell in enumerate(row):
                # Skip cells that are covered by rowspan or colspan from previous cells
                if cell.start_row_offset_idx != i or cell.start_col_offset_idx != j:
                    continue
                    
                content = html.escape(cell.text.strip())
                celltag = "th" if cell.column_header or cell.row_header else "td"
                
                attrs = []
                if cell.row_span > 1:
                    attrs.append(f'rowspan="{cell.row_span}"')
                if cell.col_span > 1:
                    attrs.append(f'colspan="{cell.col_span}"')
                    
                text_dir = get_text_direction(content)
                if text_dir == "rtl":
                    attrs.append(f'dir="{text_dir}"')
                    
                attrs_str = " ".join(attrs)
                if attrs_str:
                    attrs_str = " " + attrs_str
                    
                row_cells.append(f"<{celltag}{attrs_str}>{content}</{celltag}>")
                
            if row_cells:
                rows.append(f"<tr>{''.join(row_cells)}</tr>")
        
        tbody = f"<tbody>{''.join(rows)}</tbody>" if rows else ""
        
        if cap_text:
            table = f"<table>{cap_text}{tbody}</table>"
        elif tbody:
            table = f"<table>{tbody}</table>"
        else:
            table = "<table></table>"
            
        return SerializationResult(text=table)


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
        """Serializes the passed item."""
        params = HTMLParams(**kwargs)
        
        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return SerializationResult(text="")
            
        cap_text = doc_serializer.serialize_captions(item=item, **kwargs).text
        
        # Process the image based on image_mode
        img_text = self._get_image_html(item, doc, params)
        
        # Add classification info if available
        classification_text = ""
        for annotation in item.annotations:
            if isinstance(annotation, PictureClassificationData) and annotation.predicted_classes:
                class_name = annotation.predicted_classes[0].class_name
                confidence = annotation.predicted_classes[0].confidence
                classification_text = f'<div class="image-classification">{html.escape(class_name)} ({confidence:.2f})</div>'
                break
                
        figure = f"<figure>{img_text}{classification_text}{cap_text}</figure>"
        return SerializationResult(text=figure)
        
    def _get_image_html(self, item: PictureItem, doc: DoclingDocument, params: HTMLParams) -> str:
        """Generate HTML for the image based on image mode."""
        if params.image_mode == ImageRefMode.PLACEHOLDER:
            return params.image_placeholder
            
        elif params.image_mode == ImageRefMode.EMBEDDED:
            # Try to use the embedded image
            if (item.image is not None and 
                isinstance(item.image.uri, AnyUrl) and 
                item.image.uri.scheme == "data"):
                return self._create_img_tag(item.image.uri, item, params)
                
            # Try to get the image from document
            img = item.get_image(doc)
            if img is not None:
                imgb64 = item._image_to_base64(img)
                return self._create_img_tag(f"data:image/png;base64,{imgb64}", item, params)
                
            return params.image_placeholder
            
        elif params.image_mode == ImageRefMode.REFERENCED:
            if item.image is not None:
                if isinstance(item.image.uri, AnyUrl) and item.image.uri.scheme != "data":
                    return self._create_img_tag(item.image.uri, item, params)
                elif isinstance(item.image.uri, Path):
                    return self._create_img_tag(item.image.uri, item, params)
                    
            return params.image_placeholder
            
        return params.image_placeholder
        
    def _create_img_tag(self, src: Union[str, AnyUrl, Path], item: PictureItem, params: HTMLParams) -> str:
        """Create an HTML img tag with appropriate attributes."""
        attrs = [f'src="{src}"', 'alt="Image"']
        
        if params.add_image_dimensions and item.image is not None:
            attrs.append(f'width="{item.image.size.width}"')
            attrs.append(f'height="{item.image.size.height}"')
            
        return f"<img {' '.join(attrs)}>"


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
        """Serializes the passed item."""
        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return SerializationResult(text="")
            
        # Create a definition list (dl) for key-value pairs
        parts = ['<dl class="key-value-region">']
        
        # Group cells by their keys
        key_to_values: Dict[int, List[int]] = {}
        for link in item.graph.links:
            key_to_values.setdefault(link.source_cell_id, []).append(link.target_cell_id)
            
        # Find all cells
        cell_by_id = {cell.cell_id: cell for cell in item.graph.cells}
        
        # Process each key-value pair
        for key_id, value_ids in key_to_values.items():
            if key_id in cell_by_id:
                key_cell = cell_by_id[key_id]
                key_text = html.escape(key_cell.text)
                parts.append(f'<dt>{key_text}</dt>')
                
                for value_id in value_ids:
                    if value_id in cell_by_id:
                        value_cell = cell_by_id[value_id]
                        value_text = html.escape(value_cell.text)
                        parts.append(f'<dd>{value_text}</dd>')
        
        parts.append('</dl>')
        
        # Add caption if available
        cap_text = doc_serializer.serialize_captions(item=item, **kwargs).text
        if cap_text:
            parts.append(cap_text)
            
        return SerializationResult(text="\n".join(parts))


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
        """Serializes the passed item."""
        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return SerializationResult(text="")
            
        # Create a form representation (non-functional HTML form)
        parts = ['<div class="form-container">']
        
        # Simple representation of form items
        for cell in item.graph.cells:
            cell_text = html.escape(cell.text)
            cell_label = cell.label.value
            parts.append(f'<div class="form-item form-item-{cell_label}">{cell_text}</div>')
            
        parts.append('</div>')
        
        # Add caption if available
        cap_text = doc_serializer.serialize_captions(item=item, **kwargs).text
        if cap_text:
            parts.append(cap_text)
            
        return SerializationResult(text="\n".join(parts))


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
        """Serializes the passed item."""
        my_visited = visited or set()
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
            **kwargs,
        )
        
        # Determine list type
        tag = "ol" if isinstance(item, OrderedList) else "ul"
        
        # Build list items
        items = []
        for part in parts:
            if part.text:
                # If the part is already wrapped in <li>, use it directly
                if part.text.startswith("<li") and part.text.endswith("</li>"):
                    items.append(part.text)
                else:
                    # Otherwise wrap it in <li>
                    items.append(f"<li>{part.text}</li>")
                    
        list_html = f"<{tag}>{''.join(items)}</{tag}>"
        return SerializationResult(text=list_html)


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
        """Serializes the passed item."""
        my_visited = visited or set()
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level,
            is_inline_scope=True,
            visited=my_visited,
            **kwargs,
        )
        
        # Join parts with spaces for inline content
        inline_content = " ".join([p.text for p in parts if p.text])
        if inline_content:
            return SerializationResult(text=f"<span>{inline_content}</span>")
        return SerializationResult(text="")


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
        """Serializes the passed item."""
        if isinstance(item, DocItem):
            return SerializationResult(text=f"<!-- Unsupported item type: {item.label} -->")
        return SerializationResult(text="")


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
        return f"<s>{text}</s>"

    @override
    def serialize_hyperlink(self, text: str, hyperlink: Union[AnyUrl, Path], **kwargs) -> str:
        """Apply HTML-specific hyperlink serialization."""
        return f'<a href="{hyperlink}">{text}</a>'

    @override
    def serialize_page(self, parts: list[SerializationResult]) -> SerializationResult:
        """Serialize a page out of its parts."""
        params = self.params
        if params.prettify:
            text_res = "\n".join([p.text for p in parts if p.text])
        else:
            text_res = "".join([p.text for p in parts if p.text])
        return SerializationResult(text=text_res)

    @override
    def serialize_doc(self, pages: list[SerializationResult]) -> SerializationResult:
        """Serialize a document out of its pages."""
        params = self.params
        
        # Join pages with page breaks if specified
        if params.add_page_break and params.page_break_placeholder:
            page_sep = f"\n{params.page_break_placeholder}\n"
            content = page_sep.join([p.text for p in pages if p.text])
        else:
            content = self.serialize_page(parts=pages).text
            
        # Add HTML document structure
        head = self._generate_head()
        body = f"<body>\n{content}\n</body>"
        
        # Create full HTML document
        html_doc = f"<!DOCTYPE html>\n<html lang=\"{params.html_lang}\">\n{head}\n{body}\n</html>"
        
        return SerializationResult(text=html_doc)
        
    def _generate_head(self) -> str:
        """Generate the HTML head section with metadata and styles."""
        params = self.params
        
        head_parts = ['<head>', '<meta charset="UTF-8">']
        
        # Add metadata if requested
        if params.add_document_metadata:
            if self.doc.name:
                head_parts.append(f'<title>{html.escape(self.doc.name)}</title>')
            else:
                head_parts.append('<title>Docling Document</title>')
                
            head_parts.append('<meta name="generator" content="Docling HTML Serializer">')
            
        # Add default styles or custom CSS
        if params.css_styles:
            head_parts.append(f'<style>\n{params.css_styles}\n</style>')
        else:
            head_parts.append(self._get_default_css())
            
        head_parts.append('</head>')
        
        if params.prettify:
            return '\n'.join(head_parts)
        else:
            return ''.join(head_parts)
            
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

    @override
    def serialize_captions(
        self,
        item: FloatingItem,
        **kwargs,
    ) -> SerializationResult:
        """Serialize the item's captions."""
        params = HTMLParams(**kwargs)
        
        caption_parts = []
        for cap_ref in item.captions:
            cap_item = cap_ref.resolve(self.doc)
            if isinstance(cap_item, TextItem) and cap_item.self_ref not in self.get_excluded_refs(**kwargs):
                caption_text = html.escape(cap_item.text)
                caption_parts.append(caption_text)
                
        if caption_parts:
            caption_text = " ".join(caption_parts)
            result = f"<figcaption>{caption_text}</figcaption>"
            return SerializationResult(text=result)
            
        return SerializationResult(text="")
