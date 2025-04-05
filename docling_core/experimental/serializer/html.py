#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define classes for HTML serialization."""
import html
import sys
from pathlib import Path
from typing import Optional, Union, List

import latex2mathml.converter
import latex2mathml.exceptions
from pydantic import AnyUrl, BaseModel
from typing_extensions import override
from xml.etree.cElementTree import SubElement, tostring
from xml.sax.saxutils import unescape

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
    GroupItem,
    ImageRef,
    InlineGroup,
    KeyValueItem,
    ListItem,
    NodeItem,
    OrderedList,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
    UnorderedList,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.utils import get_html_tag_with_text_direction, get_text_direction


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

        print(" -> serialising text with label: ", item.label)
        
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
                is_inline_scope=is_inline_scope
            )
            
        elif isinstance(item, CodeItem):
            text = self._process_code(
                item=item,
                is_inline_scope=is_inline_scope
            )
            
        elif isinstance(item, ListItem):
            # List items are handled by list serializer
            text_inner = self._prepare_content(item.text)
            text = get_html_tag_with_text_direction(html_tag="li", text=text_inner)

            print("text in list-item:", text_inner)
            
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
        item: FormulaItem, 
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
            if image_mode == ImageRefMode.EMBEDDED and len(item.prov) > 0 and img_fallback:
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
                if image_mode == ImageRefMode.EMBEDDED and len(item.prov) > 0 and img_fallback:
                    return img_fallback
                elif math_formula:
                    return f"<pre>{math_formula}</pre>"

        _logger.warning("Could not parse formula with MathML")
                
        # Fallback options if we got here
        if math_formula and is_inline_scope:
            return f"<code>{math_formula}</code>"
        elif math_formula and (not is_inline_scope):
            f"<pre>{math_formula}</pre>"
        elif is_inline_scope:
            return '<span class="formula-not-decoded">Formula not decoded</span>'
        else:
            return '<div class="formula-not-decoded">Formula not decoded</div>'
    
    def _get_formula_image_fallback(self, item: TextItem, doc: DoclingDocument) -> Optional[str]:
        """Try to get an image fallback for a formula."""
        item_image = item.get_image(doc=doc)
        if item_image is not None:
            img_ref = ImageRef.from_pil(item_image, dpi=72)
            return (
                "<figure>"
                f'<img src="{img_ref.uri}" alt="{item.orig}" />'
                "</figure>"
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
        text = item.export_to_html(doc=doc, add_caption=True)
        return SerializationResult(text=text)


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
        """Serializes the passed picture item to HTML."""
        params = HTMLParams(**kwargs)
        text = item.export_to_html(
            doc=doc, add_caption=True, image_mode=params.image_mode
        )
        return SerializationResult(text=text)


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
        # This is a placeholder implementation - we could expand it
        # to use a description list (dl/dt/dd) or a table
        return SerializationResult(text="<div class='key-value-region'>Key-value data</div>")


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
        # This is a placeholder implementation
        return SerializationResult(text="<div class='form'>Form data</div>")


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
        my_visited = visited or set()
        
        # Get all child parts
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
            **kwargs,
        )

        print("parts of the list")
        for _ in parts:
            print(" -> list-parts: ", _)
        
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
                print(f"WARNING: no <li> for {part.text}")
                list_html.append(f"<li>{part.text}</li>")
                
        # Close the list
        list_html.append(f"</{tag}>")

        print(" => list: ", " ".join(list_html))
        
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
        my_visited = visited or set()
        
        # Get all parts with inline scope
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level,
            is_inline_scope=True,
            visited=my_visited,
            **kwargs,
        )

        for _ in parts:
            print("inline-parts: ", _)
        
        # Join all parts without separators
        inline_html = " ".join([p.text for p in parts])
        
        # Wrap in span if needed
        if inline_html:
            inline_html = f"<span class='inline-group'>{inline_html}</span>"

        print(" => inline: ", inline_html)
            
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
        # For group items, we don't generate any markup
        if isinstance(item, GroupItem):
            return SerializationResult(text="")
            
        # For other doc items, add a comment
        return SerializationResult(text=f"<!-- Unhandled item type: {item.__class__.__name__} -->")


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
        **kwargs,
    ) -> SerializationResult:
        """Serialize the item's captions."""
        caption_parts = []
        
        # Extract caption text from all caption items
        for cap in item.captions:
            caption_item = cap.resolve(self.doc)
            if isinstance(caption_item, TextItem):
                caption_parts.append(caption_item.text)
                
        # Join all captions with a space
        if caption_parts:
            caption_text = " ".join(caption_parts)
            text_dir = get_text_direction(caption_text)
            
            # Create proper HTML
            if text_dir == "rtl":
                return SerializationResult(
                    text=f'<figcaption dir="{text_dir}">{html.escape(caption_text)}</figcaption>'
                )
            else:
                return SerializationResult(
                    text=f'<figcaption>{html.escape(caption_text)}</figcaption>'
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
    
