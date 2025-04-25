#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define classes for LaTeX serialization."""
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union

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
    PictureTabularChartData,
    SectionHeaderItem,
    TableCell,
    TableItem,
    TextItem,
    TitleItem,
    UnorderedList,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.utils import get_text_direction

_logger = logging.getLogger(__name__)


def _prepare_latex_content(text: str) -> str:
    """Prepare text content for LaTeX inclusion."""
    # Escape special LaTeX characters
    special_chars = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
        "\\": "\\textbackslash{}",
    }
    for char, escape in special_chars.items():
        text = text.replace(char, escape)
    return text


class LaTeXOutputStyle(str, Enum):
    """LaTeX output style."""

    ARTICLE = "article"
    BOOK = "book"
    REPORT = "report"
    PRESENTATION = "presentation"


class LaTeXParams(CommonParams):
    """LaTeX-specific serialization parameters."""

    # Default layers to use for LaTeX export
    layers: set[ContentLayer] = {ContentLayer.BODY}

    # How to handle images
    image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER

    # LaTeX document properties
    document_class: str = "article"
    document_options: Optional[str] = None
    packages: list[str] = [
        "graphicx",
        "hyperref",
        "amsmath",
        "listings",
        "xcolor",
        "tabularx",
    ]

    # Presentation-specific settings
    presentation_theme: str = "default"
    presentation_color_theme: str = "default"
    presentation_font_theme: str = "default"

    add_document_metadata: bool = True
    prettify: bool = True  # Add indentation and line breaks

    # Enable charts to be printed into LaTeX as tables
    enable_chart_tables: bool = True

    # Control page breaks
    add_page_breaks: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add beamer-specific packages if using presentation style
        if self.document_class == "beamer":
            self.packages.extend([
                "beamer",
                "beamerposter",
                "pgfpages",
            ])


class LaTeXTextSerializer(BaseModel, BaseTextSerializer):
    """LaTeX-specific text item serializer."""

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
        """Serializes the passed text item to LaTeX."""
        params = LaTeXParams(**kwargs)
        res_parts: list[SerializationResult] = []

        # Prepare the LaTeX based on item type
        if isinstance(item, TitleItem):
            text = f"\\title{{{_prepare_latex_content(item.text)}}}"

        elif isinstance(item, SectionHeaderItem):
            section_level = min(item.level + 1, 3)  # LaTeX has fewer section levels
            section_commands = ["\\section", "\\subsection", "\\subsubsection"]
            text = f"{section_commands[section_level - 1]}{{{_prepare_latex_content(item.text)}}}"

        elif isinstance(item, FormulaItem):
            text = self._process_formula(
                item=item,
                doc=doc,
                image_mode=params.image_mode,
                is_inline_scope=is_inline_scope,
            )

        elif isinstance(item, CodeItem):
            text = self._process_code(item=item, is_inline_scope=is_inline_scope)

        elif isinstance(item, ListItem):
            # List items are handled by list serializer
            text = _prepare_latex_content(item.text)

        elif is_inline_scope:
            text = _prepare_latex_content(item.text)
        else:
            # Regular text item
            text = _prepare_latex_content(item.text)

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

    def _process_code(
        self,
        item: CodeItem,
        is_inline_scope: bool,
    ) -> str:
        code_text = _prepare_latex_content(item.text)
        if is_inline_scope:
            text = f"\\texttt{{{code_text}}}"
        else:
            text = f"\\begin{{lstlisting}}\n{code_text}\n\\end{{lstlisting}}"
        return text

    def _process_formula(
        self,
        item: FormulaItem,
        doc: DoclingDocument,
        image_mode: ImageRefMode,
        is_inline_scope: bool,
    ) -> str:
        """Process a formula item to LaTeX."""
        math_formula = _prepare_latex_content(item.text)

        # If formula is empty, try to use an image fallback
        if item.text == "" and item.orig != "":
            img_fallback = self._get_formula_image_fallback(item, doc)
            if (
                image_mode == ImageRefMode.EMBEDDED
                and len(item.prov) > 0
                and img_fallback
            ):
                return img_fallback

        # Use appropriate LaTeX math environment
        if is_inline_scope:
            return f"${math_formula}$"
        else:
            return f"\\begin{{equation}}\n{math_formula}\n\\end{{equation}}"

    def _get_formula_image_fallback(
        self, item: TextItem, doc: DoclingDocument
    ) -> Optional[str]:
        """Try to get an image fallback for a formula."""
        item_image = item.get_image(doc=doc)
        if item_image is not None:
            img_ref = item_image.uri
            return f"\\begin{{figure}}\n\\includegraphics{{{img_ref}}}\n\\caption{{{_prepare_latex_content(item.orig)}}}\n\\end{{figure}}"
        return None


class LaTeXTableSerializer(BaseTableSerializer):
    """LaTeX-specific table item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed table item to LaTeX."""
        nrows = item.data.num_rows
        ncols = item.data.num_cols

        res_parts: list[SerializationResult] = []
        cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
        if cap_res.text:
            res_parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            body = "\\begin{tabularx}{\\textwidth}{|" + "|".join(["X"] * ncols) + "|}\n\\hline\n"

            for i in range(nrows):
                row_parts = []
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

                    content = _prepare_latex_content(cell.text.strip())
                    if cell.column_header:
                        content = f"\\textbf{{{content}}}"

                    if colspan > 1:
                        content = f"\\multicolumn{{{colspan}}}{{|X|}}{{{content}}}"
                    if rowspan > 1:
                        content = f"\\multirow{{{rowspan}}}{{*}}{{{content}}}"

                    row_parts.append(content)

                body += " & ".join(row_parts) + " \\\\\n\\hline\n"

            body += "\\end{tabularx}\n"
            res_parts.append(create_ser_result(text=body, span_source=item))

        text_res = "".join([r.text for r in res_parts])
        return create_ser_result(text=text_res, span_source=res_parts)


class LaTeXPictureSerializer(BasePictureSerializer):
    """LaTeX-specific picture item serializer."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Export picture to LaTeX format."""
        params = LaTeXParams(**kwargs)

        res_parts: list[SerializationResult] = []

        cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
        if cap_res.text:
            res_parts.append(cap_res)

        img_text = ""
        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            if params.image_mode == ImageRefMode.REFERENCED:
                if isinstance(item.image, ImageRef) and not (
                    isinstance(item.image.uri, AnyUrl)
                    and item.image.uri.scheme == "data"
                ):
                    img_text = f"\\includegraphics{{{item.image.uri}}}"

        if img_text:
            res_parts.append(create_ser_result(text=img_text, span_source=item))

        if params.enable_chart_tables:
            # Check if picture has attached PictureTabularChartData
            tabular_chart_annotations = [
                ann
                for ann in item.annotations
                if isinstance(ann, PictureTabularChartData)
            ]
            if len(tabular_chart_annotations) > 0:
                temp_doc = DoclingDocument(name="temp")
                temp_table = temp_doc.add_table(
                    data=tabular_chart_annotations[0].chart_data
                )
                # Use LaTeXTableSerializer to serialize the table
                table_serializer = LaTeXTableSerializer()
                table_res = table_serializer.serialize(
                    item=temp_table,
                    doc_serializer=doc_serializer,
                    doc=temp_doc,
                    **kwargs,
                )
                if table_res.text:
                    res_parts.append(table_res)

        text_res = "".join([r.text for r in res_parts])
        if text_res:
            text_res = f"\\begin{{figure}}\n{text_res}\n\\end{{figure}}"

        return create_ser_result(text=text_res, span_source=res_parts)


class _LaTeXGraphDataSerializer:
    """LaTeX-specific graph-data item serializer."""

    def serialize(
        self,
        *,
        item: Union[FormItem, KeyValueItem],
        graph_data: GraphData,
        class_name: str,
    ) -> SerializationResult:
        """Serialize the graph-data to LaTeX."""
        # Build cell lookup by ID
        cell_map = {cell.cell_id: cell for cell in graph_data.cells}

        # Build relationship maps
        child_links: dict[int, list[int]] = {}
        value_links: dict[int, list[int]] = {}
        parents: set[int] = set()

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

        # Generate the LaTeX
        parts = ["\\begin{itemize}"]

        # If we have roots, make a list structure
        if root_ids:
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

        # If no hierarchy, fall back to description list
        else:
            parts = ["\\begin{description}"]
            for key_id, value_ids in value_links.items():
                key_cell = cell_map[key_id]
                key_text = _prepare_latex_content(key_cell.text)
                parts.append(f"\\item[{key_text}]")

                for value_id in value_ids:
                    value_cell = cell_map[value_id]
                    value_text = _prepare_latex_content(value_cell.text)
                    parts.append(value_text)
            parts.append("\\end{description}")

        parts.append("\\end{itemize}")

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
        cell_text = _prepare_latex_content(cell.text)

        # Format key-value pairs if this cell has values linked
        if cell_id in value_links:
            value_texts = []
            for value_id in value_links[cell_id]:
                if value_id in cell_map:
                    value_cell = cell_map[value_id]
                    value_texts.append(_prepare_latex_content(value_cell.text))

            cell_text = f"\\textbf{{{cell_text}}}: {', '.join(value_texts)}"

        # If this cell has children, create a nested list
        if cell_id in child_links and child_links[cell_id]:
            children_html = [f"\\item {cell_text}", "\\begin{itemize}"]
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
            children_html.append("\\end{itemize}")
            return "\n".join(children_html)
        else:
            return f"\\item {cell_text}"


class LaTeXKeyValueSerializer(BaseKeyValueSerializer):
    """LaTeX-specific key-value item serializer."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed key-value item to LaTeX."""
        res_parts: list[SerializationResult] = []

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            graph_serializer = _LaTeXGraphDataSerializer()

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


class LaTeXFormSerializer(BaseFormSerializer):
    """LaTeX-specific form item serializer."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed form item to LaTeX."""
        res_parts: list[SerializationResult] = []

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            graph_serializer = _LaTeXGraphDataSerializer()

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


class LaTeXListSerializer(BaseModel, BaseListSerializer):
    """LaTeX-specific list serializer."""

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
        """Serializes a list to LaTeX."""
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
                    if p.text.startswith("\\item")
                    else f"\\item {p.text}"
                )
                for p in parts
            ]
        )
        if text_res:
            env = "enumerate" if isinstance(item, OrderedList) else "itemize"
            text_res = f"\\begin{{{env}}}\n{text_res}\n\\end{{{env}}}"

        return create_ser_result(text=text_res, span_source=parts)


class LaTeXInlineSerializer(BaseInlineSerializer):
    """LaTeX-specific inline group serializer."""

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
        """Serializes an inline group to LaTeX."""
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
        inline_latex = " ".join([p.text for p in parts if p.text])

        return create_ser_result(text=inline_latex, span_source=parts)


class LaTeXFallbackSerializer(BaseFallbackSerializer):
    """LaTeX-specific fallback serializer."""

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
                text=f"% Unhandled item type: {item.__class__.__name__}",
                span_source=item,
            )
        else:
            # For group items, we don't generate any markup
            return create_ser_result()


class LaTeXDocSerializer(DocSerializer):
    """LaTeX-specific document serializer."""

    text_serializer: BaseTextSerializer = LaTeXTextSerializer()
    table_serializer: BaseTableSerializer = LaTeXTableSerializer()
    picture_serializer: BasePictureSerializer = LaTeXPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = LaTeXKeyValueSerializer()
    form_serializer: BaseFormSerializer = LaTeXFormSerializer()
    fallback_serializer: BaseFallbackSerializer = LaTeXFallbackSerializer()

    list_serializer: BaseListSerializer = LaTeXListSerializer()
    inline_serializer: BaseInlineSerializer = LaTeXInlineSerializer()

    params: LaTeXParams = LaTeXParams()

    @override
    def requires_page_break(self) -> bool:
        """Whether to add page breaks."""
        return self.params.add_page_breaks and self.params.pages is not None

    @override
    def serialize_bold(self, text: str, **kwargs) -> str:
        """Apply LaTeX-specific bold serialization."""
        return f"\\textbf{{{text}}}"

    @override
    def serialize_italic(self, text: str, **kwargs) -> str:
        """Apply LaTeX-specific italic serialization."""
        return f"\\textit{{{text}}}"

    @override
    def serialize_underline(self, text: str, **kwargs) -> str:
        """Apply LaTeX-specific underline serialization."""
        return f"\\underline{{{text}}}"

    @override
    def serialize_strikethrough(self, text: str, **kwargs) -> str:
        """Apply LaTeX-specific strikethrough serialization."""
        return f"\\sout{{{text}}}"

    @override
    def serialize_hyperlink(
        self, text: str, hyperlink: Union[AnyUrl, Path], **kwargs
    ) -> str:
        """Apply LaTeX-specific hyperlink serialization."""
        return f"\\href{{{str(hyperlink)}}}{{{text}}}"

    @override
    def serialize_doc(
        self, parts: list[SerializationResult], **kwargs
    ) -> SerializationResult:
        """Serialize a document out of its pages."""
        # Create LaTeX document structure
        latex_parts = [
            f"\\documentclass[{self.params.document_options or ''}]{{{self.params.document_class}}}",
        ]

        # Add required packages
        for package in self.params.packages:
            latex_parts.append(f"\\usepackage{{{package}}}")

        # Add presentation-specific settings if using beamer
        if self.params.document_class == "beamer":
            latex_parts.extend([
                f"\\usetheme{{{self.params.presentation_theme}}}",
                f"\\usecolortheme{{{self.params.presentation_color_theme}}}",
                f"\\usefonttheme{{{self.params.presentation_font_theme}}}",
            ])

        # Add document metadata if requested
        if self.params.add_document_metadata and self.doc.name:
            latex_parts.append(f"\\title{{{self.doc.name}}}")

        latex_parts.append("\\begin{document}")

        # Add title page/maketitle
        if self.params.document_class == "beamer":
            latex_parts.append("\\frame{\\titlepage}")
        else:
            latex_parts.append("\\maketitle")

        # Add all pages
        latex_content = "\n".join([p.text for p in parts if p.text])
        latex_parts.append(latex_content)

        # Close document
        latex_parts.append("\\end{document}")

        # Join with newlines
        latex_content = "\n".join(latex_parts)

        return create_ser_result(text=latex_content, span_source=parts)

    @override
    def serialize_captions(
        self,
        item: FloatingItem,
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
                if self.params.document_class == "beamer":
                    text_res = f"\\frametitle{{{text_res}}}"
                else:
                    text_res = f"\\caption{{{text_res}}}"
        return create_ser_result(text=text_res, span_source=results) 