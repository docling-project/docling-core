#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define classes for LaTeX serialization."""

import re
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import AnyUrl, BaseModel
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseAnnotationSerializer,
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
from docling_core.transforms.serializer.common import (
    CommonParams,
    DocSerializer,
    _get_annotation_text,
    create_ser_result,
)
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    CodeItem,
    ContentLayer,
    DescriptionAnnotation,
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
    ListGroup,
    ListItem,
    NodeItem,
    PictureClassificationData,
    PictureItem,
    PictureMoleculeData,
    PictureTabularChartData,
    RichTableCell,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)


class LaTeXParams(CommonParams):
    """LaTeX-specific serialization parameters."""

    layers: set[ContentLayer] = {ContentLayer.BODY}

    image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER
    image_placeholder: str = "% image"

    enable_chart_tables: bool = True
    include_annotations: bool = True

    indent: int = 2  # spaces for nested lists

    # If not None, emitted where page breaks occur (e.g., "\\newpage")
    page_break_command: Optional[str] = None

    # Escape LaTeX special characters in text
    escape_latex: bool = True

    # Optional LaTeX preamble configuration
    # When provided, emitted before the document environment
    # Example: "\\documentclass[11pt,a4paper]{article}"
    document_class: str = r"\documentclass[11pt,a4paper]{article}"
    # List of packages to include. Accepts either full lines
    # like "\\usepackage{graphicx}" or bare package names like "graphicx".
    packages: list[str] = [
        r"\usepackage[utf8]{inputenc} % allow utf-8 input",
        r"\usepackage[T1]{fontenc}    % use 8-bit T1 fonts",
        r"\usepackage{hyperref}       % hyperlinks",
        r"\usepackage{url}            % simple URL typesetting",
        r"\usepackage{booktabs}       % professional-quality tables",
        r"\usepackage{array}          % extended column defs",
        r"\usepackage{multirow}       % multirow cells",
        r"\usepackage{makecell}       % line breaks in cells",
        r"\usepackage{amsfonts}       % blackboard math symbols",
        r"\usepackage{nicefrac}       % compact symbols for 1/2, etc.",
        r"\usepackage{microtype}      % microtypography",
        r"\usepackage{xcolor}         % colors",
        r"\usepackage{graphicx}       % graphics",
    ]


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in text.

    Note: Do not use inside math or verbatim contexts.
    """
    if not text:
        return text
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    res = []
    for ch in text:
        res.append(replacements.get(ch, ch))
    return "".join(res)


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
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a ``TextItem`` into LaTeX, handling lists, titles, headers, code and formulas.

        Applies post-processing (escape, formatting, hyperlinks) when appropriate and
        returns a ``SerializationResult`` ready to be joined into the document.
        """
        LaTeXParams(**kwargs)
        parts: list[SerializationResult] = []

        # Inline group passthrough
        has_inline_repr = (
            item.text == ""
            and len(item.children) == 1
            and isinstance((child_group := item.children[0].resolve(doc)), InlineGroup)
        )
        if has_inline_repr:
            text = doc_serializer.serialize(item=child_group, is_inline_scope=True).text
            post_process = False
        else:
            text = item.text
            post_process = True

        if isinstance(item, (ListItem, TitleItem, SectionHeaderItem)):
            # For list items, defer environment wrapping to list serializer
            if isinstance(item, ListItem):
                if post_process:
                    text = doc_serializer.post_process(
                        text=text,
                        formatting=item.formatting,
                        hyperlink=item.hyperlink,
                    )
                text_part = f"\\item {text}"
                post_process = False
            elif isinstance(item, TitleItem):
                # Emit document title using \title{...}
                if post_process:
                    text = doc_serializer.post_process(
                        text=text,
                        formatting=item.formatting,
                        hyperlink=item.hyperlink,
                    )
                text_part = f"\\title{{{text}}}"
                post_process = False
            else:
                # Section headers: level 1->section, 2->subsection, 3->subsubsection
                # Raise error for unsupported levels
                if post_process:
                    text = doc_serializer.post_process(
                        text=text,
                        formatting=item.formatting,
                        hyperlink=item.hyperlink,
                    )
                lvl = item.level
                if lvl <= 0 or lvl >= 4:
                    raise ValueError(
                        "LaTeX serializer: SectionHeaderItem.level must be in [1, 3]"
                    )
                cmd = {1: "section", 2: "subsection", 3: "subsubsection"}[lvl]
                text_part = f"\\{cmd}{{{text}}}"
                post_process = False

        elif isinstance(item, CodeItem):
            # Inline vs block code
            if is_inline_scope:
                text_part = f"\\texttt{{{text}}}"
            else:
                text_part = f"\\begin{{verbatim}}\n{text}\n\\end{{verbatim}}"
            post_process = False

        elif isinstance(item, FormulaItem):
            if text:
                text_part = f"${text}$" if is_inline_scope else f"$${text}$$"
            elif item.orig:
                text_part = "% formula-not-decoded"
            else:
                text_part = ""
            post_process = False

        else:
            # Regular paragraph or inline text
            if post_process:
                text = doc_serializer.post_process(
                    text=text,
                    formatting=item.formatting,
                    hyperlink=item.hyperlink,
                )
            text_part = text if is_inline_scope else text

        if text_part:
            parts.append(create_ser_result(text=text_part, span_source=item))

        if isinstance(item, FloatingItem):
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                parts.append(cap_res)

        joined = (" " if is_inline_scope else "\n\n").join([p.text for p in parts])

        return create_ser_result(text=joined, span_source=parts)


class LaTeXAnnotationSerializer(BaseModel, BaseAnnotationSerializer):
    """LaTeX-specific annotation serializer."""

    def serialize(
        self,
        *,
        item: DocItem,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize supported annotations of ``item`` as LaTeX comments."""
        params = LaTeXParams(**kwargs)
        res_parts: list[SerializationResult] = []
        if not params.include_annotations:
            return create_ser_result()
        for ann in item.get_annotations():
            if isinstance(
                ann,
                (
                    PictureClassificationData,
                    DescriptionAnnotation,
                    PictureMoleculeData,
                ),
            ):
                if ann_text := _get_annotation_text(ann):
                    res_parts.append(
                        create_ser_result(
                            text=f"% annotation[{ann.kind}]: {ann_text}",
                            span_source=item,
                        )
                    )
        return create_ser_result(
            text="\n".join([r.text for r in res_parts if r.text]),
            span_source=item,
        )


class LaTeXTableSerializer(BaseTableSerializer):
    """LaTeX-specific table item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a ``TableItem`` into a LaTeX ``tabular`` wrapped in ``table``."""
        params = LaTeXParams(**kwargs)
        res_parts: list[SerializationResult] = []

        def _wrap_plain_cell_text(text: str) -> str:
            # For plain strings: escape, support explicit newlines using makecell
            if not text:
                return ""
            t = _escape_latex(text) if params.escape_latex else text
            if "\n" in t:
                lines = [ln.strip() for ln in t.splitlines()]
                return r"\makecell[l]{" + r"\\".join(lines) + "}"
            return t

        def _wrap_rich_cell_content(content: str) -> str:
            # If content contains block environments, wrap in a minipage for safety
            block_markers = [
                r"\begin{itemize}",
                r"\begin{enumerate}",
                r"\begin{verbatim}",
                r"\begin{figure}",
                r"\section{",
                r"\subsection{",
                r"\subsubsection{",
                r"\begin{tabular}",
            ]
            if any(m in content for m in block_markers) or "\n\n" in content:
                return (
                    r"\begin{minipage}[t]{\linewidth}"
                    + "\n"
                    + content
                    + "\n"
                    + r"\end{minipage}"
                )
            # single-line or inline-safe rich content
            return content

        # Build table body with span-aware rendering
        table_text = ""
        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            if params.include_annotations:
                ann_res = doc_serializer.serialize_annotations(item=item, **kwargs)
                if ann_res.text:
                    res_parts.append(ann_res)

            grid = item.data.grid
            if grid:
                nrows = len(grid)
                ncols = max(len(r) for r in grid)

                # booktabs-friendly, no vertical rules
                colspec = "".join(["l" for _ in range(ncols)])
                lines: list[str] = [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule"]

                # Detect header row to add midrule (any cell with column_header)
                header_row_idx: Optional[int] = None
                if any(c.column_header for c in grid[0]):
                    header_row_idx = 0

                for i in range(nrows):
                    row_tokens: list[str] = []
                    j = 0
                    while j < ncols:
                        cell = grid[i][j]
                        # Skip over grid gaps
                        if cell is None:
                            j += 1
                            continue

                        start_row = cell.start_row_offset_idx
                        start_col = cell.start_col_offset_idx
                        rowspan = getattr(
                            cell,
                            "row_span",
                            max(1, cell.end_row_offset_idx - cell.start_row_offset_idx),
                        )
                        colspan = getattr(
                            cell,
                            "col_span",
                            max(1, cell.end_col_offset_idx - cell.start_col_offset_idx),
                        )

                        is_start = (start_row == i) and (start_col == j)

                        if is_start:
                            # Render content
                            if isinstance(cell, RichTableCell):
                                content = doc_serializer.serialize(
                                    item=cell.ref.resolve(doc=doc), **kwargs
                                ).text
                                content = _wrap_rich_cell_content(content)
                            else:
                                content = _wrap_plain_cell_text(cell.text)

                            token = content
                            if rowspan > 1 and colspan > 1:
                                token = rf"\multicolumn{{{colspan}}}{{l}}{{\multirow{{{rowspan}}}{{*}}{{{content}}}}}"
                            elif colspan > 1:
                                token = rf"\multicolumn{{{colspan}}}{{l}}{{{content}}}"
                            elif rowspan > 1:
                                token = rf"\multirow{{{rowspan}}}{{*}}{{{content}}}"

                            row_tokens.append(token)
                            j += max(1, colspan)
                        else:
                            # Covered by a span from above or left
                            # Only emit placeholders for continuation of multirow blocks starting at this column
                            # so that column count remains correct.
                            origin_row = start_row
                            origin_col = start_col
                            # Continuation of multirow coming from above at same column
                            if (origin_col == j) and (origin_row < i) and (rowspan > 1):
                                if colspan > 1:
                                    row_tokens.append(
                                        rf"\multicolumn{{{colspan}}}{{l}}{{}}"
                                    )
                                    j += colspan
                                else:
                                    row_tokens.append("")
                                    j += 1
                            else:
                                # Covered by multicolumn from left in same row, skip silently
                                j += 1

                    line = " & ".join(row_tokens) + r" \\"  # end of row
                    # Add header midrule after header row if applicable
                    if header_row_idx is not None and i == header_row_idx:
                        lines.append(line)
                        lines.append(r"\midrule")
                    else:
                        lines.append(line)

                lines.append(r"\bottomrule")
                lines.append("\\end{tabular}")
                table_text = "\n".join(lines)

        cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
        cap_text = cap_res.text

        # Wrap in table environment when we have either content or caption
        if table_text or cap_text:
            env_lines: list[str] = []
            env_lines.append("\\begin{table}[h]")
            if cap_text:
                env_lines.append(f"\\caption{{{cap_text}}}")
            if table_text:
                env_lines.append(table_text)
            env_lines.append("\\end{table}")
            res_parts.append(
                create_ser_result(text="\n".join(env_lines), span_source=item)
            )

        return create_ser_result(
            text="\n\n".join([r.text for r in res_parts if r.text]),
            span_source=res_parts,
        )


class LaTeXPictureSerializer(BasePictureSerializer):
    """LaTeX-specific picture item serializer."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a ``PictureItem`` into a LaTeX ``figure`` with optional caption and notes."""
        params = LaTeXParams(**kwargs)
        res_parts: list[SerializationResult] = []

        fig_lines: list[str] = []

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            fig_lines.append("\\begin{figure}[h]")

            # Image inclusion
            img_tex = self._serialize_image_part(
                item=item,
                doc=doc,
                image_mode=params.image_mode,
                image_placeholder=params.image_placeholder,
            ).text
            if img_tex:
                fig_lines.append(img_tex)

            # Caption
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                fig_lines.append(f"\\caption{{{cap_res.text}}}")

            # Optional annotations
            if params.include_annotations:
                ann_res = doc_serializer.serialize_annotations(item=item, **kwargs)
                if ann_res.text:
                    fig_lines.append(ann_res.text)

            fig_lines.append("\\end{figure}")
            res_parts.append(
                create_ser_result(text="\n".join(fig_lines), span_source=item)
            )

        # Optional chart data as a simple table after the figure
        if params.enable_chart_tables:
            tabular_chart_annotations = [
                ann
                for ann in item.annotations
                if isinstance(ann, PictureTabularChartData)
            ]
            if tabular_chart_annotations:
                temp_doc = DoclingDocument(name="temp")
                temp_table = temp_doc.add_table(
                    data=tabular_chart_annotations[0].chart_data
                )
                latex_table_content = (
                    LaTeXDocSerializer(doc=temp_doc).serialize(item=temp_table).text
                )
                if latex_table_content:
                    res_parts.append(
                        create_ser_result(
                            text=latex_table_content,
                            span_source=item,
                        )
                    )

        return create_ser_result(
            text="\n\n".join([r.text for r in res_parts if r.text]),
            span_source=res_parts,
        )

    def _serialize_image_part(
        self,
        *,
        item: PictureItem,
        doc: DoclingDocument,
        image_mode: ImageRefMode,
        image_placeholder: str,
    ) -> SerializationResult:
        if image_mode == ImageRefMode.PLACEHOLDER:
            return create_ser_result(text=image_placeholder, span_source=item)
        elif image_mode == ImageRefMode.REFERENCED:
            if not isinstance(item.image, ImageRef) or (
                isinstance(item.image.uri, AnyUrl) and item.image.uri.scheme == "data"
            ):
                return create_ser_result(text=image_placeholder, span_source=item)
            else:
                return create_ser_result(
                    text=f"\\includegraphics[width=\\linewidth]{{{str(item.image.uri)}}}",
                    span_source=item,
                )
        else:  # EMBEDDED not supported natively
            return create_ser_result(
                text="% embedded image not supported in LaTeX serializer",
                span_source=item,
            )


class LaTeXKeyValueSerializer(BaseKeyValueSerializer):
    """LaTeX-specific key-value item serializer."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a ``KeyValueItem``; emits a placeholder when not excluded."""
        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            return create_ser_result(text="% missing-key-value-item", span_source=item)
        else:
            return create_ser_result()


class LaTeXFormSerializer(BaseFormSerializer):
    """LaTeX-specific form item serializer."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a ``FormItem``; emits a placeholder when not excluded."""
        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            return create_ser_result(text="% missing-form-item", span_source=item)
        else:
            return create_ser_result()


class LaTeXListSerializer(BaseModel, BaseListSerializer):
    """LaTeX-specific list serializer."""

    @override
    def serialize(
        self,
        *,
        item: ListGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a list group into a nested ``itemize``/``enumerate`` environment."""
        params = LaTeXParams(**kwargs)
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            **kwargs,
        )
        env = "enumerate" if item.first_item_is_enumerated(doc) else "itemize"
        indent_str = " " * (list_level * params.indent)
        content = "\n".join([p.text for p in parts if p.text])
        text_res = (
            f"{indent_str}\\begin{{{env}}}\n{content}\n{indent_str}\\end{{{env}}}"
            if content
            else ""
        )
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
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize inline children joining them with spaces for LaTeX output."""
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level,
            is_inline_scope=True,
            **kwargs,
        )
        text_res = " ".join([p.text for p in parts if p.text])
        return create_ser_result(text=text_res, span_source=parts)


class LaTeXFallbackSerializer(BaseFallbackSerializer):
    """LaTeX-specific fallback serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize generic nodes by concatenating serialized children or a placeholder."""
        if isinstance(item, GroupItem):
            parts = doc_serializer.get_parts(item=item, **kwargs)
            text_res = "\n\n".join([p.text for p in parts if p.text])
            return create_ser_result(text=text_res, span_source=parts)
        else:
            return create_ser_result(
                text="% missing-text",
                span_source=item if isinstance(item, DocItem) else [],
            )


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

    annotation_serializer: BaseAnnotationSerializer = LaTeXAnnotationSerializer()

    params: LaTeXParams = LaTeXParams()

    @override
    def serialize_bold(self, text: str, **kwargs: Any) -> str:
        """Return LaTeX for bold text."""
        return f"\\textbf{{{text}}}"

    @override
    def serialize_italic(self, text: str, **kwargs: Any) -> str:
        """Return LaTeX for italic text."""
        return f"\\textit{{{text}}}"

    @override
    def serialize_underline(self, text: str, **kwargs: Any) -> str:
        """Return LaTeX for underlined text."""
        return f"\\underline{{{text}}}"

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any) -> str:
        """Return LaTeX for strikethrough text (requires ``ulem`` package)."""
        return f"\\sout{{{text}}}"

    @override
    def serialize_subscript(self, text: str, **kwargs: Any) -> str:
        """Return LaTeX for subscript text."""
        return f"$_{{{text}}}$"

    @override
    def serialize_superscript(self, text: str, **kwargs: Any) -> str:
        """Return LaTeX for superscript text."""
        return f"$^{{{text}}}$"

    @override
    def serialize_hyperlink(
        self,
        text: str,
        hyperlink: Union[AnyUrl, Path],
        **kwargs: Any,
    ) -> str:
        """Return LaTeX hyperlink command (requires ``hyperref`` package)."""
        return f"\\href{{{str(hyperlink)}}}{{{text}}}"

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        r"""Assemble serialized parts into a LaTeX document with environment wrapper.

        Adds optional preamble lines (document class and packages), ensures the
        output starts with "\\begin{document}" and ends with "\\end{document}".
        """
        # Merge any runtime overrides into params
        params = self.params.merge_with_patch(patch=kwargs)

        # Join body content and handle page break replacement within the body
        body_text = "\n\n".join([p.text for p in parts if p.text])
        if params.page_break_command is not None:
            for full_match, _, _ in self._get_page_breaks(text=body_text):
                body_text = body_text.replace(full_match, params.page_break_command)

        # Post-process title: move any \title{...} into the preamble
        # and add \maketitle after \begin{document}
        title_cmd, body_text, needs_maketitle = self._post_process_title(body_text)

        # Build optional preamble
        preamble_lines: list[str] = []
        if params.document_class:
            preamble_lines.append(params.document_class + "\n")
        for pkg in params.packages:
            line = pkg.strip()
            if not line:
                continue
            if line.startswith("\\"):
                preamble_lines.append(line)
            else:
                preamble_lines.append(f"\\usepackage{{{line}}}")

        # Ensure title (if any) is before \begin{document}
        if title_cmd:
            preamble_lines.append(title_cmd)

        header = (
            "\n".join(preamble_lines + ["\n\\begin{document}"])
            if preamble_lines
            else "\\begin{document}"
        )
        footer = "\\end{document}"

        # Compose final document with optional \maketitle after begin{document}
        body_parts: list[str] = []
        if needs_maketitle:
            body_parts.append("\\maketitle")
        if body_text:
            body_parts.append(body_text)
        body_block = "\n\n".join(body_parts)

        if body_block:
            full_text = f"{header}\n\n{body_block}\n\n{footer}"
        else:
            full_text = f"{header}\n\n{footer}"

        return create_ser_result(text=full_text, span_source=parts)

    @override
    def requires_page_break(self) -> bool:
        """Return True if page break replacement is enabled."""
        return self.params.page_break_command is not None

    def _post_process_title(self, body_text: str) -> tuple[Optional[str], str, bool]:
        r"""Detect and relocate LaTeX \title{...} commands.

        - Extracts the first \title{...} command found in the body.
        - Removes all \title{...} occurrences from the body.
        - Returns (title_cmd, new_body_text, needs_maketitle).

        Note: Regex assumes no nested braces inside \title{...}.
        """
        # Match \title{...} allowing whitespace, but not nested braces
        pattern = re.compile(r"\\title\s*\{([^{}]*)\}", re.DOTALL)
        first = pattern.search(body_text)
        if not first:
            # Nothing to do
            return None, body_text, False

        title_content = first.group(1)
        title_cmd = f"\\title{{{title_content}}}"
        # Remove all \title occurrences from the body
        new_body = pattern.sub("", body_text)
        # Trim excess empty lines that might remain
        new_body = re.sub(r"\n{3,}", "\n\n", new_body).strip()
        return title_cmd, new_body, True

    def post_process(
        self,
        text: str,
        *,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        **kwargs: Any,
    ) -> str:
        """Apply LaTeX escaping before formatting/hyperlinks."""
        params = self.params.merge_with_patch(patch=kwargs)
        res = text
        if params.escape_latex:
            res = _escape_latex(res)
        res = super().post_process(
            text=res,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        return res
