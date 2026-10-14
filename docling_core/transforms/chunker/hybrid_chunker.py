"""Hybrid chunker implementation leveraging both doc structure & token awareness."""

import warnings
from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import Any, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocChunk,
    DocMeta,
    HierarchicalChunker,
)
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MARKDOWN_SEPARATOR_ROW_RE
from docling_core.types.doc import DoclingDocument, SectionHeaderItem, TableItem, TitleItem

_SEMCHUNK_AVAILABLE: bool = False
_SEMCHUNK_IMPORT_ERROR: ImportError | None = None
try:
    import semchunk

    _SEMCHUNK_AVAILABLE = True
except ImportError as e:
    _SEMCHUNK_IMPORT_ERROR = e

_SEMCHUNK_INSTALL_HINT = (
    "The 'semchunk' package is required by HybridChunker. "
    "Install it with `pip install 'docling-core[chunking]'` or "
    "`pip install 'docling-core[chunking-openai]'`."
)


def _get_default_tokenizer():
    """Get default tokenizer instance.

    Defer import to avoid pulling in transformers at module import time.
    """
    from docling_core.transforms.chunker.tokenizer.huggingface import get_default_tokenizer

    return get_default_tokenizer()


class HybridChunker(BaseChunker):
    r"""Chunker doing tokenization-aware refinements on top of document layout chunking.

    Args:
        tokenizer: The tokenizer to use; either instantiated object or name or path of
            respective pretrained model
        max_tokens: The maximum number of tokens per chunk. If not set, limit is
            resolved from the tokenizer
        repeat_table_header: Whether to repeat a table header if the table is chunked
        merge_peers: Whether to merge undersized chunks sharing same relevant metadata
        always_emit_headings: Whether to emit headings even for empty sections
        omit_header_on_overflow: Only used when repeat_table_header is True. When True,
            omit table headers for rows that would overflow with them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: BaseTokenizer = Field(default_factory=_get_default_tokenizer)
    repeat_table_header: bool = True
    merge_peers: bool = True
    omit_header_on_overflow: bool = False

    serializer_provider: BaseSerializerProvider = ChunkingSerializerProvider()
    always_emit_headings: bool = False

    @model_validator(mode="before")
    @classmethod
    def _patch(cls, data: Any) -> Any:
        if isinstance(data, dict):
            tokenizer = data.get("tokenizer")
            max_tokens = data.get("max_tokens")
            if not isinstance(tokenizer, BaseTokenizer) and (
                # some legacy param passed:
                tokenizer is not None or max_tokens is not None
            ):
                from docling_core.transforms.chunker.tokenizer.huggingface import (
                    HuggingFaceTokenizer,
                )

                warnings.warn(
                    "Deprecated initialization parameter types for HybridChunker. "
                    "For updated usage check out "
                    "https://docling-project.github.io/docling/examples/hybrid_chunking/",
                    DeprecationWarning,
                )

                if isinstance(tokenizer, str):
                    data["tokenizer"] = HuggingFaceTokenizer.from_pretrained(
                        model_name=tokenizer,
                        max_tokens=max_tokens,
                    )
                elif tokenizer is None or (
                    hasattr(tokenizer, "__module__")
                    and (tokenizer.__module__.startswith("transformers.") or tokenizer.__module__ == "transformers")
                ):
                    from docling_core.transforms.chunker.tokenizer.huggingface import (
                        HuggingFaceTokenizer,
                    )

                    kwargs = {"tokenizer": tokenizer or _get_default_tokenizer().tokenizer}
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    data["tokenizer"] = HuggingFaceTokenizer(**kwargs)
        return data

    @property
    def max_tokens(self) -> int:
        """Get maximum number of tokens allowed."""
        return self.tokenizer.get_max_tokens()

    @computed_field  # type: ignore[misc]
    @cached_property
    def _inner_chunker(self) -> HierarchicalChunker:
        return HierarchicalChunker(
            serializer_provider=self.serializer_provider,
            always_emit_headings=self.always_emit_headings,
        )

    def _count_text_tokens(self, text: str | list[str] | None):
        if text is None:
            return 0
        elif isinstance(text, list):
            total = 0
            for t in text:
                total += self._count_text_tokens(t)
            return total
        return self.tokenizer.count_tokens(text=text)

    class _ChunkLengthInfo(BaseModel):
        total_len: int
        text_len: int
        other_len: int

    def _count_chunk_tokens(self, doc_chunk: DocChunk):
        ser_txt = self.contextualize(chunk=doc_chunk)
        return self.tokenizer.count_tokens(text=ser_txt)

    def _doc_chunk_length(self, doc_chunk: DocChunk):
        text_length = self._count_text_tokens(doc_chunk.text)
        total = self._count_chunk_tokens(doc_chunk=doc_chunk)
        return self._ChunkLengthInfo(
            total_len=total,
            text_len=text_length,
            other_len=total - text_length,
        )

    def _estimate_delimiter_tokens(self) -> int:
        """Calibrate the token contribution of joining two texts."""
        return (
            self.tokenizer.count_tokens(text=f"a{self.delim}b")
            - self.tokenizer.count_tokens(text="a")
            - self.tokenizer.count_tokens(text="b")
        )

    def _estimate_meta_tokens(self, doc_chunk: DocChunk, text_tokens: int) -> int:
        """Estimate the token overhead that contextualize() adds to the chunk text.

        Callers that replace only text and doc_items may reuse this estimate
        while headings and captions stay unchanged. Tokenization at the context
        boundary can change with the text, so exact window counts are still needed.

        Args:
            doc_chunk: The reference chunk used to measure overhead.
            text_tokens: Pre-computed token count of ``doc_chunk.text``.

        Returns:
            Estimated token overhead: contextualized tokens minus text tokens.
            The difference is not guaranteed to be non-negative.
        """
        return self._count_chunk_tokens(doc_chunk=doc_chunk) - text_tokens

    def _split_by_doc_items(self, doc_chunk: DocChunk, doc_serializer: BaseDocSerializer) -> list[DocChunk]:
        """Split a chunk along doc-item boundaries, keeping each window within max_tokens.

        Each item is serialized once and its token count cached. Windows are
        grown with a running token estimate, re-anchored by exact counts at
        overflow points, and shrunk back to the largest fitting window when the
        estimate under-counts. The resulting windows match the previous
        implementation exactly, since the largest window that fits is unique.

        Args:
            doc_chunk: The chunk to split, containing one or more doc items.
            doc_serializer: Serializer for the document, used to obtain per-item text.

        Returns:
            A list of chunks whose windows respect max_tokens. A single item
            that does not fit is returned as-is and split later at the
            plain-text stage.
        """
        doc_items = doc_chunk.meta.doc_items
        num_items = len(doc_items)
        if num_items <= 1:
            meta = DocMeta(
                doc_items=doc_items,
                headings=doc_chunk.meta.headings,
                origin=doc_chunk.meta.origin,
            )
            return [DocChunk(text=doc_chunk.text, meta=meta)]

        item_texts: list[str] = []
        item_token_counts: list[int] = []
        for doc_item in doc_items:
            res_text = doc_serializer.serialize(item=doc_item).text
            text = res_text if (res_text and not isinstance(doc_item, TitleItem | SectionHeaderItem)) else ""
            item_texts.append(text)
            item_token_counts.append(self.tokenizer.count_tokens(text=text))

        # token cost of the delimiter between two joined items
        delim_tokens = self._estimate_delimiter_tokens()

        def build_window(window_start: int, window_end: int) -> DocChunk:
            meta = DocMeta(
                doc_items=doc_items[window_start : window_end + 1],
                headings=doc_chunk.meta.headings,
                origin=doc_chunk.meta.origin,
            )
            window_text = self.delim.join([t for t in item_texts[window_start : window_end + 1] if t])
            return DocChunk(text=window_text, meta=meta)

        def window_count(window_start: int, window_end: int) -> int:
            return self._count_chunk_tokens(doc_chunk=build_window(window_start, window_end))

        # constant across windows: doc_items and origin are excluded from the embed
        # serialization (DocMeta.excluded_embed), so contextualize() only sees
        # headings/captions, which are the same for every window in this chunk
        meta_overhead = self._estimate_meta_tokens(build_window(0, 0), item_token_counts[0])

        chunks: list[DocChunk] = []
        window_start = 0
        while window_start < num_items:
            window_end = window_start
            running = 0
            while window_end < num_items:
                running += item_token_counts[window_end] + (delim_tokens if window_end > window_start else 0)
                if running + meta_overhead > self.max_tokens:
                    exact = window_count(window_start, window_end)
                    if exact <= self.max_tokens:
                        # the estimate overshot; re-anchor it on the exact count
                        running = exact - meta_overhead
                        window_end += 1
                        continue
                    break
                window_end += 1

            if window_end == window_start:
                # a single item doesn't fit: emit it as-is and let the
                # plain-text splitter deal with it later
                chunks.append(build_window(window_start, window_start))
                window_start += 1
                continue

            end = window_end
            while end - 1 > window_start and window_count(window_start, end - 1) > self.max_tokens:
                end -= 1
            chunks.append(build_window(window_start, end - 1))
            window_start = end

        return chunks

    def _split_using_plain_text(
        self,
        doc_chunk: DocChunk,
        doc_serializer: BaseDocSerializer,
    ) -> list[DocChunk]:
        """Split an oversized chunk into token-bounded sub-chunks via text segmentation.

        Computes the token budget available for body text by subtracting the cost of
        headings and captions (``other_len``) from ``max_tokens``, then delegates
        segmentation to :meth:`segment`. If the first segmentation still produces
        chunks that exceed ``max_tokens`` after contextualization (due to non-additive
        tokenization), ``available_length`` is reduced by the observed overflow and
        :meth:`segment` is called again. This loop converges because ``available_length``
        strictly decreases on each iteration.

        When the loop exhausts the budget (``available_length <= 0``), headings and
        captions are stripped and the method recurses once on the stripped chunk, which
        has ``other_len == 0`` and therefore a full ``max_tokens`` budget. If headings
        and captions are already absent, :meth:`segment` is called directly with the
        full budget.

        Args:
            doc_chunk: The chunk to split. May already fit within ``max_tokens``, in
                which case it is returned unchanged.
            doc_serializer: Serializer for the current document, forwarded to
                :meth:`segment`.

        Returns:
            A list of :class:`DocChunk` objects whose contextualized token counts
            respect ``max_tokens``.
        """
        lengths = self._doc_chunk_length(doc_chunk)
        if lengths.total_len <= self.max_tokens:
            return [DocChunk(**doc_chunk.export_json_dict())]

        available_length = self.max_tokens - lengths.other_len

        while available_length > 0:
            segments = self.segment(doc_chunk, available_length, doc_serializer)
            chunks = [DocChunk(text=s, meta=doc_chunk.meta) for s in segments]
            overflow = max(self._count_chunk_tokens(chunk) - self.max_tokens for chunk in chunks)
            if overflow <= 0:
                return chunks
            available_length -= overflow

        if doc_chunk.meta.headings is not None or doc_chunk.meta.captions is not None:
            warnings.warn(
                "Headers and captions for this chunk are longer than the total "
                "available size for the chunk, so they will be ignored: "
                f"{doc_chunk.text=}, {doc_chunk.meta=}"
            )
            new_chunk = DocChunk(**doc_chunk.export_json_dict())
            new_chunk.meta.captions = None
            new_chunk.meta.headings = None
            return self._split_using_plain_text(doc_chunk=new_chunk, doc_serializer=doc_serializer)

        # Headings and captions are already absent; return segmented chunks directly.
        segments = self.segment(doc_chunk, self.max_tokens, doc_serializer)
        return [DocChunk(text=s, meta=doc_chunk.meta) for s in segments]

    def segment(self, doc_chunk: DocChunk, available_length: int, doc_serializer: BaseDocSerializer) -> list[str]:
        """Split a single doc chunk into a list of text segments.

        Two strategies are applied depending on the chunk's content:

        - Table with header repetition: when ``repeat_table_header`` is
          enabled and the chunk contains exactly one ``TableItem``, the table is
          split line-by-line via ``LineBasedTokenChunker``.  The header rows are
          used as a prefix so they are reproduced at the top of every segment.
          Any preamble text that precedes the first table row (e.g. a caption) is
          included in the prefix to account for its token cost, but is stripped
          from all segments after the first so it does not repeat.

          When ``omit_header_on_overflow`` is ``False`` and a body row is too
          wide to fit alongside the full header, the table is partitioned vertically
          (by columns) and horizontally (by rows for tall tables) so that each
          resulting sub-table is a complete, valid Markdown table with matching headers.
          When ``omit_header_on_overflow`` is ``True``, wide rows are emitted without
          the header instead.

        - Semantic splitting: for all other chunks, ``semchunk`` splits the
          text at natural semantic boundaries up to ``available_length`` tokens.

        Args:
            doc_chunk: The chunk to segment.
            available_length: Maximum token budget available for each segment.
            doc_serializer: Serializer for the current document; must be a
                ``ChunkingDocSerializer`` for the table path to activate.

        Returns:
            A list of text segments derived from ``doc_chunk.text``.
        """
        segments = []
        if (
            self.repeat_table_header
            and isinstance(doc_serializer, ChunkingDocSerializer)
            and len(doc_chunk.meta.doc_items) == 1
            and isinstance(doc_chunk.meta.doc_items[0], TableItem)
        ):
            header_lines, body_lines = doc_serializer.table_serializer.get_header_and_body_lines(
                table_text=doc_chunk.text
            )

            if header_lines:
                header_start = doc_chunk.text.find(header_lines[0])
                preamble = doc_chunk.text[:header_start] if header_start > 0 else ""
            else:
                preamble = ""
            table_prefix = "".join(header_lines)
            full_prefix = preamble + table_prefix

            # Check whether any body row is too wide to fit alongside the full header.
            # Uses the concatenated token count (not the sum) to match the same
            # non-additive tokenization behaviour used by LineBasedTokenChunker.
            any_row_overflows = body_lines and any(
                self.tokenizer.count_tokens(table_prefix + line) > available_length for line in body_lines
            )

            if any_row_overflows and not self.omit_header_on_overflow:
                # omit_header_on_overflow=False: the user wants the header present
                # in every chunk.  Since splitting a Markdown row mid-cell produces
                # an invalid fragment, split the table vertically (by columns) so
                # that each sub-table is a complete, valid Markdown table.
                segments = self._split_table_by_columns(
                    header_lines=header_lines,
                    body_lines=body_lines,
                    preamble=preamble,
                    available_length=available_length,
                )
            else:
                # fallback for rows that don't fit.
                line_chunker = LineBasedTokenChunker(
                    tokenizer=self.tokenizer,
                    max_tokens_override=available_length,
                    prefix=full_prefix,
                    omit_prefix_on_overflow=self.omit_header_on_overflow,
                    serializer_provider=self.serializer_provider,
                )
                segments = line_chunker.chunk_text(lines=body_lines)
                if preamble:
                    segments = segments[:1] + [
                        s[len(preamble) :] if s.startswith(full_prefix) else s for s in segments[1:]
                    ]
        else:
            if not _SEMCHUNK_AVAILABLE:
                raise ImportError(_SEMCHUNK_INSTALL_HINT) from _SEMCHUNK_IMPORT_ERROR
            sem_chunker = semchunk.chunkerify(self.tokenizer.get_tokenizer(), chunk_size=available_length)
            sem_segments = sem_chunker(doc_chunk.text)
            segments = cast(list[str], sem_segments)
        return segments

    def _merge_chunks_with_matching_metadata(self, chunks: list[DocChunk]) -> list[DocChunk]:
        """Merge neighboring chunks with matching headings, keeping windows within max_tokens.

        Chunk texts are counted once and windows are grown with a running token
        estimate, re-anchored by exact counts at overflow points and shrunk back
        to the largest fitting window when the estimate under-counts. The merged
        output matches the previous implementation exactly, since the largest
        window that fits is unique.

        Args:
            chunks: The chunks to merge, in document order.

        Returns:
            The merged list of chunks.
        """
        output_chunks: list[DocChunk] = []
        num_chunks = len(chunks)

        # token cost of the delimiter between two joined chunk texts
        delim_tokens = self._estimate_delimiter_tokens()
        chunk_token_counts = [self.tokenizer.count_tokens(text=chk.text) for chk in chunks]

        def build_candidate(window_start: int, window_end: int) -> DocChunk:
            """Build a merged chunk from the window (window_end is inclusive)."""
            doc_items = [it for chk in chunks[window_start : window_end + 1] for it in chk.meta.doc_items]
            return DocChunk(
                # TODO: merging should ideally be done by the serializer:
                text=self.delim.join([chk.text for chk in chunks[window_start : window_end + 1]]),
                meta=DocMeta(
                    doc_items=doc_items,
                    headings=chunks[window_start].meta.headings,
                    origin=chunks[window_end].meta.origin,
                ),
            )

        window_start = 0
        while window_start < num_chunks:
            first_chunk = chunks[window_start]
            headings = first_chunk.meta.headings

            # constant token cost of the merged meta (headings, origin) around
            # the joined text; all candidates in this window share it
            meta_overhead = self._estimate_meta_tokens(first_chunk, chunk_token_counts[window_start])

            window_end = window_start + 1
            fit_end = window_start
            running = chunk_token_counts[window_start]
            while window_end < num_chunks and chunks[window_end].meta.headings == headings:
                running += chunk_token_counts[window_end] + delim_tokens
                if running + meta_overhead <= self.max_tokens:
                    fit_end = window_end
                    window_end += 1
                    continue
                candidate = build_candidate(window_start, window_end)
                exact = self._count_chunk_tokens(doc_chunk=candidate)
                if exact <= self.max_tokens:
                    # the estimate overshot; re-anchor it on the exact count
                    running = exact - meta_overhead
                    fit_end = window_end
                    window_end += 1
                    continue
                break  # the window overflowed

            # exact-verify the largest accepted window, in case the estimate
            # under-counted (and shrink back if it did)
            while (
                fit_end > window_start
                and self._count_chunk_tokens(doc_chunk=build_candidate(window_start, fit_end)) > self.max_tokens
            ):
                fit_end -= 1

            if fit_end == window_start:
                output_chunks.append(first_chunk)
            else:
                output_chunks.append(build_candidate(window_start, fit_end))
            window_start = fit_end + 1

        return output_chunks

    def _split_table_by_columns(
        self,
        header_lines: list[str],
        body_lines: list[str],
        preamble: str,
        available_length: int,
    ) -> list[str]:
        """Split a Markdown table into column-wise sub-tables that fit within the token budget.

        When a table row is too wide to fit alongside the repeated header, splitting
        the row mid-cell would produce an invalid Markdown fragment. This method
        instead partitions the table columns into groups using a two-tier strategy:

        1. Column-Wise Fit (All Rows): Binary searches for the largest column slice
           ``[col_start, mid)`` where all body rows fit together with the sub-table
           header within ``available_length`` tokens. If such a slice exists, it is
           emitted as a complete, multi-row sub-table chunk.
        2. 2D Slicing Fallback (Tall Tables): If even a single-column slice with all
           body rows exceeds the budget, binary searches for the largest column
           slice where each individual body row fits alongside the sub-table
           header, and delegates row-wise chunking of that column slice to
           ``LineBasedTokenChunker``.

        The preamble (caption or other text preceding the header) is prepended only
        to the first sub-table segment.

        Args:
            header_lines: The header row(s) and separator row of the original table,
                each as a newline-terminated string.
            body_lines: The data rows of the original table, each as a
                newline-terminated string.
            preamble: Text preceding the table header (e.g. a caption). Included
                in the first returned segment only.
            available_length: Maximum token budget for each segment (text only,
                before context overhead is added by the caller).

        Returns:
            A list of text segments, each a complete Markdown sub-table string
            with matching column headers and separator rows.
        """
        # Exclude the separator row from header_lines for cell parsing; rebuild it.
        data_header_lines = [l for l in header_lines if not MARKDOWN_SEPARATOR_ROW_RE.match(l.rstrip("\n"))]

        def parse_cells(row: str) -> list[str]:
            """Split a Markdown table row string into a list of cell strings.

            Strips the optional leading and trailing ``|`` delimiters, then splits
            on unescaped ``|`` characters. A ``|`` is considered escaped when it is
            preceded by an odd number of backslashes. Each cell value is stripped of
            surrounding whitespace before being returned.

            Args:
                row: A single Markdown table row, e.g. ``| a | b\\|c | d |``.

            Returns:
                A list of cell strings with whitespace stripped. Returns an empty
                list for blank rows.
            """
            line = row.strip()
            if not line:
                return []
            if line.startswith("|"):
                line = line[1:]
            if line.endswith("|"):
                num_bs = len(line[:-1]) - len(line[:-1].rstrip("\\"))
                if num_bs % 2 == 0:
                    line = line[:-1]

            cells: list[str] = []
            current: list[str] = []
            bs_count = 0
            for char in line:
                if char == "\\":
                    bs_count += 1
                    current.append(char)
                elif char == "|":
                    if bs_count % 2 == 1:
                        current.append(char)
                    else:
                        cells.append("".join(current).strip())
                        current = []
                    bs_count = 0
                else:
                    bs_count = 0
                    current.append(char)
            cells.append("".join(current).strip())
            return cells

        header_cell_rows: list[list[str]] = [parse_cells(l) for l in data_header_lines]
        body_cell_rows: list[list[str]] = [parse_cells(l) for l in body_lines if l.strip()]

        # Determine the effective column count (min across all rows for safety).
        n_cols = min(
            (len(r) for r in header_cell_rows + body_cell_rows),
            default=0,
        )
        if n_cols == 0:
            # Malformed table: fall back to emitting body rows without header.
            return ["".join(body_lines)]

        def build_subtable(col_start: int, col_end: int, include_preamble: bool) -> str:
            """Construct the text of a sub-table covering columns [col_start, col_end) with all body rows."""
            lines: list[str] = []
            if include_preamble and preamble:
                lines.append(preamble)
            for hrow in header_cell_rows:
                lines.append("| " + " | ".join(hrow[col_start:col_end]) + " |\n")
            lines.append("| " + " | ".join(["-"] * (col_end - col_start)) + " |\n")
            for brow in body_cell_rows:
                lines.append("| " + " | ".join(brow[col_start:col_end]) + " |\n")
            return "".join(lines)

        def build_header(c_start: int, c_end: int, include_preamble: bool) -> str:
            """Construct the header text (and optional preamble) for columns [c_start, c_end)."""
            lines: list[str] = []
            if include_preamble and preamble:
                lines.append(preamble)
            for hrow in header_cell_rows:
                lines.append("| " + " | ".join(hrow[c_start:c_end]) + " |\n")
            lines.append("| " + " | ".join(["-"] * (c_end - c_start)) + " |\n")
            return "".join(lines)

        def row_fits_with_header(c_start: int, c_end: int, include_preamble: bool) -> bool:
            """Return True if all individual body rows can fit alongside the sub-table header."""
            hdr = build_header(c_start, c_end, include_preamble=include_preamble)
            if self.tokenizer.count_tokens(hdr) >= available_length:
                return False
            for brow in body_cell_rows:
                row_line = "| " + " | ".join(brow[c_start:c_end]) + " |\n"
                if self.tokenizer.count_tokens(hdr + row_line) > available_length:
                    return False
            return True

        segments: list[str] = []
        col_start = 0
        first = True

        while col_start < n_cols:
            # First, try to find the maximum columns that fit ALL body rows in a single chunk.
            lo, hi = col_start + 1, n_cols
            best_all_rows_end: int | None = None
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = build_subtable(col_start, mid, include_preamble=first)
                if self.tokenizer.count_tokens(candidate) <= available_length:
                    best_all_rows_end = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            if best_all_rows_end is not None:
                # All body rows fit within the token budget for columns [col_start, best_all_rows_end).
                segments.append(build_subtable(col_start, best_all_rows_end, include_preamble=first))
                col_start = best_all_rows_end
            else:
                # Table is tall: even a single column with all body rows exceeds available_length.
                # Find the maximum columns where individual rows fit alongside the sub-header,
                # then chunk rows using LineBasedTokenChunker.
                lo, hi = col_start + 1, n_cols
                best_row_end: int | None = None
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if row_fits_with_header(col_start, mid, include_preamble=first):
                        best_row_end = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1

                if best_row_end is None:
                    best_row_end = col_start + 1

                sub_header = build_header(col_start, best_row_end, include_preamble=first)
                sub_body_lines = ["| " + " | ".join(brow[col_start:best_row_end]) + " |\n" for brow in body_cell_rows]

                line_chunker = LineBasedTokenChunker(
                    tokenizer=self.tokenizer,
                    max_tokens_override=available_length,
                    prefix=sub_header,
                    omit_prefix_on_overflow=self.omit_header_on_overflow,
                    serializer_provider=self.serializer_provider,
                )
                slice_segments = line_chunker.chunk_text(lines=sub_body_lines)
                if preamble and first:
                    slice_segments = slice_segments[:1] + [
                        s[len(preamble) :] if s.startswith(sub_header) else s for s in slice_segments[1:]
                    ]
                segments.extend(slice_segments)
                col_start = best_row_end

            first = False

        return segments

    def chunk(
        self,
        dl_doc: DoclingDocument,
        **kwargs: Any,
    ) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        my_doc_ser = self.serializer_provider.get_serializer(doc=dl_doc)
        res: Iterable[DocChunk]
        res = self._inner_chunker.chunk(
            dl_doc=dl_doc,
            **kwargs,
        )  # type: ignore
        res = [x for c in res for x in self._split_by_doc_items(c, doc_serializer=my_doc_ser)]
        res = [x for c in res for x in self._split_using_plain_text(c, doc_serializer=my_doc_ser)]
        if self.merge_peers:
            res = self._merge_chunks_with_matching_metadata(res)
        return iter(res)
