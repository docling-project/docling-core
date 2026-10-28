from collections import Counter
from dataclasses import dataclass

import pytest
import tiktoken
from transformers import AutoTokenizer

import docling_core.transforms.chunker.hybrid_chunker as _hybrid_mod
from docling_core.transforms.chunker.base import BaseChunker
from docling_core.transforms.chunker.doc_chunk import DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk,
    HierarchicalChunker,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.html import HTMLTableSerializer
from docling_core.transforms.serializer.markdown import (
    MARKDOWN_SEPARATOR_ROW_RE,
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin, TableItem
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.items.table.table_data import TableCell, TableData

from .test_utils import assert_or_generate_json_ground_truth, build_single_cell_rich_table_doc

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64
INPUT_FILE_0 = "tests/data/chunker/0_inp_dl_doc.json"
INPUT_FILE_2 = "tests/data/chunker/2_inp_dl_doc.json"


INNER_TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)


def warns_headers_exceed_budget():
    """Expect the warning raised when a chunk's headers and captions alone exceed `max_tokens`.

    The ground-truth tests intentionally use a tiny `MAX_TOKENS` budget to force splitting,
    which makes the headers of some chunks longer than the whole budget.
    """
    return pytest.warns(
        UserWarning,
        match="Headers and captions for this chunk are longer than the total available size",
    )


# ---------------------------------------------------------------------------
# Reusable serializer-provider classes
# ---------------------------------------------------------------------------


class CompactMarkdownSerializerProvider(ChunkingSerializerProvider):
    """Markdown serializer using compact table format."""

    def get_serializer(self, doc: DoclingDocument):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


class HTMLSerializerProvider(ChunkingSerializerProvider):
    """Serializer provider that renders tables as HTML."""

    def get_serializer(self, doc: DoclingDocument):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=HTMLTableSerializer(),
        )


def _build_wide_header_table(caption_text: str | None = None) -> tuple[DoclingDocument, TableItem]:
    doc = DoclingDocument(name="wide_table")
    doc.add_heading(text="Section heading", level=1)
    caption = doc.add_text(label=DocItemLabel.CAPTION, text=caption_text) if caption_text is not None else None
    cells = [
        TableCell(
            text=f"Column {col} heading" if row == 0 else f"Value {col}",
            row_span=1,
            col_span=1,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
            column_header=row == 0,
        )
        for row in range(2)
        for col in range(50)
    ]
    table = doc.add_table(
        data=TableData(num_rows=2, num_cols=50, table_cells=cells),
        caption=caption,
    )
    return doc, table


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dl_doc_0() -> DoclingDocument:
    """DoclingDocument loaded from INPUT_FILE_0 (the paper with large tables)."""
    with open(INPUT_FILE_0, encoding="utf-8") as f:
        return DoclingDocument.model_validate_json(f.read())


@pytest.fixture(scope="module")
def markdown_chunker_250() -> HybridChunker:
    """HybridChunker(max_tokens=250, merge_peers, repeat_table_header, compact Markdown)."""
    return HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=INNER_TOKENIZER, max_tokens=250),
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )


def test_chunk_merge_peers():
    EXPECTED_OUT_FILE = "tests/data/chunker/2a_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
    )

    with warns_headers_exceed_budget():
        chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_chunk_with_model_name():
    EXPECTED_OUT_FILE = "tests/data/chunker/2a_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer.from_pretrained(
            model_name=EMBED_MODEL_ID,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
    )

    with warns_headers_exceed_budget():
        chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_chunk_deprecated_max_tokens():
    EXPECTED_OUT_FILE = "tests/data/chunker/2a_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    with pytest.warns(DeprecationWarning, match="Deprecated initialization"):
        chunker = HybridChunker(
            max_tokens=MAX_TOKENS,
        )

    with warns_headers_exceed_budget():
        chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_contextualize():
    EXPECTED_OUT_FILE = "tests/data/chunker/2a_out_ser_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
    )

    with warns_headers_exceed_budget():
        chunks = chunker.chunk(dl_doc=dl_doc)

    act_data = dict(
        root=[
            dict(
                text=chunk.text,
                ser_text=(ser_text := chunker.contextualize(chunk)),
                num_tokens=chunker.tokenizer.count_tokens(ser_text),
            )
            for chunk in chunks
        ]
    )
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_chunk_no_merge_peers():
    EXPECTED_OUT_FILE = "tests/data/chunker/2b_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=False,
    )

    with warns_headers_exceed_budget():
        chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_chunk_deprecated_explicit_hf_obj():
    EXPECTED_OUT_FILE = "tests/data/chunker/2c_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    with pytest.warns(DeprecationWarning, match="Deprecated initialization"):
        chunker = HybridChunker(
            tokenizer=INNER_TOKENIZER,
        )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_ignore_deprecated_param_if_new_tokenizer_passed():
    EXPECTED_OUT_FILE = "tests/data/chunker/2c_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
        ),
        max_tokens=MAX_TOKENS,
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_deprecated_no_max_tokens():
    EXPECTED_OUT_FILE = "tests/data/chunker/2c_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
        ),
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_contextualize_altered_delim():
    EXPECTED_OUT_FILE = "tests/data/chunker/2d_out_ser_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
        delim="####",
    )

    with warns_headers_exceed_budget():
        chunks = chunker.chunk(dl_doc=dl_doc)

    act_data = dict(
        root=[
            dict(
                text=chunk.text,
                ser_text=(ser_text := chunker.contextualize(chunk)),
                num_tokens=chunker.tokenizer.count_tokens(ser_text),
            )
            for chunk in chunks
        ]
    )
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_chunk_multiple_content_layers(doc_with_layers, markdown_chunker_250):
    """Test that hybrid chunking can include multiple content layers in the output."""

    default_text = "\n".join(chunk.text for chunk in markdown_chunker_250.chunk(dl_doc=doc_with_layers))

    assert "Main body content" in default_text
    assert "Page Header" not in default_text
    assert "Page Footer" not in default_text

    class MultiLayerSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: DoclingDocument):
            params = MarkdownParams(layers={ContentLayer.BODY, ContentLayer.FURNITURE})
            return ChunkingDocSerializer(doc=doc, params=params)

    multilayer_chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        serializer_provider=MultiLayerSerializerProvider(),
    )
    multilayer_chunks = list(multilayer_chunker.chunk(dl_doc=doc_with_layers))
    multilayer_text = "\n".join(chunk.text for chunk in multilayer_chunks)

    assert "Main body content" in multilayer_text
    assert "Page Header" in multilayer_text
    assert "Page Footer" in multilayer_text

    act_data = dict(root=[DocChunk.model_validate(chunk).export_json_dict() for chunk in multilayer_chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        "tests/data/chunker/2i_out_chunks_multilayer.json",
    )


def test_chunk_custom_serializer():
    EXPECTED_OUT_FILE = "tests/data/chunker/2e_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    class MySerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: DoclingDocument):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
            )

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
        serializer_provider=MySerializerProvider(),
    )

    with warns_headers_exceed_budget():
        chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_chunk_openai():
    EXPECTED_OUT_FILE = "tests/data/chunker/2f_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model("gpt-4o"),
            max_tokens=128 * 1024,
        )
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_chunk_default():
    EXPECTED_OUT_FILE = "tests/data/chunker/2g_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    chunker = HybridChunker()

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_chunk_single_cell_rich_table():
    doc = build_single_cell_rich_table_doc("Important body text inside layout table")

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
    )

    chunks = list(chunker.chunk(dl_doc=doc))

    assert len(chunks) == 1
    assert chunks[0].text == "Important body text inside layout table"
    assert [item.self_ref for item in chunks[0].meta.doc_items] == [doc.tables[0].self_ref]


def test_chunk_explicit():
    EXPECTED_OUT_FILE = "tests/data/chunker/2g_out_chunks.json"

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer.from_pretrained(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=256,
        ),
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_shadowed_headings_wout_content():
    @dataclass
    class Setup:
        exp: str  # expected output file path
        chunker: BaseChunker

    setups = [
        Setup(
            exp="tests/data/chunker/2h_out_chunks_hier_emit_false.json",
            chunker=HierarchicalChunker(always_emit_headings=False),
        ),
        Setup(
            exp="tests/data/chunker/2h_out_chunks_hier_emit_true.json",
            chunker=HierarchicalChunker(always_emit_headings=True),
        ),
        Setup(
            exp="tests/data/chunker/2h_out_chunks_hybr_emit_false.json",
            chunker=HybridChunker(always_emit_headings=False),
        ),
        Setup(
            exp="tests/data/chunker/2h_out_chunks_hybr_emit_true.json",
            chunker=HybridChunker(always_emit_headings=True),
        ),
    ]

    # prepare document with different types of empty "sections" and headings shadowing each other
    doc = DoclingDocument(name="")
    doc.add_heading(text="Section 1", level=1)
    doc.add_heading(text="Section 1.1", level=2)
    doc.add_heading(text="Section 1.2", level=2)
    doc.add_heading(text="Section 2", level=1)
    doc.add_heading(text="Section 2.1", level=2)
    doc.add_heading(text="Section 2.1.1", level=3)
    doc.add_heading(text="Section 3", level=1)
    doc.add_heading(text="Section 3.1", level=2)
    doc.add_text(text="Foo", label=DocItemLabel.TEXT)
    doc.add_heading(text="Section 4", level=1)
    doc.add_heading(text="Section 4.1", level=2)

    for setup in setups:
        chunker = setup.chunker
        chunk_iter = chunker.chunk(dl_doc=doc)
        chunks = list(chunk_iter)
        act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
        assert_or_generate_json_ground_truth(
            act_data,
            setup.exp,
        )


def test_chunk_with_repeat_table_header(dl_doc_0, markdown_chunker_250):
    """Test that table headers are repeated when a table is split across chunks."""
    EXPECTED_OUT_FILE = "tests/data/chunker/0d_out_chunks.json"

    dl_doc = dl_doc_0
    chunker = markdown_chunker_250
    serializer = chunker.serializer_provider.get_serializer(dl_doc)

    # Serialize each table item individually to get expected content
    table_contents = {}
    for table_item in dl_doc.tables:
        # Serialize the table
        ser_result = serializer.serialize(
            item=table_item,
        )
        table_contents[table_item.self_ref] = ser_result.text

    chunks = list(chunker.chunk(dl_doc=dl_doc))

    # Verify we got chunks
    assert len(chunks) > 0, "Expected at least one chunk from the input document"

    # For each table, verify its content appears in chunks
    for table_ref, table_text in table_contents.items():
        # Get header and body lines from the serialized table
        if table_text:
            _, body_lines = serializer.table_serializer.get_header_and_body_lines(table_text=table_text)

            # Find all chunks that contain content from this table
            chunks_with_table = [chunk for chunk in chunks if table_ref in [i.self_ref for i in chunk.meta.doc_items]]

            # Verify table content appears in at least one chunk
            assert len(chunks_with_table) > 0, f"Table {table_ref} content should appear in at least one chunk"

            # Verify every chunk for this table has a header-like structure
            # (i.e. contains the separator row), which is produced by both
            # row-wise and column-wise splitting.
            all_chunk_text = "\n".join(chunk.text for chunk in chunks_with_table)
            if len(chunks_with_table) > 1:
                for chunk in chunks_with_table:
                    has_sep = any(MARKDOWN_SEPARATOR_ROW_RE.match(ln.rstrip("\n")) for ln in chunk.text.splitlines())
                    assert has_sep, (
                        f"Table {table_ref}: split chunk is missing the separator row "
                        f"(no header-like structure). Chunk: {chunk.text[:200]!r}"
                    )

            # Verify all cell values from all body rows appear somewhere across
            # the chunks for this table.  Row-wise splitting preserves whole rows;
            # column-wise splitting distributes cells across column-slice sub-tables.
            for body_line in body_lines:
                cells = [c.strip() for c in body_line.strip().strip("|").split("|") if c.strip()]
                for cell in cells:
                    assert cell in all_chunk_text, (
                        f"Table {table_ref}: cell value {cell!r} from row {body_line.strip()!r} not found in any chunk"
                    )

    # Save chunks to output file for inspection
    chunks_data = [
        {
            "text": chunk.text,
            "meta": {
                "doc_items": [item.self_ref for item in chunk.meta.doc_items] if chunk.meta.doc_items else [],
                "headings": chunk.meta.headings,
            },
        }
        for chunk in chunks
    ]

    assert_or_generate_json_ground_truth(chunks_data, EXPECTED_OUT_FILE)


def test_repeat_table_header_produces_valid_markdown_in_all_chunks(dl_doc_0, markdown_chunker_250):
    """Regression test: each chunk of a split table must contain only complete rows.

    When a table is split across multiple chunks with repeat_table_header=True:
    - Every table line in a chunk must start with '|' (no mid-cell fragments).
    - When the header is present, the separator row must immediately follow it
      with no blank line in between.
    - When a row is too wide to fit alongside the header, it is emitted without
      the header (graceful degradation) rather than split mid-cell.
    """
    dl_doc = dl_doc_0
    chunks = list(markdown_chunker_250.chunk(dl_doc=dl_doc))

    assert len(chunks) > 0, "Expected at least one chunk from the input document"

    sep_re = MARKDOWN_SEPARATOR_ROW_RE

    # Collect all table self_refs that appear in more than one chunk (i.e. split tables).
    table_refs = {t.self_ref for t in dl_doc.tables}
    ref_chunk_count: Counter = Counter()
    for chunk in chunks:
        seen = set()
        for item in chunk.meta.doc_items:
            if item.self_ref in table_refs and item.self_ref not in seen:
                ref_chunk_count[item.self_ref] += 1
                seen.add(item.self_ref)

    split_table_refs = {ref for ref, count in ref_chunk_count.items() if count > 1}
    assert split_table_refs, (
        "Expected at least one table to be split across chunks at max_tokens=250; "
        "check that the input document still contains a large enough table."
    )

    for i, chunk in enumerate(chunks):
        chunk_table_refs = {item.self_ref for item in chunk.meta.doc_items} & split_table_refs
        if not chunk_table_refs:
            continue  # not a table chunk — skip

        lines = chunk.text.splitlines()
        # Primary invariant: no mid-cell fragment lines.  A mid-cell fragment is
        # a line that contains '|' but does not start with '|'.  Non-pipe lines
        # (image placeholders, captions, etc.) are allowed anywhere in the chunk.
        for j, ln in enumerate(lines):
            if ln.strip() and not ln.startswith("|") and "|" in ln:
                assert False, f"Chunk {i}, line {j}: mid-cell fragment — table row was split: {ln!r}"

        # When the chunk begins with a header (has a separator row), the separator
        # must immediately follow the header with no blank line in between.
        sep_indices = [j for j, ln in enumerate(lines) if sep_re.match(ln)]
        if sep_indices:
            sep_idx = sep_indices[0]
            assert sep_idx > 0, f"Chunk {i}: separator is on the first line (no header above it)"
            assert lines[sep_idx - 1].startswith("|"), (
                f"Chunk {i}: line before separator is not a table row: {lines[sep_idx - 1]!r}"
            )
            assert lines[sep_idx - 1] != "", (
                f"Chunk {i}: blank line between header row and separator row — invalid Markdown table"
            )


def test_table_caption_not_repeated_in_split_chunks():
    """Regression test: a caption, even starting with '|', must not be treated as the
    table header row and must not appear in any chunk after the first."""
    caption_text = "| Table X: caption starting with pipe |"

    doc = DoclingDocument(name="pipe_caption")
    caption = doc.add_text(label=DocItemLabel.CAPTION, text=caption_text)
    num_cols, num_rows = 3, 10
    cells = []
    for col in range(num_cols):
        cells.append(
            TableCell(
                text=f"Col{col}",
                row_span=1,
                col_span=1,
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=col,
                end_col_offset_idx=col + 1,
                column_header=True,
            )
        )
    for row in range(1, num_rows + 1):
        for col in range(num_cols):
            cells.append(
                TableCell(
                    text=f"r{row}c{col}",
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + 1,
                    column_header=False,
                )
            )
    doc.add_table(
        data=TableData(num_rows=num_rows + 1, num_cols=num_cols, table_cells=cells),
        caption=caption,
    )

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=INNER_TOKENIZER, max_tokens=MAX_TOKENS),
        repeat_table_header=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )
    chunks = list(chunker.chunk(dl_doc=doc))

    assert len(chunks) > 1, "Table should be split into multiple chunks"

    # The caption must appear only in the first chunk.
    assert caption_text in chunks[0].text, f"Caption should appear in the first chunk: {chunks[0].text!r}"
    for chunk in chunks[1:]:
        assert caption_text not in chunk.text, (
            f"Caption starting with '|' was incorrectly repeated in a non-first chunk: {chunk.text!r}"
        )

    # Every chunk must contain the real table header row, not the caption.
    sep_re = MARKDOWN_SEPARATOR_ROW_RE
    for i, chunk in enumerate(chunks):
        lines = chunk.text.splitlines()
        sep_indices = [j for j, ln in enumerate(lines) if sep_re.match(ln)]
        assert sep_indices, f"Chunk {i} has no Markdown separator row: {chunk.text!r}"
        sep_idx = sep_indices[0]
        assert sep_idx > 0, f"Chunk {i}: separator is on the first line (no header above it)"
        assert lines[sep_idx - 1].startswith("|"), (
            f"Chunk {i}: line before separator is not a table row: {lines[sep_idx - 1]!r}"
        )


def test_table_caption_stripping_keeps_rows_without_repeated_header():
    """Row text is preserved when a captioned table is split with omit_header_on_overflow=True."""
    caption_text = "Table 1: Quarterly revenue by region"
    long_note = " ".join(["growth"] * 40)
    rows = [["Region", "Notes"], ["North", "stable"], ["South", long_note], ["East", "flat"]]

    doc = DoclingDocument(name="caption_omit_header")
    caption = doc.add_text(label=DocItemLabel.CAPTION, text=caption_text)
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=(r == 0),
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    doc.add_table(data=TableData(num_rows=len(rows), num_cols=2, table_cells=cells), caption=caption)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=INNER_TOKENIZER, max_tokens=50),
        repeat_table_header=True,
        omit_header_on_overflow=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )
    with pytest.warns(UserWarning, match="Prefix omitted"):
        chunks = list(chunker.chunk(dl_doc=doc))
    texts = [chunk.text for chunk in chunks]

    assert len(texts) > 1, "Table should be split into multiple chunks"
    # the long row fits only without the header, so it is emitted without it ...
    long_row = f"| South | {long_note} |"
    assert any(text.startswith(long_row) for text in texts), texts
    # ... and no table content is lost
    assert sum(text.count("growth") for text in texts) == 40, texts
    assert caption_text in texts[0]
    assert all(caption_text not in text for text in texts[1:]), texts


def test_chunk_html_table_serializer(dl_doc_0):
    """Test chunking with HTML table serializer."""
    EXPECTED_OUT_FILE = "tests/data/chunker/0e_out_chunks.json"

    dl_doc = dl_doc_0
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=INNER_TOKENIZER, max_tokens=500),
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=HTMLSerializerProvider(),
    )

    serializer = chunker.serializer_provider.get_serializer(dl_doc)

    # Serialize each table item individually to get expected content
    table_contents = {}
    for table_item in dl_doc.tables:
        # Serialize the table
        ser_result = serializer.serialize(
            item=table_item,
        )
        table_contents[table_item.self_ref] = ser_result.text

    chunks = list(chunker.chunk(dl_doc=dl_doc))

    # Verify we got chunks
    assert len(chunks) > 0, "Expected at least one chunk from the input document"

    # For each table, verify its content appears in chunks
    for table_ref, table_text in table_contents.items():
        # Get header and body lines from the serialized table
        if table_text:
            header_lines, body_lines = serializer.table_serializer.get_header_and_body_lines(table_text=table_text)

            # Find all chunks that contain content from this table
            chunks_with_table = [chunk for chunk in chunks if table_ref in [i.self_ref for i in chunk.meta.doc_items]]

            # Verify table content appears in at least one chunk
            assert len(chunks_with_table) > 0, f"Table {table_ref} content should appear in at least one chunk"

            # If table is split across multiple chunks, verify header is repeated
            if len(chunks_with_table) > 1:
                # Each chunk with table content should have the header
                for chunk in chunks_with_table:
                    # Check if header lines are present
                    has_header = all(header_line.strip() in chunk.text for header_line in header_lines)
                    assert has_header, (
                        f"Table {table_ref} split across chunks should have header repeated in each chunk. "
                        f"Missing header in chunk: {chunk.text[:200]}..."
                    )

            # Verify all body lines appear somewhere in the chunks
            all_chunk_text = "\n".join(chunk.text for chunk in chunks_with_table)
            for body_line in body_lines:
                assert body_line.strip() in all_chunk_text, (
                    f"Table {table_ref} body line '{body_line.strip()}' should appear in chunks"
                )

    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        EXPECTED_OUT_FILE,
    )


def test_html_parser_error_handling(dl_doc_0):
    """Test that HTML parsing errors are caught and handled gracefully."""
    dl_doc = dl_doc_0
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=INNER_TOKENIZER, max_tokens=MAX_TOKENS),
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=HTMLSerializerProvider(),
    )

    # Chunking with HTML serializer should complete without errors
    with pytest.warns(UserWarning, match="Chunks prefix is too long"):
        chunks = list(chunker.chunk(dl_doc=dl_doc))
    assert len(chunks) > 0, "Should produce chunks even with potential HTML parsing issues"

    # Verify all chunks have valid structure
    for chunk in chunks:
        assert isinstance(chunk.text, str), "Chunk text should be a string"
        assert chunk.meta is not None, "Chunk should have metadata"


@pytest.mark.parametrize(
    "tokenizer",
    [
        pytest.param(
            lambda: HuggingFaceTokenizer(tokenizer=INNER_TOKENIZER, max_tokens=MAX_TOKENS),
            id="huggingface",
        ),
        pytest.param(
            lambda: OpenAITokenizer(tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens=MAX_TOKENS),
            id="openai",
        ),
    ],
)
def test_chunk_with_tokenizer_backends(sample_doc, tokenizer):
    """HybridChunker produces chunks with both supported tokenizer backends."""
    chunker = HybridChunker(tokenizer=tokenizer())
    chunks = list(chunker.chunk(dl_doc=sample_doc))
    assert len(chunks) >= 1
    assert all(isinstance(c.text, str) for c in chunks)


def test_chunk_raises_on_missing_semchunk(monkeypatch):
    """HybridChunker raises ImportError when semchunk is not installed.

    Regression test for lazy-loading: the error must be raised at chunk time,
    not at import time, so users of chunking-openai or chunking-hf without the
    other extra can still import the module safely.
    """
    with open(INPUT_FILE_2, encoding="utf-8") as f:
        dl_doc = DoclingDocument.model_validate_json(f.read())

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=INNER_TOKENIZER, max_tokens=MAX_TOKENS),
    )

    monkeypatch.setattr(_hybrid_mod, "_SEMCHUNK_AVAILABLE", False)
    monkeypatch.setattr(_hybrid_mod, "_SEMCHUNK_IMPORT_ERROR", ImportError("semchunk not installed"))

    with pytest.raises(ImportError, match="semchunk"):
        list(chunker.chunk(dl_doc=dl_doc))


class _CountingSerializerProxy:
    """Wraps a serializer and counts serialize calls."""

    def __init__(self, inner):
        self._inner = inner
        self.serialize_calls = 0

    def serialize(self, *args, **kwargs):
        self.serialize_calls += 1
        return self._inner.serialize(*args, **kwargs)


def test_split_by_doc_items_serializes_each_item_once():
    """_split_by_doc_items serializes each item exactly once."""
    num_items = 200
    doc = DoclingDocument(name="t")
    doc.add_heading(text="Chapter 1", level=1)
    grp = doc.add_list_group(name="l")
    for i in range(num_items):
        doc.add_list_item(text=f"list item number {i} with some content", parent=grp)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=INNER_TOKENIZER, max_tokens=MAX_TOKENS),
        merge_peers=False,
    )
    ser = chunker.serializer_provider.get_serializer(doc)

    # the whole list lands in a single chunk with num_items doc items
    hier_chunk = next(iter(chunker._inner_chunker.chunk(doc)))
    assert len(hier_chunk.meta.doc_items) == num_items

    proxy = _CountingSerializerProxy(ser)
    chunker._split_by_doc_items(hier_chunk, proxy)

    assert proxy.serialize_calls == num_items


class _TestTokenizer(BaseTokenizer):
    """Provide the token budget and backend for deterministic test tokenizers."""

    max_tokens: int

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self):
        return self


class _MergingNewlineTokenizer(_TestTokenizer):
    """Deterministic tokenizer whose per-item counts under-estimate joined text.

    Counting is word-based, but each newline followed by a list-marker dash
    merges into a single token. Summing per-item counts therefore
    under-estimates the joined window, forcing the shrink-back path in
    _split_by_doc_items.
    """

    def count_tokens(self, text: str) -> int:
        return len(text.split(" ")) + text.count("\n-")


def _assert_chunk_windows(chunker, chunks, expected_items, min_window_items=1):
    """Check ordered item conservation, window growth, and exact token limits."""
    assert [it for c in chunks for it in c.meta.doc_items] == expected_items
    assert max(len(c.meta.doc_items) for c in chunks) > min_window_items
    assert all(chunker._count_chunk_tokens(c) <= chunker.max_tokens for c in chunks)


def test_split_by_doc_items_shrinks_back_when_estimate_under_counts():
    """Windows stay within max_tokens even when the running estimate under-counts."""
    doc = DoclingDocument(name="t")
    grp = doc.add_list_group(name="l")
    for i in range(12):
        doc.add_list_item(text=f"w{i}", parent=grp)

    chunker = HybridChunker(
        tokenizer=_MergingNewlineTokenizer(max_tokens=10),
        merge_peers=False,
    )
    ser = chunker.serializer_provider.get_serializer(doc)
    hier_chunk = next(iter(chunker._inner_chunker.chunk(doc)))
    chunks = chunker._split_by_doc_items(hier_chunk, ser)

    _assert_chunk_windows(chunker, chunks, hier_chunk.meta.doc_items, min_window_items=2)


def test_doc_meta_excluded_embed_keeps_contextualize_window_invariant():
    """Split and merge share this contract: doc_items and origin add no context."""
    assert "doc_items" in DocMeta.excluded_embed
    assert "origin" in DocMeta.excluded_embed

    doc = _paragraph_doc(2)
    chunker = HybridChunker(tokenizer=_WordCountTokenizer(max_tokens=100))
    reference = DocChunk(
        text="same text",
        meta=DocMeta(doc_items=[doc.texts[1]], headings=["Chapter 1"]),
    )
    window = DocChunk(
        text=reference.text,
        meta=DocMeta(
            doc_items=doc.texts[1:],
            headings=reference.meta.headings,
            origin=DocumentOrigin(mimetype="text/plain", binary_hash=0, filename="other.txt"),
        ),
    )
    assert chunker.contextualize(window) == chunker.contextualize(reference)


class _WordCountTokenizer(_TestTokenizer):
    """Deterministic word-counting tokenizer with additive per-chunk counts."""

    calls: int = 0
    chars: int = 0

    def count_tokens(self, text: str) -> int:
        self.calls += 1
        self.chars += len(text)
        return len(text.split())


class _UnderCountingTokenizer(_TestTokenizer):
    """Word-counting tokenizer whose per-chunk counts under-estimate joined text.

    A chunk that starts with "paragraph" counts one token less than its words,
    while the joined window only loses that token once. Summing per-chunk counts
    therefore under-estimates the joined window, forcing the shrink-back path in
    _merge_chunks_with_matching_metadata.
    """

    def count_tokens(self, text: str) -> int:
        return len(text.split()) - (1 if text.startswith("paragraph") else 0)


def _paragraph_doc(num_paragraphs: int) -> DoclingDocument:
    doc = DoclingDocument(name="t")
    doc.add_heading(text="Chapter 1", level=1)
    for i in range(num_paragraphs):
        doc.add_text(label=DocItemLabel.PARAGRAPH, text=f"paragraph number {i} explains the results in some detail.")
    return doc


def _chunks_before_merge(chunker, doc):
    ser = chunker.serializer_provider.get_serializer(doc)
    res = [x for c in chunker._inner_chunker.chunk(doc) for x in chunker._split_by_doc_items(c, doc_serializer=ser)]
    return [x for c in res for x in chunker._split_using_plain_text(c, doc_serializer=ser)]


@pytest.mark.parametrize("max_tokens", [100, 500, 5000])
def test_merge_chunks_tokenizes_linearly(max_tokens):
    """_merge_chunks_with_matching_metadata must not re-tokenize the growing window per step.

    The old implementation re-tokenized the whole joined window on every
    iteration, i.e. ~N·w/2 chunk-sized counts for N chunks in windows of w.
    With additive word counts, estimates are exact: each output window needs
    one metadata count and one final verification, plus an overflow check for
    each window except the last. No re-anchoring or shrink-back is needed.
    """
    doc = _paragraph_doc(200)
    tok = _WordCountTokenizer(max_tokens=max_tokens)
    chunker = HybridChunker(tokenizer=tok, merge_peers=False)
    chunks = _chunks_before_merge(chunker, doc)

    tok.calls = 0
    tok.chars = 0
    merged = chunker._merge_chunks_with_matching_metadata(chunks)

    # Measure only the merge stage, before correctness assertions count tokens.
    merge_calls, merge_chars = tok.calls, tok.chars
    overflow_events = len(merged) - 1  # all inputs have the same headings
    calibration_calls = 3  # counts of a+delimiter+b, a, and b
    assert merge_calls <= len(chunks) + calibration_calls + 2 * len(merged) + overflow_events

    # Call counts alone cannot catch the old implementation: it also makes O(N)
    # calls, but each call re-counts a growing prefix. Bound the actual input
    # characters as well. Each input text is counted once; final verification
    # counts each output once. Overflow checks count each output at most once
    # more, plus a delimiter and the next input text. Metadata checks count one
    # contextualized input per output. A maximum single-input context bounds
    # both metadata checks and the extra input at each overflow.
    text_chars = sum(len(c.text) for c in chunks)
    output_chars = sum(len(chunker.contextualize(c)) for c in merged)
    max_input_chars = max(len(chunker.contextualize(c)) for c in chunks)
    calibration_chars = len(f"a{chunker.delim}b") + len("a") + len("b")
    assert merge_chars <= (
        calibration_chars
        + text_chars
        + 2 * output_chars
        + (len(merged) + overflow_events) * max_input_chars
        + overflow_events * len(chunker.delim)
    )

    _assert_chunk_windows(chunker, merged, [it for c in chunks for it in c.meta.doc_items])


class _OverCountingTokenizer(_TestTokenizer):
    """Word-counting tokenizer whose per-chunk counts over-estimate joined text.

    A chunk that starts with "paragraph" counts one token more than its words,
    while the joined window only gains that token once. Summing per-chunk counts
    therefore over-estimates the joined window, forcing the re-anchor path in
    _merge_chunks_with_matching_metadata.
    """

    def count_tokens(self, text: str) -> int:
        return len(text.split()) + (1 if text.startswith("paragraph") else 0)


@pytest.mark.parametrize(
    "tokenizer_cls",
    [
        pytest.param(_UnderCountingTokenizer, id="under-count-shrink"),
        pytest.param(_OverCountingTokenizer, id="over-count-reanchor"),
    ],
)
def test_merge_chunks_corrects_token_estimate(tokenizer_cls):
    """Exact checks correct both directions of error in the running estimate."""
    doc = _paragraph_doc(30)
    chunker = HybridChunker(tokenizer=tokenizer_cls(max_tokens=100), merge_peers=False)
    chunks = _chunks_before_merge(chunker, doc)

    merged = chunker._merge_chunks_with_matching_metadata(chunks)

    _assert_chunk_windows(chunker, merged, [it for c in chunks for it in c.meta.doc_items])


def test_contextualized_markdown_table_chunks_respect_token_limit():
    max_tokens = 64
    doc, _ = _build_wide_header_table()
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model("text-embedding-3-small"),
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
        repeat_table_header=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )

    chunks = list(chunker.chunk(dl_doc=doc))

    assert len(chunks) > 1
    assert all(tokenizer.count_tokens(chunker.contextualize(chunk)) <= max_tokens for chunk in chunks)


def test_contextualized_markdown_table_chunks_recheck_final_count():
    max_tokens = 70
    doc, _ = _build_wide_header_table(caption_text="\nA")
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
        repeat_table_header=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )

    chunks = list(chunker.chunk(dl_doc=doc))
    counts = [tokenizer.count_tokens(chunker.contextualize(chunk)) for chunk in chunks]

    assert max(counts) <= max_tokens


@pytest.mark.parametrize(
    ("max_tokens", "cell_prefix", "heading_text"),
    [
        (96, "x", None),
        (128, "a-b/", "Section heading"),
    ],
)
def test_split_markdown_table_chunks_respect_exact_token_boundaries(
    max_tokens: int,
    cell_prefix: str,
    heading_text: str | None,
):
    num_cols = 12
    num_rows = 3
    doc = DoclingDocument(name="overflow")
    if heading_text is not None:
        doc.add_heading(text=heading_text, level=1)
    cells = [
        TableCell(
            text=f"{cell_prefix}{row}_{col}",
            row_span=1,
            col_span=1,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
            column_header=row == 0,
        )
        for row in range(num_rows)
        for col in range(num_cols)
    ]
    doc.add_table(
        data=TableData(
            num_rows=num_rows,
            num_cols=num_cols,
            table_cells=cells,
        )
    )
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model("text-embedding-3-small"),
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
        repeat_table_header=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )

    chunks = list(chunker.chunk(dl_doc=doc))
    counts = [tokenizer.count_tokens(chunker.contextualize(chunk)) for chunk in chunks]

    assert max(counts) <= max_tokens


def test_split_markdown_table_preserves_all_cell_values():
    """Splitting a wide table must preserve every cell value across all chunks.

    When a table is too wide for its rows to fit alongside the repeated header,
    ``HybridChunker`` splits it by columns (``omit_header_on_overflow=False``,
    the default).  Each column-slice chunk is a complete, valid Markdown sub-table.
    The union of all chunk texts must contain every cell value from the original.
    """
    doc, table = _build_wide_header_table(caption_text="Wide table caption")
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model("text-embedding-3-small"),
        max_tokens=64,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
        repeat_table_header=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )
    serializer = chunker.serializer_provider.get_serializer(doc)
    full_table_text = serializer.serialize(item=table).text

    chunks = [
        chunk
        for chunk in chunker.chunk(dl_doc=doc)
        if any(item.self_ref == table.self_ref for item in chunk.meta.doc_items)
    ]

    assert len(chunks) > 1

    # Every chunk must be a structurally valid Markdown sub-table (has separator row).
    for i, chunk in enumerate(chunks):
        has_sep = any(MARKDOWN_SEPARATOR_ROW_RE.match(ln.rstrip("\n")) for ln in chunk.text.splitlines())
        assert has_sep, f"Chunk {i} has no Markdown separator row: {chunk.text[:200]!r}"

    # No chunk may contain a mid-cell fragment.
    all_chunk_text = "\n".join(chunk.text for chunk in chunks)
    for i, chunk in enumerate(chunks):
        for j, ln in enumerate(chunk.text.splitlines()):
            if ln.strip() and not ln.startswith("|") and "|" in ln:
                raise AssertionError(f"Chunk {i}, line {j}: mid-cell fragment — row was split: {ln!r}")

    # All cell values from the original table appear somewhere across all chunks.
    _, body_lines = serializer.table_serializer.get_header_and_body_lines(table_text=full_table_text)
    for body_line in body_lines:
        cells = [c.strip() for c in body_line.strip().strip("|").split("|") if c.strip()]
        for cell in cells:
            assert cell in all_chunk_text, f"Cell value {cell!r} from row {body_line.strip()!r} not found in any chunk"


def test_wide_table_is_split_by_columns_when_rows_do_not_fit_with_header():
    """Regression: a table whose rows are too wide to fit with the header must be
    split by columns (not mid-cell) when ``omit_header_on_overflow=False``.

    Each resulting column-slice chunk must be a complete, valid Markdown sub-table
    with its own header, and no chunk may exceed ``max_tokens``.

    Scenario: header ≈ 137 tokens, body row ≈ 106 tokens, header+row ≈ 243 tokens.
    With a 9-token heading context, available_length = max_tokens(250) - 9 = 241.
    So header+row(243) > available_length(241) but header+row ≤ max_tokens(250).
    """
    max_tokens = 250
    # 15 columns with 12-char cell text gives header≈137 tokens, body row≈106 tokens,
    # and header+row≈243 tokens — just over available_length=241 but under max_tokens=250.
    num_cols = 15
    cell_text = "aaaaaaaaaaaa"  # 12 chars

    # A heading that produces ~9 tokens of context overhead.
    doc = DoclingDocument(name="ctx_overhead")
    doc.add_heading(text="Baselines for Object Detection", level=1)

    cells = [
        TableCell(
            text=cell_text if row == 0 else "vvvvvv",
            row_span=1,
            col_span=1,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
            column_header=row == 0,
        )
        for row in range(4)
        for col in range(num_cols)
    ]
    doc.add_table(data=TableData(num_rows=4, num_cols=num_cols, table_cells=cells))

    tokenizer = HuggingFaceTokenizer(
        tokenizer=INNER_TOKENIZER,
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
        repeat_table_header=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )

    chunks = list(chunker.chunk(dl_doc=doc))

    # The table must be split into more than one chunk.
    assert len(chunks) > 1, "Wide table should be split into multiple chunks"

    # No chunk may exceed max_tokens.
    for i, chunk in enumerate(chunks):
        tc = tokenizer.count_tokens(chunker.contextualize(chunk))
        assert tc <= max_tokens, f"Chunk {i} exceeds token limit: {tc} > {max_tokens}"

    # Every chunk must be a structurally valid Markdown sub-table (has separator row).
    for i, chunk in enumerate(chunks):
        has_sep = any(MARKDOWN_SEPARATOR_ROW_RE.match(ln.rstrip("\n")) for ln in chunk.text.splitlines())
        assert has_sep, f"Chunk {i} has no Markdown separator row: {chunk.text[:200]!r}"

    # No chunk may contain a mid-cell fragment (line with '|' not starting with '|').
    for i, chunk in enumerate(chunks):
        for j, ln in enumerate(chunk.text.splitlines()):
            if ln.strip() and not ln.startswith("|") and "|" in ln:
                raise AssertionError(f"Chunk {i}, line {j}: mid-cell fragment — row was split mid-cell: {ln!r}")

    # All cell values from the original table appear somewhere across all chunks.
    all_text = "\n".join(chunk.text for chunk in chunks)
    expected_values = {cell_text, "vvvvvv"}
    for val in expected_values:
        assert val in all_text, f"Cell value {val!r} missing from all chunks"


def test_table_under_heading_chunks_respect_token_limit():
    """HybridChunker must not produce chunks exceeding max_tokens.

    A 3-column table with 30 data rows placed under a heading causes
    _split_using_plain_text to compute available_length = max_tokens - heading_overhead.
    The segment() table path must honour available_length, not max_tokens, so that
    contextualize() does not push the final chunk over the limit.
    """
    max_tokens = 64

    doc = DoclingDocument(name="table_under_heading")
    doc.add_heading(text="Quarterly revenue by region and product line", level=1)

    num_rows = 31  # 1 header + 30 data rows
    num_cols = 3
    headers = ["Region", "Product", "Revenue"]
    cells = []
    for col in range(num_cols):
        cells.append(
            TableCell(
                text=headers[col],
                row_span=1,
                col_span=1,
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=col,
                end_col_offset_idx=col + 1,
                column_header=True,
            )
        )
    for row in range(1, num_rows):
        cells.append(
            TableCell(
                text=f"North{row}",
                row_span=1,
                col_span=1,
                start_row_offset_idx=row,
                end_row_offset_idx=row + 1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
                column_header=False,
            )
        )
        cells.append(
            TableCell(
                text=f"Widget{row}",
                row_span=1,
                col_span=1,
                start_row_offset_idx=row,
                end_row_offset_idx=row + 1,
                start_col_offset_idx=1,
                end_col_offset_idx=2,
                column_header=False,
            )
        )
        cells.append(
            TableCell(
                text=f"${row * 1000}",
                row_span=1,
                col_span=1,
                start_row_offset_idx=row,
                end_row_offset_idx=row + 1,
                start_col_offset_idx=2,
                end_col_offset_idx=3,
                column_header=False,
            )
        )
    doc.add_table(data=TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells))

    tokenizer = HuggingFaceTokenizer(
        tokenizer=INNER_TOKENIZER,
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
        repeat_table_header=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )

    chunks = list(chunker.chunk(dl_doc=doc))

    assert len(chunks) > 1, "Table should be split into multiple chunks"

    for i, chunk in enumerate(chunks):
        tc = tokenizer.count_tokens(chunker.contextualize(chunk))
        assert tc <= max_tokens, (
            f"Chunk {i} exceeds token limit: {tc} > {max_tokens} — "
            f"contextualized text: {chunker.contextualize(chunk)!r}"
        )


def test_split_markdown_table_with_escaped_pipes_and_empty_cells():
    """Verify that wide tables containing escaped pipes and empty cells are properly split across columns."""
    num_cols = 12
    num_rows = 3
    doc = DoclingDocument(name="escaped_pipes_table")
    cells = []
    for col in range(num_cols):
        # Column header
        header_text = f"Header \\| {col}" if col % 2 == 0 else f"Header {col}"
        cells.append(
            TableCell(
                text=header_text,
                row_span=1,
                col_span=1,
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=col,
                end_col_offset_idx=col + 1,
                column_header=True,
            )
        )
    for row in range(1, num_rows):
        for col in range(num_cols):
            # Alternate between text with escaped pipe, empty text, and normal text
            if col % 3 == 0:
                cell_text = f"Val \\| {row}_{col}"
            elif col % 3 == 1:
                cell_text = ""
            else:
                cell_text = f"Val_{row}_{col}"
            cells.append(
                TableCell(
                    text=cell_text,
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + 1,
                    column_header=False,
                )
            )
    doc.add_table(data=TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells))

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=64,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
        repeat_table_header=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )

    chunks = list(chunker.chunk(dl_doc=doc))

    assert len(chunks) > 1, "Wide table should be split into multiple chunks"
    for i, chunk in enumerate(chunks):
        tc = tokenizer.count_tokens(chunker.contextualize(chunk))
        assert tc <= 64, f"Chunk {i} exceeds token limit: {tc} > 64"
        # Verify valid Markdown table structure
        assert any(MARKDOWN_SEPARATOR_ROW_RE.match(ln.rstrip("\n")) for ln in chunk.text.splitlines())


def test_tall_and_wide_table_chunks_respect_token_limit():
    """Verify that a table with both many columns and many rows is partitioned across columns and rows."""
    num_cols = 12
    num_rows = 25  # 1 header + 24 body rows
    max_tokens = 96

    doc = DoclingDocument(name="tall_and_wide_table")
    doc.add_heading(text="Large Financial Matrix Overview", level=1)

    cells = []
    for col in range(num_cols):
        cells.append(
            TableCell(
                text=f"Metric_{col}",
                row_span=1,
                col_span=1,
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=col,
                end_col_offset_idx=col + 1,
                column_header=True,
            )
        )
    for row in range(1, num_rows):
        for col in range(num_cols):
            cells.append(
                TableCell(
                    text=f"R{row}C{col}_val",
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + 1,
                    column_header=False,
                )
            )
    doc.add_table(data=TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells))

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
        repeat_table_header=True,
        serializer_provider=CompactMarkdownSerializerProvider(),
    )

    chunks = list(chunker.chunk(dl_doc=doc))

    assert len(chunks) > 2, "Tall and wide table should be split into multiple chunks across columns and rows"
    all_chunk_text = "\n".join(chunk.text for chunk in chunks)

    for i, chunk in enumerate(chunks):
        tc = tokenizer.count_tokens(chunker.contextualize(chunk))
        assert tc <= max_tokens, (
            f"Chunk {i} exceeds token limit: {tc} > {max_tokens} — "
            f"contextualized text: {chunker.contextualize(chunk)!r}"
        )
        # Verify valid Markdown table structure (has separator row)
        assert any(MARKDOWN_SEPARATOR_ROW_RE.match(ln.rstrip("\n")) for ln in chunk.text.splitlines())

    # Verify that cells from top and bottom rows, left and right columns are preserved
    sample_values = [
        "Metric_0",
        f"Metric_{num_cols - 1}",
        "R1C0_val",
        f"R1C{num_cols - 1}_val",
        f"R{num_rows - 1}C0_val",
        f"R{num_rows - 1}C{num_cols - 1}_val",
    ]
    for val in sample_values:
        assert val in all_chunk_text, f"Sample cell value {val!r} not found across chunks"


def test_split_using_plain_text_no_headings_avoids_infinite_recursion():
    """Verify that _split_using_plain_text does not recurse infinitely when headings and captions are absent."""
    max_tokens = 32
    doc = DoclingDocument(name="no_headings_doc")
    # A single large paragraph without any headings or captions
    long_text = "This is a long paragraph designed to exceed the chunk token budget. " * 10
    doc.add_text(label=DocItemLabel.PARAGRAPH, text=long_text)

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
    )

    chunks = list(chunker.chunk(dl_doc=doc))

    assert len(chunks) > 1, "Large paragraph should be split into multiple chunks"
    for i, chunk in enumerate(chunks):
        tc = tokenizer.count_tokens(chunker.contextualize(chunk))
        assert tc <= max_tokens, f"Chunk {i} exceeds token limit: {tc} > {max_tokens}"
        assert chunk.meta.headings is None
