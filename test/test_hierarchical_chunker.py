from pathlib import Path

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk,
    TripletTableSerializer,
)
from docling_core.transforms.serializer.html import HTMLDocSerializer
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc import DocItemLabel, DoclingDocument, PictureItem, TableData, TextItem
from docling_core.types.doc.document import TableCell

from .test_utils import assert_or_generate_json_ground_truth


def test_chunk():
    with open("test/data/chunker/0_inp_dl_doc.json", encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(
        merge_list_items=True,
    )
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(act_data, "test/data/chunker/0_out_chunks.json")


def test_chunk_custom_serializer():
    with open("test/data/chunker/0_inp_dl_doc.json", encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    class MySerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: DoclingDocument):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
            )

    chunker = HierarchicalChunker(
        merge_list_items=True,
        serializer_provider=MySerializerProvider(),
    )

    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(act_data, "test/data/chunker/0b_out_chunks.json")



def test_traverse_pictures():
    """Test that traverse_pictures parameter works correctly with HierarchicalChunker."""

    # Load a document that has TextItem children of PictureItem
    INPUT_FILE = "test/data/doc/concatenated.json"
    dl_doc = DoclingDocument.load_from_json(Path(INPUT_FILE))

    # Verify the document has non-caption TextItem children of PictureItem
    has_non_caption_text_in_picture = False
    for item, _ in dl_doc.iterate_items(with_groups=True, traverse_pictures=True):
        if isinstance(item, TextItem) and item.parent:
            parent = item.parent.resolve(dl_doc)
            if isinstance(parent, PictureItem) and item.label != DocItemLabel.CAPTION:
                has_non_caption_text_in_picture = True
                break

    assert has_non_caption_text_in_picture, (
        "Test document should have non-caption TextItem children of PictureItem"
    )

    # Test 1: Default behavior (traverse_pictures=False)
    # Only caption TextItems should be included
    chunker_default = HierarchicalChunker()
    chunks_default = list(chunker_default.chunk(dl_doc=dl_doc))

    caption_count_default = 0
    non_caption_count_default = 0
    for chunk in chunks_default:
        for doc_item in chunk.meta.doc_items:
            if isinstance(doc_item, TextItem) and doc_item.parent:
                parent = doc_item.parent.resolve(dl_doc)
                if isinstance(parent, PictureItem):
                    if doc_item.label == DocItemLabel.CAPTION:
                        caption_count_default += 1
                    else:
                        non_caption_count_default += 1

    # Test 2: With traverse_pictures=True
    # Both caption and non-caption TextItems should be included
    class TraversePicturesSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            params = MarkdownParams(traverse_pictures=True)
            return ChunkingDocSerializer(doc=doc, params=params)

    chunker_traverse = HierarchicalChunker(
        serializer_provider=TraversePicturesSerializerProvider(),
    )
    chunks_traverse = list(chunker_traverse.chunk(dl_doc=dl_doc))

    caption_count_traverse = 0
    non_caption_count_traverse = 0
    for chunk in chunks_traverse:
        for doc_item in chunk.meta.doc_items:
            if isinstance(doc_item, TextItem) and doc_item.parent:
                parent = doc_item.parent.resolve(dl_doc)
                if isinstance(parent, PictureItem):
                    if doc_item.label == DocItemLabel.CAPTION:
                        caption_count_traverse += 1
                    else:
                        non_caption_count_traverse += 1

    assert non_caption_count_default == 0, (
        f"With traverse_pictures=False (default), non-caption TextItems that are "
        f"children of PictureItems should NOT be included. Found {non_caption_count_default}"
    )

    assert caption_count_default > 0, (
        "Caption TextItems should be included even with traverse_pictures=False"
    )

    assert non_caption_count_traverse > 0, (
        f"With traverse_pictures=True, non-caption TextItems that are children of "
        f"PictureItems should be included. Found {non_caption_count_traverse}"
    )

    assert caption_count_traverse >= caption_count_default, (
        f"Caption count should not decrease with traverse_pictures=True. "
        f"Got {caption_count_traverse} vs {caption_count_default}"
    )

    total_items_default = sum(len(chunk.meta.doc_items) for chunk in chunks_default)
    total_items_traverse = sum(len(chunk.meta.doc_items) for chunk in chunks_traverse)
    assert total_items_traverse > total_items_default, (
        f"With traverse_pictures=True, more doc_items should be included in chunks. "
        f"Got {total_items_traverse} vs {total_items_default}"
    )


def test_triplet_table_serializer_single_column():
    """Test TripletTableSerializer with a single-column table."""

    doc = DoclingDocument(name="test_single_column")
    table_data = TableData(num_rows=4, num_cols=1)
    table_item = doc.add_table(data=table_data)
    doc.add_table_cell(
        table_item=table_item,
        cell=TableCell(
            text="Country",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            column_header=True,
        ),
    )
    for i, name in enumerate(["Italy", "Canada", "Switzerland"], start=1):
        doc.add_table_cell(
            table_item=table_item,
            cell=TableCell(
                text=name,
                start_row_offset_idx=i,
                end_row_offset_idx=i + 1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
            ),
        )

    serializer = ChunkingDocSerializer(doc=doc)
    table_serializer = TripletTableSerializer()

    result = table_serializer.serialize(
        item=table_item,
        doc_serializer=serializer,
        doc=doc,
    )

    expected = "Country = Italy. Country = Canada. Country = Switzerland"
    assert result.text == expected, f"Expected '{expected}', got '{result.text}'"


def _make_cell(
    text, row, col, *, column_header=False, row_header=False, row_span=1, col_span=1
):
    """Helper to build a TableCell with less boilerplate."""
    return TableCell(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + row_span,
        start_col_offset_idx=col,
        end_col_offset_idx=col + col_span,
        column_header=column_header,
        row_header=row_header,
        row_span=row_span,
        col_span=col_span,
    )


def test_triplet_table_serializer_both_headers():
    """Column headers + row headers → row_hdr, col_hdr = value."""

    #        | Q1  | Q2
    # -------+-----+-----
    #  Rev   | 100 | 200
    #  Cost  |  50 |  80
    doc = DoclingDocument(name="test_both_headers")
    table_data = TableData(num_rows=3, num_cols=3)
    table_item = doc.add_table(data=table_data)

    for j, text in enumerate(["", "Q1", "Q2"]):
        doc.add_table_cell(
            table_item=table_item,
            cell=_make_cell(text, 0, j, column_header=True),
        )
    for i, (rh, vals) in enumerate(
        [("Rev", ["100", "200"]), ("Cost", ["50", "80"])], start=1
    ):
        doc.add_table_cell(
            table_item=table_item,
            cell=_make_cell(rh, i, 0, row_header=True),
        )
        for j, val in enumerate(vals, start=1):
            doc.add_table_cell(
                table_item=table_item,
                cell=_make_cell(val, i, j),
            )

    ser = ChunkingDocSerializer(doc=doc)
    result = TripletTableSerializer().serialize(
        item=table_item, doc_serializer=ser, doc=doc
    )
    expected = "Rev, Q1 = 100. Rev, Q2 = 200. Cost, Q1 = 50. Cost, Q2 = 80"
    assert result.text == expected, f"Expected '{expected}', got '{result.text}'"


def test_triplet_table_serializer_row_headers_only():
    """Row headers but no column headers → row_hdr, Column j = value."""

    #  Rev  | 100 | 200
    #  Cost |  50 |  80
    doc = DoclingDocument(name="test_row_headers_only")
    table_data = TableData(num_rows=2, num_cols=3)
    table_item = doc.add_table(data=table_data)

    for i, (rh, vals) in enumerate(
        [("Rev", ["100", "200"]), ("Cost", ["50", "80"])]
    ):
        doc.add_table_cell(
            table_item=table_item,
            cell=_make_cell(rh, i, 0, row_header=True),
        )
        for j, val in enumerate(vals, start=1):
            doc.add_table_cell(
                table_item=table_item,
                cell=_make_cell(val, i, j),
            )

    ser = ChunkingDocSerializer(doc=doc)
    result = TripletTableSerializer().serialize(
        item=table_item, doc_serializer=ser, doc=doc
    )
    expected = (
        "Rev, Column 1 = 100. Rev, Column 2 = 200. "
        "Cost, Column 1 = 50. Cost, Column 2 = 80"
    )
    assert result.text == expected, f"Expected '{expected}', got '{result.text}'"


def test_triplet_table_serializer_no_headers():
    """No headers at all → Row i, Column j = value."""

    doc = DoclingDocument(name="test_no_headers")
    table_data = TableData(num_rows=2, num_cols=2)
    table_item = doc.add_table(data=table_data)

    for i, row_vals in enumerate([["a", "b"], ["c", "d"]]):
        for j, val in enumerate(row_vals):
            doc.add_table_cell(
                table_item=table_item,
                cell=_make_cell(val, i, j),
            )

    ser = ChunkingDocSerializer(doc=doc)
    result = TripletTableSerializer().serialize(
        item=table_item, doc_serializer=ser, doc=doc
    )
    expected = (
        "Row 0, Column 0 = a. Row 0, Column 1 = b. "
        "Row 1, Column 0 = c. Row 1, Column 1 = d"
    )
    assert result.text == expected, f"Expected '{expected}', got '{result.text}'"


def test_triplet_table_serializer_single_column_no_header():
    """Single-column, no headers → Row i, Column 0 = value."""

    doc = DoclingDocument(name="test_single_col_no_header")
    table_data = TableData(num_rows=3, num_cols=1)
    table_item = doc.add_table(data=table_data)

    for i, val in enumerate(["x", "y", "z"]):
        doc.add_table_cell(
            table_item=table_item,
            cell=_make_cell(val, i, 0),
        )

    ser = ChunkingDocSerializer(doc=doc)
    result = TripletTableSerializer().serialize(
        item=table_item, doc_serializer=ser, doc=doc
    )
    expected = "Row 0, Column 0 = x. Row 1, Column 0 = y. Row 2, Column 0 = z"
    assert result.text == expected, f"Expected '{expected}', got '{result.text}'"


def test_triplet_table_serializer_merged_cell():
    """A row-spanning cell is repeated in the DataFrame, producing one triplet per row."""

    # | Year | Revenue |
    # | 2024 |     100 |   ← "2024" spans rows 1-2
    # |      |     200 |
    doc = DoclingDocument(name="test_merged")
    table_data = TableData(num_rows=3, num_cols=2)
    table_item = doc.add_table(data=table_data)

    doc.add_table_cell(
        table_item=table_item,
        cell=_make_cell("Year", 0, 0, column_header=True),
    )
    doc.add_table_cell(
        table_item=table_item,
        cell=_make_cell("Revenue", 0, 1, column_header=True),
    )
    doc.add_table_cell(
        table_item=table_item,
        cell=_make_cell("2024", 1, 0, row_span=2),
    )
    doc.add_table_cell(
        table_item=table_item,
        cell=_make_cell("100", 1, 1),
    )
    doc.add_table_cell(
        table_item=table_item,
        cell=_make_cell("200", 2, 1),
    )

    ser = ChunkingDocSerializer(doc=doc)
    result = TripletTableSerializer().serialize(
        item=table_item, doc_serializer=ser, doc=doc
    )
    expected = "2024, Revenue = 100. 2024, Revenue = 200"
    assert result.text == expected, f"Expected '{expected}', got '{result.text}'"


def test_triplet_table_serializer_nested_table():
    """A cell containing a nested table embeds the inner table's triplet text."""
    from docling_core.types.doc.document import RichTableCell

    doc = DoclingDocument(name="test_nested")

    outer = doc.add_table(data=TableData(num_rows=1, num_cols=2))

    inner = doc.add_table(data=TableData(num_rows=1, num_cols=2), parent=outer)
    doc.add_table_cell(
        table_item=inner, cell=_make_cell("x", 0, 0),
    )
    doc.add_table_cell(
        table_item=inner, cell=_make_cell("y", 0, 1),
    )

    doc.add_table_cell(
        table_item=outer, cell=_make_cell("plain", 0, 0),
    )
    doc.add_table_cell(
        table_item=outer,
        cell=RichTableCell(
            text="",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            ref=inner.get_ref(),
        ),
    )

    ser = ChunkingDocSerializer(doc=doc)
    result = TripletTableSerializer().serialize(
        item=outer, doc_serializer=ser, doc=doc
    )

    # The outer table has no headers → positional notation.
    # Cell (0,1) contains the inner table, which is also headerless.
    expected = (
        "Row 0, Column 0 = plain. "
        "Row 0, Column 1 = Row 0, Column 0 = x. Row 0, Column 1 = y"
    )
    assert result.text == expected, f"Expected '{expected}', got '{result.text}'"


def test_chunk_rich_table_custom_serializer(rich_table_doc: DoclingDocument):
    doc = rich_table_doc

    class MySerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: DoclingDocument):
            return HTMLDocSerializer(
                doc=doc,
                table_serializer=TripletTableSerializer(),
            )

    chunker = HierarchicalChunker(
        merge_list_items=True,
        serializer_provider=MySerializerProvider(),
    )

    chunks = chunker.chunk(dl_doc=doc)
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )

    assert_or_generate_json_ground_truth(act_data, "test/data/chunker/0c_out_chunks.json")
