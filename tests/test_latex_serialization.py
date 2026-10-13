"""Test LaTeX serialization.

These tests leverage DOCLING_GEN_TEST_DATA via GEN_TEST_DATA to optionally
generate or update the golden files. Additionally, if a golden file does not
exist, the test will create it to bootstrap the baseline.
"""

import warnings
from pathlib import Path

import pytest
import yaml

from docling_core.transforms.serializer.latex import LaTeXDocSerializer, LaTeXParams
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    BaseMeta,
    CodeMetaField,
    DescriptionAnnotation,
    DescriptionMetaField,
    DocItemLabel,
    DoclingDocument,
    EntitiesMetaField,
    EntityMention,
    FloatingMeta,
    Formatting,
    KeywordsMetaField,
    LanguageMetaField,
    MoleculeMetaField,
    PictureClassificationClass,
    PictureClassificationData,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    PictureTabularChartData,
    SummaryMetaField,
    TableData,
    TabularChartMetaField,
    TopicsMetaField,
)
from docling_core.types.doc.items.table.table_data import TableCell

from .test_data_gen_flag import GEN_TEST_DATA


def verify_or_update(exp_file: Path, actual: str):
    exp_file.parent.mkdir(parents=True, exist_ok=True)
    # If GEN_TEST_DATA is enabled or the expected file is missing, write/update it
    if GEN_TEST_DATA or not exp_file.exists():
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(f"{actual}\n")
    else:
        with open(exp_file, encoding="utf-8") as f:
            expected = f.read().rstrip()
        assert expected == actual


def test_latex_basic_activities():
    src = Path("./tests/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            # Do not add page-break replacements by default
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_latex_inline_and_formatting():
    src = Path("./tests/data/doc/inline_and_formatting.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_dummy_doc():
    src = Path("tests/data/doc/dummy_doc.yaml")

    # Read YAML file of manual reference doc
    with open(src, encoding="utf-8") as fp:
        dict_from_yaml = yaml.safe_load(fp)

    doc = DoclingDocument.model_validate(dict_from_yaml)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_constructed_doc(sample_doc: DoclingDocument):
    doc = sample_doc

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    src = Path("tests/data/doc/construct_doc.yaml")
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_constructed_rich_table_doc(rich_table_doc: DoclingDocument):
    doc = rich_table_doc

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    src = Path("tests/data/doc/construct_rich_table_doc.yaml")
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_latex_paper():
    src = Path("./tests/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_latex_nested_lists():
    src = Path("./tests/data/doc/polymers.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_inline_group_no_duplication():
    r"""Regression test: inline-group content must not appear twice.

    When a SectionHeaderItem, TitleItem or ListItem delegates its text to a
    child InlineGroup (the shape produced by Markdown/DOCX/HTML import for any
    formatted heading or list item), the serialized text must appear exactly
    once — inside the LaTeX command (\section{}, \item, …) and *not* again
    as a standalone paragraph immediately after it.
    """
    doc = DoclingDocument(name="t")

    # Formatted section heading via InlineGroup
    heading = doc.add_heading(text="", level=1)
    ig = doc.add_inline_group(parent=heading)
    doc.add_text(
        label="text",
        text="Partially formatted",
        formatting=Formatting(italic=True),
        parent=ig,
    )
    doc.add_text(label="text", text="heading", parent=ig)

    # Formatted list item via InlineGroup
    lg = doc.add_list_group()
    li = doc.add_list_item(text="", parent=lg)
    ig2 = doc.add_inline_group(parent=li)
    doc.add_text(label="text", text="Term", formatting=Formatting(bold=True), parent=ig2)
    doc.add_text(label="text", text=": definition", parent=ig2)

    body = LaTeXDocSerializer(doc=doc).serialize().text.split("\\begin{document}")[1]

    # Each piece of content must appear exactly once in the body
    assert body.count("\\textit{Partially formatted} heading") == 1
    assert body.count("\\textbf{Term} : definition") == 1

    # The content must be inside the LaTeX command, not on a standalone line
    assert "\\section{\\textit{Partially formatted} heading}" in body
    assert "\\item \\textbf{Term} : definition" in body


def test_latex_column_spanning_rich_cell_keeps_text_in_every_column(col_span_rich_table_doc):
    actual = LaTeXDocSerializer(doc=col_span_rich_table_doc).serialize().text
    assert " & Partner & Partner \\\\ \\hline\nStatus & Done & Open \\\\ \\hline" in actual


_DEPRECATED_ANNOTATIONS = r"Field `annotations` is deprecated"


def _chart_data() -> TableData:
    """Build a 2x2 table to be used as chart data."""
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
        )
        for r, row in enumerate([["Header 1", "Header 2"], ["Val 1", "Val 2"]])
        for c, text in enumerate(row)
    ]
    return TableData(num_rows=2, num_cols=2, table_cells=cells)


def _doc_with_legacy_annotations() -> DoclingDocument:
    """Build a document with a picture and a table that only have legacy annotations."""
    doc = DoclingDocument(name="test")
    pic = doc.add_picture()
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1))
    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        pic.annotations.append(
            PictureClassificationData(
                provenance="test",
                predicted_classes=[PictureClassificationClass(class_name="line_chart", confidence=0.9)],
            )
        )
    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        pic.annotations.append(DescriptionAnnotation(provenance="test", text="A legacy line chart"))
    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        pic.annotations.append(PictureTabularChartData(chart_data=_chart_data(), title="Legacy chart"))
    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        table.annotations.append(DescriptionAnnotation(provenance="test", text="A legacy table description"))
    return doc


def _doc_with_meta() -> DoclingDocument:
    """Build a document with a picture and a table that have `meta` and no annotations."""
    doc = DoclingDocument(name="test")
    pic = doc.add_picture()
    pic.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[PictureClassificationPrediction(class_name="bar_chart", confidence=0.9)]
        ),
        description=DescriptionMetaField(text="A sample bar chart"),
        molecule=MoleculeMetaField(smi="CC1=NNC=N1"),
        tabular_chart=TabularChartMetaField(chart_data=_chart_data(), title="Chart table"),
    )
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1))
    table.meta = FloatingMeta(description=DescriptionMetaField(text="A table description"))
    return doc


def test_latex_picture_meta_path():
    """Picture meta is serialized inside the figure in a stable order, followed by the chart table."""
    doc = DoclingDocument(name="test")
    pic = doc.add_picture()
    pic.meta = PictureMeta(
        molecule=MoleculeMetaField(smi="CC1=NNC=N1"),
        description=DescriptionMetaField(text="A sample bar chart"),
        classification=PictureClassificationMetaField(
            predictions=[PictureClassificationPrediction(class_name="bar_chart", confidence=0.9)]
        ),
    )

    actual = LaTeXDocSerializer(doc=doc).serialize().text
    assert (
        "\\begin{figure}[h]\n"
        "% image\n"
        "% annotation[classification]: bar chart\n"
        "% annotation[description]: A sample bar chart\n"
        "% annotation[molecule_data]: CC1=NNC=N1\n"
        "\\end{figure}"
    ) in actual


def test_latex_table_meta_path():
    """Table meta is serialized before the table environment."""
    doc = DoclingDocument(name="test")
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1))
    table.meta = FloatingMeta(description=DescriptionMetaField(text="Table metadata description"))

    actual = LaTeXDocSerializer(doc=doc).serialize().text
    assert actual.index("% annotation[description]: Table metadata description") < actual.index("\\begin{table}[h]")


def test_latex_tabular_chart_meta_path():
    """The tabular chart in picture meta is rendered as a table, not as a comment."""
    doc = DoclingDocument(name="test")
    pic = doc.add_picture()
    pic.meta = PictureMeta(tabular_chart=TabularChartMetaField(chart_data=_chart_data(), title="Chart table"))

    actual = LaTeXDocSerializer(doc=doc).serialize().text
    assert "\\end{figure}\n\n\\begin{table}[h]" in actual
    assert "Header 1 & Header 2 \\\\ \\hline\nVal 1 & Val 2 \\\\ \\hline" in actual
    assert "annotation[tabular_chart]" not in actual

    actual = LaTeXDocSerializer(doc=doc, params=LaTeXParams(enable_chart_tables=False)).serialize().text
    assert "Header 1" not in actual


def test_latex_meta_path_raises_no_deprecation_warning():
    """Serializing items that have meta must not touch the deprecated `annotations` field."""
    doc = _doc_with_meta()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        actual = LaTeXDocSerializer(doc=doc).serialize().text

    assert "% annotation[description]: A sample bar chart" in actual
    assert "% annotation[description]: A table description" in actual
    assert "Header 1 & Header 2" in actual


def test_latex_picture_legacy_annotations_path():
    """Legacy picture annotations are still serialized, with the `annotations` deprecation warning."""
    doc = _doc_with_legacy_annotations()

    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        actual = LaTeXDocSerializer(doc=doc).serialize().text

    assert (
        "\\begin{figure}[h]\n"
        "% image\n"
        "% annotation[classification]: line chart\n"
        "% annotation[description]: A legacy line chart\n"
        "\\end{figure}"
    ) in actual


def test_latex_table_legacy_annotations_path():
    """Legacy table annotations are still serialized, with the `annotations` deprecation warning."""
    doc = DoclingDocument(name="test")
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1))
    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        table.annotations.append(DescriptionAnnotation(provenance="test", text="A legacy table description"))

    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        actual = LaTeXDocSerializer(doc=doc).serialize().text

    assert "% annotation[description]: A legacy table description" in actual
    assert actual.index("% annotation[description]: A legacy table description") < actual.index("\\begin{table}[h]")


def test_latex_tabular_chart_legacy_annotations_path():
    """Legacy `PictureTabularChartData` is still rendered as a table when the picture has no meta."""
    doc = _doc_with_legacy_annotations()

    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        actual = LaTeXDocSerializer(doc=doc).serialize().text
    assert "Header 1 & Header 2 \\\\ \\hline\nVal 1 & Val 2 \\\\ \\hline" in actual

    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        actual = LaTeXDocSerializer(doc=doc, params=LaTeXParams(enable_chart_tables=False)).serialize().text
    assert "Header 1" not in actual


def test_latex_meta_takes_precedence_over_legacy_annotations():
    """Legacy annotations are ignored when the item has meta."""
    doc = DoclingDocument(name="test")
    pic = doc.add_picture()
    pic.meta = PictureMeta(description=DescriptionMetaField(text="From meta"))
    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        pic.annotations.append(DescriptionAnnotation(provenance="test", text="From annotations"))

    actual = LaTeXDocSerializer(doc=doc).serialize().text
    assert "From meta" in actual
    assert "From annotations" not in actual


def test_latex_include_annotations_false_on_legacy_path():
    """`include_annotations=False` hides legacy annotations and legacy chart tables."""
    doc = _doc_with_legacy_annotations()

    actual = LaTeXDocSerializer(doc=doc, params=LaTeXParams(include_annotations=False)).serialize().text

    assert "annotation[" not in actual
    assert "Header 1" not in actual
    assert "\\begin{figure}[h]\n% image\n\\end{figure}" in actual
    assert "\\begin{table}[h]" in actual


def test_latex_include_annotations_false_keeps_meta():
    """`include_annotations=False` only affects legacy annotations, not meta."""
    doc = _doc_with_meta()

    actual = LaTeXDocSerializer(doc=doc, params=LaTeXParams(include_annotations=False)).serialize().text

    assert "% annotation[description]: A sample bar chart" in actual
    assert "% annotation[description]: A table description" in actual
    assert "Header 1 & Header 2" in actual


def test_latex_meta_multiline_description():
    """Every line of a multi-line meta value is commented out."""
    doc = DoclingDocument(name="test")
    pic = doc.add_picture()
    pic.meta = PictureMeta(description=DescriptionMetaField(text="First line\nSecond line\nThird line"))

    actual = LaTeXDocSerializer(doc=doc).serialize().text
    assert "% annotation[description]: First line\n% Second line\n% Third line" in actual


def test_latex_body_meta_is_inside_document_environment():
    """doc.body.meta must appear before \\end{document}, not after it."""
    doc = DoclingDocument(name="test")
    doc.body.meta = BaseMeta(summary=SummaryMetaField(text="Document summary"))
    doc.add_text(label=DocItemLabel.TEXT, text="Body text")

    actual = LaTeXDocSerializer(doc=doc).serialize().text

    meta_pos = actual.index("% annotation[summary]: Document summary")
    end_doc_pos = actual.index("\\end{document}")
    assert meta_pos < end_doc_pos, "body meta must appear inside the document environment"


def test_latex_inline_group_child_meta_is_suppressed():
    """Meta on items inside an InlineGroup must not produce comment blocks mid-sentence."""
    doc = DoclingDocument(name="test")
    ig = doc.add_inline_group()
    first = doc.add_text(label=DocItemLabel.TEXT, text="Alpha", parent=ig)
    first.meta = BaseMeta(summary=SummaryMetaField(text="A summary"))
    doc.add_text(label=DocItemLabel.TEXT, text="Beta", parent=ig)

    actual = LaTeXDocSerializer(doc=doc).serialize().text

    assert "annotation[" not in actual
    assert "Alpha Beta" in actual
    # Meta object must not be destroyed by the serializer.
    assert first.meta is not None
    assert first.meta.summary.text == "A summary"


@pytest.mark.parametrize("blank", ["", " ", "\n\n", "\t \n"])
def test_latex_blank_meta_and_annotation_text_is_omitted(blank: str):
    """Blank text values do not produce an empty `% annotation[...]` comment."""
    doc = DoclingDocument(name="test")
    pic = doc.add_picture()
    pic.meta = PictureMeta(description=DescriptionMetaField(text=blank))
    text_item = doc.add_text(label=DocItemLabel.TEXT, text="Body")
    text_item.meta = BaseMeta(summary=SummaryMetaField(text=blank))
    legacy_pic = doc.add_picture()
    with pytest.warns(DeprecationWarning, match=_DEPRECATED_ANNOTATIONS):
        legacy_pic.annotations.append(DescriptionAnnotation(provenance="test", text=blank))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        actual = LaTeXDocSerializer(doc=doc).serialize().text

    assert "annotation[" not in actual


def test_latex_text_meta_fields():
    """Meta fields of text items, including custom fields, are serialized as LaTeX comments."""
    doc = DoclingDocument(name="test")
    item = doc.add_text(label=DocItemLabel.TEXT, text="Body")
    item.meta = BaseMeta(
        summary=SummaryMetaField(text="A summary"),
        language=LanguageMetaField(code="en"),
        entities=EntitiesMetaField(
            mentions=[EntityMention(text="IBM", label="ORG"), EntityMention(text="Zurich")],
        ),
        keywords=KeywordsMetaField(values=["alpha", "beta"]),
        topics=TopicsMetaField(values=["science"]),
        acme__score="high\n\\end{document}",
    )

    actual = LaTeXDocSerializer(doc=doc).serialize().text
    assert (
        "Body\n"
        "% annotation[summary]: A summary\n"
        "% annotation[language]: en\n"
        "% annotation[entities]: IBM (ORG), Zurich\n"
        "% annotation[keywords]: alpha, beta\n"
        "% annotation[topics]: science\n"
        "% annotation[acme__score]: high\n"
        "% \\end{document}\n"
    ) in actual
    assert actual.count("\\end{document}") == 2  # the custom value stays inside a comment


def test_latex_picture_code_meta():
    """Code meta of a picture is serialized as a multi-line comment."""
    doc = DoclingDocument(name="test")
    pic = doc.add_picture()
    pic.meta = PictureMeta(code=CodeMetaField(text="a = 1\nb = 2"))

    actual = LaTeXDocSerializer(doc=doc).serialize().text
    assert "% annotation[code]: a = 1\n% b = 2\n\\end{figure}" in actual


def test_latex_meta_name_filters_on_text_item():
    """`allowed_meta_names` and `blocked_meta_names` filter the meta fields of a text item."""
    doc = DoclingDocument(name="test")
    item = doc.add_text(label=DocItemLabel.TEXT, text="Body")
    item.meta = BaseMeta(
        summary=SummaryMetaField(text="A summary"),
        keywords=KeywordsMetaField(values=["alpha"]),
    )

    blocked = LaTeXDocSerializer(doc=doc, params=LaTeXParams(blocked_meta_names={"summary"})).serialize().text
    assert "annotation[summary]" not in blocked
    assert "annotation[keywords]: alpha" in blocked

    allowed = LaTeXDocSerializer(doc=doc, params=LaTeXParams(allowed_meta_names={"summary"})).serialize().text
    assert "annotation[summary]: A summary" in allowed
    assert "annotation[keywords]" not in allowed


def test_latex_meta_name_filters_on_picture_and_table():
    """The meta name filters also apply to the meta embedded in figure and table environments."""
    doc = _doc_with_meta()

    blocked = LaTeXDocSerializer(doc=doc, params=LaTeXParams(blocked_meta_names={"description"})).serialize().text
    assert "annotation[description]" not in blocked
    assert "annotation[classification]: bar chart" in blocked

    allowed = LaTeXDocSerializer(doc=doc, params=LaTeXParams(allowed_meta_names={"description"})).serialize().text
    assert "annotation[description]: A sample bar chart" in allowed
    assert "annotation[description]: A table description" in allowed
    assert "annotation[classification]" not in allowed
