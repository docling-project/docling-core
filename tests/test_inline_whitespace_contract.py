"""The InlineGroup whitespace contract.

Runs carry their own significant whitespace; serializers concatenate them faithfully and never
insert a separator of their own. This module is the contract table: one set of inline-run
sequences checked against every text format, plus a DocLang round-trip identity check and the
load-time normalization that keeps pre-contract documents rendering as they used to.
"""

import copy
import json
import re
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import AnyUrl

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.transforms.serializer.doclang import DocLangDocSerializer, DocLangParams
from docling_core.transforms.serializer.html import HTMLDocSerializer
from docling_core.transforms.serializer.latex import LaTeXDocSerializer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc.document import (
    CURRENT_VERSION,
    CodeLanguageLabel,
    DocItemLabel,
    DoclingDocument,
    Formatting,
    InlineGroup,
)

# (name, runs) where a run is (text, formatting, hyperlink)
Run = tuple[str, Formatting | None, str | None]

_BOLD = Formatting(bold=True)
_ITALIC = Formatting(italic=True)
_SUB = Formatting(script="sub")
_SUPER = Formatting(script="super")

# Expected output per format. `html` is the inline-group span only, `latex`/`doclang` the body.
CONTRACT_TABLE: list[tuple[str, list[Run], dict[str, str]]] = [
    (
        "ordinal",
        [("2", None, None), ("nd", _SUPER, None)],
        {
            "md": "2nd",
            "itxt": "2nd",
            "html": "2<sup>nd</sup>",
            "latex": "2$^{nd}$",
            "doclang": "<doclang><text>2<superscript>nd</superscript></text></doclang>",
        },
    ),
    (
        "formula_unit",
        [("O", None, None), ("2", _SUB, None), (".", None, None)],
        {
            "md": "O2.",
            "itxt": "O2.",
            "html": "O<sub>2</sub>.",
            "latex": "O$_{2}$.",
            "doclang": "<doclang><text>O<subscript>2</subscript>.</text></doclang>",
        },
    ),
    (
        "water",
        [("H", None, None), ("2", _SUB, None), ("O", None, None)],
        {
            "md": "H2O",
            "itxt": "H2O",
            "html": "H<sub>2</sub>O",
            "latex": "H$_{2}$O",
            "doclang": "<doclang><text>H<subscript>2</subscript>O</text></doclang>",
        },
    ),
    (
        "spaced_emphasis",
        [("Advanced Topics", _BOLD, None), (" in ", None, None), ("Machine Learning", _ITALIC, None)],
        {
            "md": "**Advanced Topics** in *Machine Learning*",
            "itxt": "Advanced Topics in Machine Learning",
            "html": "<strong>Advanced Topics</strong> in <em>Machine Learning</em>",
            "latex": "\\textbf{Advanced Topics} in \\textit{Machine Learning}",
            "doclang": (
                "<doclang><text><bold>Advanced Topics</bold> in <italic>Machine Learning</italic></text></doclang>"
            ),
        },
    ),
    (
        "trailing_period",
        [("This is ", None, None), ("italic text", _ITALIC, None), (".", None, None)],
        {
            "md": "This is *italic text*.",
            "itxt": "This is italic text.",
            "html": "This is <em>italic text</em>.",
            "latex": "This is \\textit{italic text}.",
            "doclang": "<doclang><text>This is <italic>italic text</italic>.</text></doclang>",
        },
    ),
    (
        # Whitespace *inside* a formatted run: it must end up outside the emphasis markers,
        # otherwise CommonMark renders `**bold **` as literal asterisks.
        "whitespace_inside_bold",
        [("bold ", _BOLD, None), ("tail", None, None)],
        {
            "md": "**bold** tail",
            "itxt": "bold tail",
            "html": "<strong>bold</strong> tail",
            "latex": "\\textbf{bold} tail",
            "doclang": "<doclang><text><bold>bold </bold>tail</text></doclang>",
        },
    ),
    (
        # ... and outside the *complete* decoration stack, hyperlink included.
        "whitespace_inside_bold_link",
        [("bold ", _BOLD, "https://example.com/x"), ("tail", None, None)],
        {
            "md": "[**bold**](https://example.com/x) tail",
            "itxt": "bold tail",
            "html": '<a href="https://example.com/x"><strong>bold</strong></a> tail',
            "latex": "\\href{https://example.com/x}{\\textbf{bold}} tail",
            # DocLang has no place for a hyperlink on an inline run (pre-existing gap, not
            # part of this contract): the href is dropped. The whitespace still hoists.
            "doclang": "<doclang><text><bold>bold </bold>tail</text></doclang>",
        },
    ),
    (
        # A whitespace-only run survives as one space, with no empty markers.
        "whitespace_only_run",
        [("a", None, None), (" ", _BOLD, None), ("b", None, None)],
        {
            "md": "a b",
            "itxt": "a b",
            "html": "a b",
            "latex": "a b",
            "doclang": "<doclang><text>a<bold> </bold>b</text></doclang>",
        },
    ),
]


def _build(runs: list[Run]) -> DoclingDocument:
    doc = DoclingDocument(name="contract")
    group = doc.add_inline_group()
    for text, formatting, hyperlink in runs:
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            parent=group,
            formatting=formatting,
            hyperlink=AnyUrl(hyperlink) if hyperlink else None,
        )
    return doc


def _inline_span(doc: DoclingDocument) -> str:
    html = doc.export_to_html(split_page_view=False)
    match = re.search(r"<span class='inline-group'>(.*?)</span>\s*$", html, re.DOTALL | re.MULTILINE)
    if match is None:
        match = re.search(r"<span class='inline-group'>(.*)</span>", html, re.DOTALL)
    assert match is not None, html
    return match.group(1)


def _latex_body(doc: DoclingDocument) -> str:
    text = LaTeXDocSerializer(doc=doc).serialize().text
    body = text.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
    return body.strip()


def _doclang_compact(doc: DoclingDocument) -> str:
    """Serialize to DocLang without pretty-printing, so whitespace is unambiguous."""
    params = DocLangParams(include_version=False, pretty_indentation=None)
    return DocLangDocSerializer(doc=doc, params=params).serialize().text.strip()


@pytest.mark.parametrize(("name", "runs", "expected"), CONTRACT_TABLE, ids=[c[0] for c in CONTRACT_TABLE])
def test_contract_table(name: str, runs: list[Run], expected: dict[str, str]) -> None:
    """Inline runs are concatenated faithfully in every text format."""
    doc = _build(runs)
    actual = {
        "md": doc.export_to_markdown(),
        "itxt": doc.export_to_text(),
        "html": _inline_span(doc),
        "latex": _latex_body(doc),
        "doclang": _doclang_compact(doc),
    }
    assert actual == expected


@pytest.mark.parametrize(("name", "runs", "expected"), CONTRACT_TABLE, ids=[c[0] for c in CONTRACT_TABLE])
def test_doclang_roundtrip_is_lossless(name: str, runs: list[Run], expected: dict[str, str]) -> None:
    """DocLang carries the run whitespace losslessly: bare where it reads as written, else in `<content>`.

    Checked both pretty-printed (the default, which must keep each text block on one line) and
    compact, since the whitespace channel must survive the pretty printer.
    """
    doc = _build(runs)
    # DocLang drops hyperlinks on inline runs (pre-existing, see the table), so compare the
    # whitespace rather than the full markdown for that one case.
    expected_md = expected["md"] if not any(link for _, _, link in runs) else "**bold** tail"
    for params in (DocLangParams(), DocLangParams(pretty_indentation=None)):
        xml = DocLangDocSerializer(doc=doc, params=params).serialize().text
        roundtripped = DocLangDocDeserializer().deserialize_str(xml)
        assert roundtripped.export_to_markdown() == expected_md, (name, params.pretty_indentation)
        assert roundtripped.export_to_text() == doc.export_to_text(), (name, params.pretty_indentation)


def test_no_separator_is_invented_between_runs() -> None:
    """The serializers add nothing of their own: joined run text == the plain text export."""
    for name, runs, _ in CONTRACT_TABLE:
        doc = _build(runs)
        assert doc.export_to_text() == "".join(text for text, _, _ in runs), name


def test_host_text_stays_apart_from_its_inline_group() -> None:
    """Children of a text item are block structure, so the host text keeps its separator.

    Producers put a section's first paragraph under its heading; joining the two with "" would
    read ``ResultsWe use bold words.``. Host boundaries match the pre-contract output; only the
    runs inside the group are concatenated.
    """
    doc = DoclingDocument(name="host")
    heading = doc.add_heading(text="Results", level=1)
    group = doc.add_inline_group(parent=heading)
    for text, formatting in [("We use ", None), ("bold", _BOLD), (" words.", None)]:
        doc.add_text(label=DocItemLabel.TEXT, text=text, parent=group, formatting=formatting)
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Table")
    picture = doc.add_picture()
    picture.captions.append(caption.get_ref())
    caption_group = doc.add_inline_group(parent=caption)
    doc.add_text(label=DocItemLabel.TEXT, text="1", parent=caption_group, formatting=_BOLD)
    doc.add_text(label=DocItemLabel.TEXT, text=": results", parent=caption_group)

    assert heading._get_plain_text(doc) == "Results We use bold words."
    assert doc.export_to_markdown().startswith("## Results\n\nWe use **bold** words.")
    html = HTMLDocSerializer(doc=doc).serialize().text
    assert "<h2>Results</h2>\n<span class='inline-group'>We use <strong>bold</strong> words.</span>" in html
    assert _doclang_compact(doc).startswith(
        '<doclang><heading level="2">Results We use <bold>bold</bold> words.</heading>'
    )
    assert picture.caption_text(doc) == "Table 1: results"
    assert MarkdownDocSerializer(doc=doc).serialize_captions(item=picture).text == "Table **1**: results"


def test_list_item_text_stays_apart_from_its_inline_group() -> None:
    """A list item's own text and its inline-group child follow the same host rule in Markdown."""
    doc = DoclingDocument(name="host")
    lst = doc.add_list_group()
    item = doc.add_list_item(text="Intro", parent=lst)
    group = doc.add_inline_group(parent=item)
    for text, formatting in [("more ", None), ("x", _BOLD), (" y", None)]:
        doc.add_text(label=DocItemLabel.TEXT, text=text, parent=group, formatting=formatting)

    assert doc.export_to_markdown() == "- Intro more **x** y"
    assert item._get_plain_text(doc) == "Intro more x y"


# --------------------------------------------------------------------------------------------
# Load normalization for pre-contract documents
# --------------------------------------------------------------------------------------------

_LEGACY_VERSION = "1.8.0"


def _legacy_doc_dict(runs: list[str], *, version: str = _LEGACY_VERSION, label: str = "text") -> dict:
    return {
        "schema_name": "DoclingDocument",
        "version": version,
        "name": "legacy",
        "body": {
            "name": "_root_",
            "self_ref": "#/body",
            "label": "unspecified",
            "children": [{"$ref": "#/groups/0"}],
        },
        "furniture": {"name": "_root_", "self_ref": "#/furniture", "label": "unspecified", "children": []},
        "groups": [
            {
                "self_ref": "#/groups/0",
                "parent": {"$ref": "#/body"},
                "label": "inline",
                "name": "group",
                "content_layer": "body",
                "children": [{"$ref": f"#/texts/{i}"} for i in range(len(runs))],
            }
        ],
        "texts": [
            {
                "self_ref": f"#/texts/{i}",
                "parent": {"$ref": "#/groups/0"},
                "label": label,
                "prov": [],
                "orig": text,
                "text": text,
                "children": [],
                "content_layer": "body",
            }
            for i, text in enumerate(runs)
        ],
        "pictures": [],
        "tables": [],
        "key_value_items": [],
        "form_items": [],
        "pages": {},
    }


def _load(data: dict) -> DoclingDocument:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return DoclingDocument.model_validate(json.loads(json.dumps(data)))


@pytest.mark.parametrize(
    ("runs", "expected_runs", "expected_md"),
    [
        # Space-less DOCX-style runs stay spaced, as the old `" "` join rendered them.
        (["Normal", "italic"], ["Normal ", "italic"], "Normal italic"),
        # The legacy extra-space bug is retained: this used to render `H 2 O`.
        (["H", "2", "O"], ["H ", "2 ", "O"], "H 2 O"),
        # An existing boundary keeps its legacy double space -- the old serializer added one too.
        (["left ", "right"], ["left  ", "right"], "left  right"),
    ],
)
def test_legacy_documents_keep_their_separators(runs: list[str], expected_runs: list[str], expected_md: str) -> None:
    doc = _load(_legacy_doc_dict(runs))
    assert [t.text for t in doc.texts] == expected_runs
    assert doc.export_to_markdown() == expected_md


def test_current_version_documents_are_not_normalized() -> None:
    doc = _load(_legacy_doc_dict(["H", "2", "O"], version=CURRENT_VERSION))
    assert [t.text for t in doc.texts] == ["H", "2", "O"]
    assert doc.export_to_markdown() == "H2O"


def test_versionless_input_is_not_guessed_at() -> None:
    data = _legacy_doc_dict(["H", "2", "O"])
    del data["version"]
    assert [t.text for t in _load(data).texts] == ["H", "2", "O"]


def test_migration_is_one_way() -> None:
    """Loading stamps CURRENT_VERSION, so a re-save cannot pick up a second separator."""
    once = _load(_legacy_doc_dict(["H", "2"]))
    assert once.version == CURRENT_VERSION
    twice = _load(json.loads(once.model_dump_json()))
    assert [t.text for t in twice.texts] == [t.text for t in once.texts] == ["H ", "2"]


@pytest.mark.parametrize(
    ("labels", "expected_md", "expected_txt"),
    [
        # The separator goes on whichever neighbour is not delimiter-wrapped ...
        (("text", "code"), "a `b`", "a b"),
        (("code", "text"), "`a` b", "a b"),
        (("text", "formula"), "a $b$", "a $b$"),
        (("formula", "text"), "$a$ b", "$a$ b"),
        # ... and when both are, it becomes a plain run of its own, because appending to
        # either side would produce `a``b` or $a$$b$.
        (("code", "code"), "`a` `b`", "a b"),
        (("formula", "formula"), "$a$ $b$", "$a$ $b$"),
        (("code", "formula"), "`a` $b$", "a $b$"),
        (("formula", "code"), "$a$ `b`", "$a$ b"),
    ],
)
def test_legacy_separator_never_lands_inside_a_delimiter(
    labels: tuple[str, str], expected_md: str, expected_txt: str
) -> None:
    data = _legacy_doc_dict(["a", "b"])
    for item, label in zip(data["texts"], labels):
        item["label"] = label
    doc = _load(data)
    assert doc.export_to_markdown() == expected_md
    assert doc.export_to_text() == expected_txt


def _load_once(data: dict) -> DoclingDocument:
    """Load a legacy dict, checking that the caller's value is untouched and a re-load is stable."""
    pristine = copy.deepcopy(data)
    doc = _load(data)
    assert data == pristine
    assert _load(json.loads(doc.model_dump_json())).export_to_markdown() == doc.export_to_markdown()
    return doc


def _locate(data: dict, pages: list[int]) -> dict:
    data["pages"] = {str(p): {"page_no": p, "size": {"width": 100, "height": 100}} for p in set(pages)}
    for item, page in zip(data["texts"], pages):
        bbox = {"l": 0, "t": 0, "r": 10, "b": 10, "coord_origin": "BOTTOMLEFT"}
        item["prov"] = [{"page_no": page, "bbox": bbox, "charspan": [0, len(item["text"])]}]
    return data


@pytest.mark.parametrize(
    ("label", "both", "first", "second"),
    [("code", "`a` `b`", "`a`", "`b`"), ("formula", "$a$ $b$", "$a$", "$b$")],
)
def test_legacy_standalone_separator_survives_page_filters_with_its_runs(
    label: str, both: str, first: str, second: str
) -> None:
    """Page filters keep only located items; the separator must survive exactly when both runs do."""
    same_page = _load_once(_locate(_legacy_doc_dict(["a", "b"], label=label), [1, 1]))
    assert same_page.filter(page_nrs={1}).export_to_markdown() == both
    assert same_page.export_to_markdown(page_no=1) == both

    split = _load_once(_locate(_legacy_doc_dict(["a", "b"], label=label), [1, 2]))
    assert split.filter(page_nrs={1}).export_to_markdown() == first
    assert split.filter(page_nrs={2}).export_to_markdown() == second
    assert split.export_to_markdown(page_no=1) == first


def _host(doc: DoclingDocument, group: InlineGroup, label: DocItemLabel, runs: list[str]) -> None:
    """An empty-text item whose visible content is a child inline group."""
    nested = doc.add_inline_group(parent=doc.add_text(label=label, text="", parent=group))
    for text in runs:
        doc.add_text(label=DocItemLabel.TEXT, text=text, parent=nested)


def _nested(doc: DoclingDocument, group: InlineGroup, runs: list[str]) -> None:
    nested = doc.add_inline_group(parent=group)
    for text in runs:
        doc.add_text(label=DocItemLabel.TEXT, text=text, parent=nested)


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        pytest.param(
            lambda d, g: (d.add_text(DocItemLabel.TEXT, "left", parent=g), _host(d, g, DocItemLabel.FORMULA, ["x"])),
            "left $x$",
            id="formula-host-last",
        ),
        pytest.param(
            lambda d, g: (
                d.add_text(DocItemLabel.TEXT, "left", parent=g),
                _host(d, g, DocItemLabel.FORMULA, ["x"]),
                d.add_text(DocItemLabel.TEXT, "right", parent=g),
            ),
            "left $x$ right",
            id="formula-host",
        ),
        pytest.param(
            lambda d, g: (
                d.add_text(DocItemLabel.FORMULA, "l", parent=g),
                _host(d, g, DocItemLabel.FORMULA, ["x"]),
                d.add_text(DocItemLabel.CODE, "r", parent=g),
            ),
            "$l$ $x$ `r`",
            id="formula-host-between-delimited",
        ),
        # The DocLang deserializer writes these two for nested mixed-content elements.
        pytest.param(
            lambda d, g: (
                d.add_text(DocItemLabel.TEXT, "left", parent=g),
                _host(d, g, DocItemLabel.FOOTNOTE, ["a", "b"]),
                d.add_text(DocItemLabel.TEXT, "right", parent=g),
            ),
            "left a b right",
            id="footnote-host",
        ),
        pytest.param(
            lambda d, g: (
                d.add_text(DocItemLabel.TEXT, "left", parent=g),
                _nested(d, g, ["a", "b"]),
                d.add_text(DocItemLabel.TEXT, "right", parent=g),
            ),
            "left a b right",
            id="nested-group",
        ),
    ],
)
def test_legacy_separator_counts_runs_whose_text_is_in_a_child_group(build, expected: str) -> None:
    """An empty host renders its child inline group, so the old join spaced it like any other run."""
    doc = DoclingDocument(name="legacy")
    build(doc, doc.add_inline_group())
    data = doc.model_dump(mode="json") | {"version": _LEGACY_VERSION}
    assert _load_once(data).export_to_markdown() == expected


def test_legacy_migration_does_not_touch_caller_input() -> None:
    """The caller owns the dict it passes in, and repeated validation must be stable.

    The migration inserts a standalone separator run for adjacent delimited runs, so an
    in-place edit would both corrupt the caller's value and compound on every re-validation.
    """
    data = _legacy_doc_dict(["a", "b"])
    data["texts"][0]["label"] = "code"
    data["texts"][1]["label"] = "code"
    pristine = copy.deepcopy(data)

    rendered = [_load(data).export_to_markdown() for _ in range(3)]

    assert rendered == ["`a` `b`"] * 3
    assert data == pristine


def test_legacy_load_warns_once() -> None:
    import docling_core.types.doc.document as document_module

    original = document_module._warned_legacy_inline_separators
    document_module._warned_legacy_inline_separators = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DoclingDocument.model_validate(_legacy_doc_dict(["a", "b"]))
            DoclingDocument.model_validate(_legacy_doc_dict(["c", "d"]))
        messages = [str(w.message) for w in caught if w.category is UserWarning]
        assert len(messages) == 1, messages
        assert _LEGACY_VERSION in messages[0]
    finally:
        document_module._warned_legacy_inline_separators = original


def test_chunk_text_follows_the_contract() -> None:
    """The chunker has no inline serializer, so it inherits the Markdown join.

    That makes chunk text -- and therefore embeddings -- change with this contract. Pinned
    here because the drift is otherwise invisible: nothing in a chunker golden fails, but a
    persisted vector index silently stops matching.
    """
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

    doc = DoclingDocument(name="chunk")
    doc.add_heading(text="Chemistry")
    group = doc.add_inline_group()
    for text, formatting in [
        ("Water is ", None),
        ("H", None),
        ("2", _SUB),
        ("O", None),
        (" and the ", None),
        ("melting point", _BOLD),
        (" is 0 C.", None),
    ]:
        doc.add_text(label=DocItemLabel.TEXT, text=text, parent=group, formatting=formatting)

    chunks = list(HierarchicalChunker().chunk(doc))
    assert [c.text for c in chunks] == ["Water is H2O and the **melting point** is 0 C."]


# DocLang 0.9 whitespace rules: (name, DocLang body, expected plain text)
DOCLANG_WHITESPACE_READINGS: list[tuple[str, str, str]] = [
    (
        "spec_example",
        "<text>\n  Water (H<subscript>2</subscript>O) is <bold>clear </bold>\n  <italic> and odorless</italic>.\n"
        "  This keeps <content>two  spaces</content> as written.\n</text>",
        "Water (H2O) is clear and odorless. This keeps two  spaces as written.",
    ),
    ("newline_next_to_tag", "<text>Hello\n  <bold>world</bold>\n</text>", "Hello world"),
    ("tight_tags", "<text>2<superscript>nd</superscript></text>", "2nd"),
    ("tabs_and_runs", "<text>a \t\r\n b</text>", "a b"),
    ("comment_disregarded", "<text>a<!-- note -->b c <!-- note --> d</text>", "ab c d"),
    ("content_is_literal", "<text><content> a\n</content> b </text>", " a\n b"),
    ("element_head_disregarded", '<text>\n  <location value="1"/>\n  a\n  <label value="x"/>\n</text>', "a"),
    ("virtual_text", "<list>\n  <ldiv/>\n  Four\n  <bold>Five</bold>\n</list>", "- Four Five"),
    ("newline_before_tag", "<text>Hi\n<bold>there</bold></text>", "Hi there"),
    # In a virtual <text>, <code> and <formula> are inline: whitespace around them reads as a space.
    (
        "virtual_text_inline_formula",
        "<list>\n  <ldiv/>\n  A.\n  <formula>f(x)</formula>\n  is odd\n</list>",
        "- A. $f(x)$ is odd",
    ),
    (
        # Its element head is disregarded, so the line breaks around it are one sequence.
        "virtual_text_inline_code_with_head",
        '<list>\n  <ldiv/>\n  <content>run: </content>\n  <code>\n    <label value="Python"/>\n    x\n  </code>\n'
        "  <content> now</content>\n</list>",
        "- run:  x  now",
    ),
    (
        "virtual_text_leading_code",
        "<list><ldiv/><code>foo</code> returns a value</list>",
        "- foo returns a value",
    ),
    ("inline_handwriting", "<text>Check <handwriting>out</handwriting> these</text>", "Check out these"),
    ("inline_rtl", "<text>Check <rtl>out</rtl> these</text>", "Check out these"),
    ("empty_content_between_spaces", "<text>a <content></content> b</text>", "a b"),
    ("whitespace_only_inline_code", "<text>a<code> </code><content>\t</content></text>", "a \t"),
    ("whitespace_only_formatting_run", "<text>a<italic> </italic></text>", "a"),
    ("empty_formatting_run", "<text>a<italic></italic></text>", "a"),
    ("tab_in_formatting_run", "<text>a<italic><content>\t</content></italic></text>", "a\t"),
    # Whitespace means space, tab, CR and LF only: other Unicode spaces are ordinary characters.
    ("nbsp_edges", "<text>\u00a0a\u00a0</text>", "\u00a0a\u00a0"),
    ("nbsp_only", "<text>\u00a0</text>", "\u00a0"),
    ("em_space_edges", "<text>\u2003a\u2003</text>", "\u2003a\u2003"),
    ("nbsp_edges_mixed", "<text>\u00a0<bold>a</bold>\u00a0</text>", "\u00a0a\u00a0"),
    ("nbsp_heading", "<heading>\u00a0a\u00a0</heading>", "\u00a0a\u00a0"),
    ("nbsp_list_item", "<list><ldiv/>\u00a0a\u00a0</list>", "- \u00a0a\u00a0"),
    ("nbsp_caption", "<picture><caption>\u00a0a\u00a0</caption></picture>", "\u00a0a\u00a0"),
]


@pytest.mark.parametrize(
    ("name", "body", "expected"), DOCLANG_WHITESPACE_READINGS, ids=[c[0] for c in DOCLANG_WHITESPACE_READINGS]
)
def test_doclang_reads_whitespace_per_spec(name: str, body: str, expected: str) -> None:
    """DocLang 0.9: a whitespace sequence reads as one space, none at text block edges."""
    doc = DocLangDocDeserializer().deserialize_str(f"<doclang>{body}</doclang>")
    assert doc.export_to_text() == expected, name


def _marker(doc: DoclingDocument) -> str:
    return doc.texts[0].marker


def _text(doc: DoclingDocument) -> str:
    return doc.texts[0].text


def _cell(doc: DoclingDocument) -> str:
    return doc.tables[0].data.table_cells[0].text


DOCLANG_LITERAL_READINGS: list[tuple[str, str, Callable[[DoclingDocument], str], str]] = [
    ("nbsp_marker", "<list><ldiv><marker>\u00a01.</marker></ldiv>a</list>", _marker, "\u00a01."),
    ("nbsp_cell", "<table><fcel/>\u00a0a\u00a0<nl/></table>", _cell, "\u00a0a\u00a0"),
    # `<content>` is literal, wherever it is.
    ("content_marker", "<list><ldiv><marker><content> 1. </content></marker></ldiv>a</list>", _marker, " 1. "),
    ("content_heading", "<heading><content> a </content></heading>", _text, " a "),
    ("content_field_heading", "<field_heading><content> a </content></field_heading>", _text, " a "),
    ("content_list_item", "<list><ldiv/><content> a </content></list>", _text, " a "),
    ("content_list_item_text", "<list><ldiv/><text><content> a </content></text></list>", _text, " a "),
    ("content_list_item_sublist", "<list><ldiv/><content> a </content><list><ldiv/>b</list></list>", _text, " a "),
    ("content_caption", "<picture><caption><content> a </content></caption></picture>", _text, " a "),
    ("content_cell", "<table><fcel/><content> a </content><nl/></table>", _cell, " a "),
    # A table cell is a virtual <text> too, whatever it starts with.
    ("cell_leading_bold", "<table><fcel/><bold>foo</bold> tail<nl/></table>", _cell, "foo tail"),
    ("cell_inline_code", "<table><fcel/>see <code>x</code> tail<nl/></table>", _cell, "see x tail"),
    ("cell_runs_separated", "<table><fcel/><bold>a</bold> <italic>b</italic><nl/></table>", _cell, "a b"),
    ("cell_run_edge_space", "<table><fcel/><bold>a </bold><italic>b</italic><nl/></table>", _cell, "a b"),
    ("cell_handwriting", "<table><fcel/><handwriting>a</handwriting> b<nl/></table>", _cell, "a b"),
]


@pytest.mark.parametrize(
    ("name", "body", "read", "expected"), DOCLANG_LITERAL_READINGS, ids=[c[0] for c in DOCLANG_LITERAL_READINGS]
)
def test_doclang_literal_readings(name: str, body: str, read: Callable[[DoclingDocument], str], expected: str) -> None:
    """What a Markdown export would normalize away, read from the item."""
    assert read(DocLangDocDeserializer().deserialize_str(f"<doclang>{body}</doclang>")) == expected, name


def test_doclang_cell_with_leading_inline_elements_keeps_all_text() -> None:
    """The runs of a virtual-text cell are items too, in an inline group, and read back the same."""
    doc = DocLangDocDeserializer().deserialize_str(
        "<doclang><table><fcel/><bold>foo</bold> tail<nl/><fcel/>see <code>x</code> tail<nl/></table></doclang>"
    )
    assert [t.text for t in doc.texts] == ["foo", " tail", "see ", "x", " tail"]
    roundtripped = DocLangDocDeserializer().deserialize_str(_doclang_compact(doc))
    assert [t.text for t in roundtripped.texts] == [t.text for t in doc.texts]
    assert [c.text for c in roundtripped.tables[0].data.table_cells] == ["foo tail", "see x tail"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("<list><ldiv/>foo <checkbox/> bar</list>", ["foo ", "", " bar"]),
        ("<table><fcel/>foo <checkbox/> bar<nl/></table>", ["foo ", "", " bar"]),
        ("<list><ldiv/><checkbox/> y</list>", ["", "y"]),
        ("<table><fcel/><checkbox/> y<nl/></table>", ["", "y"]),
        ("<list><ldiv/><checkbox/> <code>x</code> <code>y</code></list>", ["", "x", " ", "y"]),
    ],
)
def test_doclang_checkbox_is_part_of_virtual_text(body: str, expected: list[str]) -> None:
    """Runs around a checkbox keep their spaces, but a leading one marks the text after it: no space of its own."""
    doc = DocLangDocDeserializer().deserialize_str(f"<doclang>{body}</doclang>")
    assert [t.text for t in doc.texts if t.label != DocItemLabel.LIST_ITEM] == expected


def test_doclang_attributes_a_space_to_the_first_character() -> None:
    """Rule 4: a sequence spanning several elements belongs to the one holding its first character."""
    doc = DocLangDocDeserializer().deserialize_str(
        "<doclang><text>is <bold>clear </bold>\n<italic> and</italic></text></doclang>"
    )
    assert [(t.text, t.formatting) for t in doc.texts if isinstance(t.parent.resolve(doc), InlineGroup)] == [
        ("is ", None),
        ("clear ", _BOLD),
        ("and", _ITALIC),
    ]


@pytest.mark.parametrize("text", ["two  spaces", "a\tb", "line\nbreak"])
def test_doclang_wraps_whitespace_the_rules_would_collapse(text: str) -> None:
    """Whitespace other than single spaces goes in `<content>`, so it reads as written."""
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    for params in (DocLangParams(), DocLangParams(pretty_indentation=None)):
        xml = DocLangDocSerializer(doc=doc, params=params).serialize().text
        assert "<content>" in xml, params.pretty_indentation
        assert DocLangDocDeserializer().deserialize_str(xml).texts[0].text == text


@pytest.mark.parametrize(
    ("runs", "expected"),
    [
        (["a ", "b"], "<text>a b</text>"),  # unwrapped: reads the same bare
        ([" a", "b"], "<text><content> a</content>b</text>"),  # block start
        (["a", "b "], "<text>a<content>b </content></text>"),  # block end
        (["a ", " b"], "<text><content>a </content><content> b</content></text>"),  # would merge
    ],
)
def test_doclang_keeps_content_only_where_needed(runs: list[str], expected: str) -> None:
    """`<content>` is kept only where the whitespace rules would change the reading."""
    doc = DoclingDocument(name="t")
    group = doc.add_inline_group()
    for text in runs:
        doc.add_text(label=DocItemLabel.TEXT, text=text, parent=group)
    assert _doclang_compact(doc) == f"<doclang>{expected}</doclang>"
    assert DocLangDocDeserializer().deserialize_str(_doclang_compact(doc)).export_to_text() == "".join(runs)


def test_doclang_list_item_with_inline_code_and_formula_roundtrips() -> None:
    """A list item with inline code or formula is a virtual <text>: written on one line, read back as is."""
    doc = DoclingDocument(name="t")
    group_list = doc.add_list_group()
    for add_inline in (
        lambda group: doc.add_code(text='print("x")', code_language=CodeLanguageLabel.PYTHON, parent=group),
        lambda group: doc.add_formula(text="E=mc^2", parent=group),
    ):
        group = doc.add_inline_group(parent=doc.add_list_item(text="", parent=group_list))
        doc.add_text(label=DocItemLabel.TEXT, text="Here ", parent=group)
        add_inline(group)
        doc.add_text(label=DocItemLabel.TEXT, text=" inline", parent=group)

    pretty = DocLangDocSerializer(doc=doc, params=DocLangParams(include_version=False)).serialize().text
    assert '  Here <code><label value="Python"/><![CDATA[print("x")]]></code> inline\n' in pretty
    assert "  Here <formula>E=mc^2</formula> inline\n" in pretty
    for xml in (pretty, _doclang_compact(doc)):
        roundtripped = DocLangDocDeserializer().deserialize_str(xml)
        assert roundtripped.export_to_text() == doc.export_to_text()
        assert [t.text for t in roundtripped.texts] == [t.text for t in doc.texts]


def test_doclang_inline_handwriting_keeps_its_runs_and_label() -> None:
    """The handwritten golden reads back complete, with its spans labeled, and serializes to itself."""
    xml = Path("tests/data/doc/handwritten_text.gt.dclg.xml").read_text()
    doc = DocLangDocDeserializer().deserialize_str(xml)
    assert [(t.text, t.label) for t in doc.texts if t.text][-4:] == [
        ("Check ", DocItemLabel.TEXT),
        ("out", DocItemLabel.HANDWRITTEN_TEXT),
        (" these", DocItemLabel.TEXT),
        (" hand-written spans", DocItemLabel.HANDWRITTEN_TEXT),
    ]
    assert DocLangDocSerializer(doc=doc, params=DocLangParams(include_version=False)).serialize().text.strip() == (
        xml.strip()
    )


def test_doclang_keeps_a_separator_between_inline_code_runs() -> None:
    """Bare, a lone space would no longer make the list item a virtual <text>, and the code would turn into blocks."""
    doc = DoclingDocument(name="t")
    group = doc.add_inline_group(parent=doc.add_list_item(text="", parent=doc.add_list_group()))
    doc.add_code(text="x", parent=group)
    doc.add_text(label=DocItemLabel.TEXT, text=" ", parent=group)
    doc.add_code(text="y", parent=group)
    for params in (DocLangParams(), DocLangParams(pretty_indentation=None)):
        xml = DocLangDocSerializer(doc=doc, params=params).serialize().text
        assert [t.text for t in DocLangDocDeserializer().deserialize_str(xml).texts if t.text] == ["x", " ", "y"]


def test_doclang_pretty_print_keeps_content_closed_and_opened_on_one_line() -> None:
    """A line closing one `<content>` and opening another must keep the blank lines of the latter."""
    doc = DoclingDocument(name="t")
    group = doc.add_inline_group()
    for text, formatting in (("a\n\nb", None), (" ", _BOLD), ("c\n\nd", None)):
        doc.add_text(label=DocItemLabel.TEXT, text=text, parent=group, formatting=formatting)
    xml = DocLangDocSerializer(doc=doc, params=DocLangParams()).serialize().text
    assert not xml.endswith("\n")
    assert "".join(t.text for t in DocLangDocDeserializer().deserialize_str(xml).texts) == "a\n\nb c\n\nd"
