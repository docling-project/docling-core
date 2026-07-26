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
from typing import Optional

import pytest
from pydantic import AnyUrl

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.transforms.serializer.doclang import DocLangDocSerializer, DocLangParams
from docling_core.transforms.serializer.latex import LaTeXDocSerializer
from docling_core.types.doc.document import (
    CURRENT_VERSION,
    DocItemLabel,
    DoclingDocument,
    Formatting,
)

# (name, runs) where a run is (text, formatting, hyperlink)
Run = tuple[str, Optional[Formatting], Optional[str]]

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
                "<doclang><text><bold>Advanced Topics</bold><content> in </content>"
                "<italic>Machine Learning</italic></text></doclang>"
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
            "doclang": "<doclang><text><content>This is </content><italic>italic text</italic>.</text></doclang>",
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
            "doclang": "<doclang><text><bold><content>bold </content></bold>tail</text></doclang>",
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
            "doclang": "<doclang><text><bold><content>bold </content></bold>tail</text></doclang>",
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
            "doclang": "<doclang><text>a<bold><content> </content></bold>b</text></doclang>",
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
    """DocLang carries the run whitespace losslessly, via `<content>`.

    Checked both pretty-printed (the default, where bare text nodes pick up indentation) and
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
