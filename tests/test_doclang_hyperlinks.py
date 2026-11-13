"""Hyperlink targets and their text scopes survive DocLang and DCLX conversion."""

from pathlib import Path

import pytest
from pydantic import AnyUrl

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.transforms.serializer.doclang import DocLangDocSerializer, DocLangParams
from docling_core.types.doc import (
    CodeLanguageLabel,
    DocItemLabel,
    DoclingDocument,
    Formatting,
    RichTableCell,
    TableCell,
    TableData,
    TextItem,
)
from tests.doclang_validation import assert_valid_dclg_xml


def _text_runs(doc: DoclingDocument) -> list[tuple[str, str | None, Formatting | None]]:
    return [
        (item.text, str(item.hyperlink) if item.hyperlink is not None else None, item.formatting)
        for item, _ in doc.iterate_items(traverse_pictures=True)
        if isinstance(item, TextItem) and item.text
    ]


@pytest.mark.parametrize("indent", [None, "  "])
@pytest.mark.parametrize(
    "target",
    [
        AnyUrl("https://example.com/docs?a=1&b=2#part"),
        AnyUrl("mailto:maintainer@example.com"),
        Path("../guide.md#example"),
        Path("#details"),
    ],
)
def test_inline_link_roundtrip(target: AnyUrl | Path, indent: str | None) -> None:
    doc = DoclingDocument(name="inline-link")
    group = doc.add_inline_group()
    doc.add_text(label=DocItemLabel.TEXT, text="before ", parent=group)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="linked",
        parent=group,
        hyperlink=target,
        formatting=Formatting(bold=True, italic=True),
    )
    doc.add_text(label=DocItemLabel.TEXT, text=" after", parent=group)

    xml = DocLangDocSerializer(doc=doc, params=DocLangParams(pretty_indentation=indent)).serialize().text
    assert_valid_dclg_xml(xml)
    loaded = DocLangDocDeserializer().deserialize_str(xml)

    assert _text_runs(loaded) == _text_runs(doc)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            '<text><href uri="https://example.com/?a=1&amp;b=2#part"/>linked</text>',
            [("linked", "https://example.com/?a=1&b=2#part", None)],
        ),
        (
            '<text><href uri="https://example.com/"/>before <bold>linked</bold> after</text>',
            [
                ("before ", "https://example.com/", None),
                ("linked", "https://example.com/", Formatting(bold=True)),
                (" after", "https://example.com/", None),
            ],
        ),
        (
            '<text><href uri="https://example.com/outer"/>before '
            '<text><href uri="https://example.com/inner"/>inner</text> after</text>',
            [
                ("before ", "https://example.com/outer", None),
                ("inner", "https://example.com/inner", None),
                (" after", "https://example.com/outer", None),
            ],
        ),
        (
            '<text>before <text><href uri="https://example.com/"/>linked</text> after</text>',
            [("before ", None, None), ("linked", "https://example.com/", None), (" after", None, None)],
        ),
    ],
)
def test_read_href_scope(body: str, expected: list[tuple[str, str | None, Formatting | None]]) -> None:
    xml = f'<doclang version="0.9">{body}</doclang>'
    assert_valid_dclg_xml(xml)

    loaded = DocLangDocDeserializer().deserialize_str(xml)

    assert _text_runs(loaded) == expected


@pytest.mark.parametrize("container", ["paragraph", "heading", "list", "table"])
def test_adjacent_links_preserve_text_and_targets(container: str, tmp_path: Path) -> None:
    doc = DoclingDocument(name="adjacent-links")
    if container == "paragraph":
        group = doc.add_inline_group()
    elif container == "heading":
        heading = doc.add_heading(text="", level=1)
        group = doc.add_inline_group(parent=heading)
    elif container == "list":
        item = doc.add_list_item(text="", parent=doc.add_list_group())
        group = doc.add_inline_group(parent=item)
    else:
        table = doc.add_table(data=TableData(num_rows=1, num_cols=1))
        group = doc.add_inline_group(parent=table)
        doc.add_table_cell(
            table,
            RichTableCell(
                text="firstsecond",
                ref=group.get_ref(),
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
            ),
        )
    doc.add_text(label=DocItemLabel.TEXT, text="first", parent=group, hyperlink=AnyUrl("https://example.com/first"))
    doc.add_text(label=DocItemLabel.TEXT, text="second", parent=group, hyperlink=AnyUrl("https://example.com/second"))

    path = tmp_path / "adjacent-links.dclx"
    doc.save_as_doclang_archive(path)
    loaded = DoclingDocument.load_from_doclang_archive(path)

    assert _text_runs(loaded) == _text_runs(doc)


def test_conversation_links_survive_dclx(tmp_path: Path) -> None:
    doc = DoclingDocument(name="conversation")
    doc.add_title(text="Source", hyperlink=AnyUrl("https://github.com/owner/repo/issues/1"))
    doc.add_heading(text="Labels", level=1, hyperlink=AnyUrl("https://github.com/owner/repo/labels"))
    table = doc.add_table(data=TableData(num_rows=1, num_cols=2))
    label_group = doc.add_inline_group(parent=table)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="bug",
        parent=label_group,
        hyperlink=AnyUrl("https://github.com/owner/repo/labels/bug"),
    )
    doc.add_table_cell(
        table,
        RichTableCell(
            text="bug",
            ref=label_group.get_ref(),
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
        ),
    )
    doc.add_table_cell(
        table,
        TableCell(
            text="Unexpected behavior",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
        ),
    )
    doc.add_list_item(
        text="Comment",
        parent=doc.add_list_group(),
        hyperlink=AnyUrl("https://github.com/owner/repo/issues/1#issuecomment-2"),
    )
    doc.add_code(
        text='\n    value = "<tag>&amp;</tag>"  \n\n',
        code_language=CodeLanguageLabel.PYTHON,
        hyperlink=AnyUrl("https://github.com/owner/repo/pull/3#discussion_r4"),
    )

    xml = doc.export_to_doclang()
    assert_valid_dclg_xml(xml)
    path = tmp_path / "conversation.dclx"
    doc.save_as_doclang_archive(path)
    loaded = DoclingDocument.load_from_doclang_archive(path)

    assert _text_runs(loaded) == _text_runs(doc)
    assert loaded.tables[0].data.grid[0][1].text == "Unexpected behavior"


def test_linked_host_with_unlinked_run_roundtrip() -> None:
    doc = DoclingDocument(name="linked-host")
    footnote = doc.add_text(label=DocItemLabel.FOOTNOTE, text="", hyperlink=AnyUrl("https://example.com/host"))
    group = doc.add_inline_group(parent=footnote)
    doc.add_text(label=DocItemLabel.TEXT, text="plain ", parent=group)
    doc.add_text(label=DocItemLabel.TEXT, text="linked", parent=group, hyperlink=AnyUrl("https://example.com/run"))

    xml = DocLangDocSerializer(doc=doc).serialize().text
    assert_valid_dclg_xml(xml)
    loaded = DocLangDocDeserializer().deserialize_str(xml)

    assert _text_runs(loaded) == _text_runs(doc)


def test_heading_own_href_is_not_copied_to_runs() -> None:
    xml = '<doclang version="0.9"><heading level="2"><href uri="https://example.com/"/>a <formula>x</formula></heading></doclang>'
    loaded = DocLangDocDeserializer().deserialize_str(xml)

    assert [str(it.hyperlink) for it, _ in loaded.iterate_items() if isinstance(it, TextItem)] == [
        "https://example.com/"
    ]


def test_link_shared_by_all_runs_moves_to_enclosing_element() -> None:
    doc = DoclingDocument(name="shared-link")
    group = doc.add_inline_group()
    link = AnyUrl("https://example.com/")
    doc.add_text(label=DocItemLabel.TEXT, text="plain ", parent=group, hyperlink=link)
    doc.add_text(label=DocItemLabel.TEXT, text="bold", parent=group, hyperlink=link, formatting=Formatting(bold=True))

    xml = DocLangDocSerializer(doc=doc, params=DocLangParams(pretty_indentation=None)).serialize().text
    assert xml.count("<href ") == 1
    assert_valid_dclg_xml(xml)
    loaded = DocLangDocDeserializer().deserialize_str(xml)

    assert _text_runs(loaded) == _text_runs(doc)


def test_heading_with_shared_link_keeps_its_runs() -> None:
    doc = DoclingDocument(name="heading-shared-link")
    link = AnyUrl("https://example.com/")
    heading = doc.add_heading(text="", level=1, hyperlink=link)
    group = doc.add_inline_group(parent=heading)
    doc.add_text(label=DocItemLabel.TEXT, text="a ", parent=group, hyperlink=link)
    doc.add_text(label=DocItemLabel.TEXT, text="b", parent=group, hyperlink=link, formatting=Formatting(bold=True))

    xml = DocLangDocSerializer(doc=doc).serialize().text
    assert_valid_dclg_xml(xml)
    loaded = DocLangDocDeserializer().deserialize_str(xml)

    assert _text_runs(loaded) == _text_runs(doc)
    # The heading has no text of its own for its link to cover, so only the runs keep it.
    assert loaded.texts[0].hyperlink is None


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("linked", [(True, True, True), (True, False, True), (False, True, False)])
def test_threaded_heading_keeps_run_order_and_links(level: int, linked: tuple[bool, bool, bool]) -> None:
    fragments = []
    expected = []
    location = '<location value="10"/><location value="20"/><location value="30"/><location value="40"/>'
    for index, (text, has_link) in enumerate(zip(("One", "Two", "Three"), linked)):
        target = f"https://example.com/{index}" if has_link else None
        body = f'<text><href uri="{target}"/>{text}</text>' if target else text
        fragments.append(f'<heading level="{level}"><thread thread_id="1"/>{location}{body}</heading>')
        expected.append((text, target, None))
    xml = '<doclang version="0.9">' + "<page_break/>".join(fragments) + "</doclang>"
    assert_valid_dclg_xml(xml)

    loaded = DocLangDocDeserializer().deserialize_str(xml)

    assert _text_runs(loaded) == expected
    headings = [item for item in loaded.texts if item.label in {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}]
    assert len(headings) == 1
    assert headings[0].label == (DocItemLabel.TITLE if level == 1 else DocItemLabel.SECTION_HEADER)
    assert [prov.page_no for prov in headings[0].prov] == [1, 2, 3]
    assert all(prov.charspan == (0, 0) for prov in headings[0].prov)

    serialized = DocLangDocSerializer(doc=loaded, params=DocLangParams(pretty_indentation=None)).serialize().text
    assert_valid_dclg_xml(serialized)
    reloaded = DocLangDocDeserializer().deserialize_str(serialized)
    assert _text_runs(reloaded) == expected
