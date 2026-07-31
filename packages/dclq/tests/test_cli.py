import json
import zipfile
from pathlib import Path

from dclq import __version__
from dclq.cli import main

DOCUMENT = """<doclang xmlns="https://www.doclang.ai/ns/v0" version="0.7">
  <heading level="1">Report</heading>
  <heading level="2">Results</heading>
  <text>Revenue grew.</text>
  <text><thread thread_id="9"/>alpha </text>
  <text><thread thread_id="9"/>beta</text>
  <text>See <bold>this <italic>important</italic> article</bold> for details.</text>
  <code><bold>public</bold><content> class Main {}</content></code>
  <formula>E = mc^2</formula>
  <list><ldiv/><content>Termination notice</content></list>
  <page_break/>
  <heading level="3">Details</heading>
  <text>Nested section content that is long enough to truncate.</text>
  <table>
    <ched/>Metric<ched/>Value<nl/>
    <rhed/>Margin<fcel/>48.2%<nl/>
  </table>
  <heading level="2">Other</heading>
  <text>Outside the section.</text>
</doclang>
"""

PROJECTION_DOCUMENT = """<doclang xmlns="https://www.doclang.ai/ns/v0" version="0.7">
  <code><content>  SELECT 1;
  </content></code>
  <picture>
    <caption>Chart caption</caption>
    <summary>Chart summary</summary>
  </picture>
</doclang>
"""


SCOPE_DOCUMENT = """<doclang xmlns="https://www.doclang.ai/ns/v0" version="0.7">
  <heading level="1">First</heading>
  <text>Alpha</text>
  <text>Beta</text>
  <heading level="1">Second</heading>
  <text>Gamma</text>
  <list><ldiv/><content>Item one</content><ldiv/><content>Item two</content></list>
  <table>
    <ched/>Metric<ched/>Value<nl/>
    <rhed/>Margin<fcel/>48.2%<nl/>
  </table>
</doclang>
"""


def _document(tmp_path: Path) -> Path:
    path = tmp_path / "report.dclg"
    path.write_text(DOCUMENT, encoding="utf-8")
    return path


def test_search_returns_source_identity_and_structural_context(tmp_path: Path, capsys) -> None:
    path = _document(tmp_path)

    assert main(["grep", "48\\.2", str(path), "-C", "1", "-A", "0", "--format", "json"]) == 0
    result = json.loads(capsys.readouterr().out)[0]

    assert set(result) == {
        "document",
        "xpaths",
        "logical_type",
        "text",
        "pages",
        "doc_items",
        "matches",
        "context",
        "cell_context",
    }
    assert result["xpaths"] == ["/d:doclang/d:table[1]/d:fcel[1]"]
    assert result["text"] == "48.2%"
    assert result["cell_context"]["row_headers"] == ["Margin"]
    assert result["cell_context"]["column_headers"] == ["Value"]
    assert "table" not in result
    assert result["context"]["before"][0]["xpaths"][0].endswith("/d:rhed[1]")
    assert set(result["context"]["before"][0]) == {
        "document",
        "xpaths",
        "logical_type",
        "text",
        "pages",
        "doc_items",
    }
    assert result["context"]["after"] == []

    assert main(["grep", "beta", str(path), "--format", "json"]) == 0
    threaded = json.loads(capsys.readouterr().out)[0]
    assert threaded["xpaths"] == ["/d:doclang/d:text[2]", "/d:doclang/d:text[3]"]
    assert "context" not in threaded
    assert "truncated" not in threaded


def test_retrieval_commands_share_xpath_addresses(tmp_path: Path, capsys) -> None:
    path = _document(tmp_path)

    assert main(["show", str(path), "/heading[2]", "--section", "--format", "json"]) == 0
    section = json.loads(capsys.readouterr().out)
    assert [record["xpaths"][0] for record in section] == [
        "/d:doclang/d:heading[2]",
        "/d:doclang/d:text[1]",
        "/d:doclang/d:text[2]",
        "/d:doclang/d:text[4]",
        "/d:doclang/d:code[1]",
        "/d:doclang/d:formula[1]",
        "/d:doclang/d:list[1]/d:ldiv[1]",
        "/d:doclang/d:heading[3]",
        "/d:doclang/d:text[5]",
        "/d:doclang/d:table[1]/d:ched[1]",
        "/d:doclang/d:table[1]/d:ched[2]",
        "/d:doclang/d:table[1]/d:rhed[1]",
        "/d:doclang/d:table[1]/d:fcel[1]",
    ]
    assert section[0]["logical_type"] == "heading"
    assert section[0]["pages"] == [1]
    assert section[-1]["logical_type"] == "table_cell"
    assert section[-1]["pages"] == [2]
    assert section[-1]["doc_items"] == ["#/tables/0"]
    assert section[-1]["cell_context"] == {
        "row": 1,
        "column": 1,
        "row_headers": ["Margin"],
        "column_headers": ["Value"],
    }
    assert all("matches" not in record for record in section)
    assert all(record["logical_type"] != "section" for record in section)

    assert (
        main(["show", str(path), "/heading[2] | /heading[3]", "--section", "--max-chars", "8", "--format", "json"]) == 0
    )
    bounded = json.loads(capsys.readouterr().out)
    assert [record["xpaths"][0] for record in bounded] == [record["xpaths"][0] for record in section]
    assert bounded[8]["truncated"] is True
    assert bounded[-1]["text"] == "48.2%"

    assert main(["show", str(path), "/heading[2]", "--section", "-n"]) == 0
    section_text = capsys.readouterr().out
    assert "/heading[2]:Results" in section_text
    assert "/text[1]:Revenue grew." in section_text
    assert "/table[1]/fcel[1]:48.2%" in section_text
    assert "/heading[2]:\n" not in section_text
    assert "Outside the section" not in section_text

    assert main(["show", str(path), "/heading[2]", "--section", "--format", "jsonl"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == len(section)

    assert main(["show", str(path), "/heading[2]", "--section", "-A", "1"]) == 2
    assert "context flags cannot be combined with --section" in capsys.readouterr().err

    assert main(["show", str(path), "/table[1]", "--format", "json"]) == 0
    table = json.loads(capsys.readouterr().out)[0]
    assert isinstance(table["text"], str)
    assert table["table"]["cells"] == [
        {"row": 0, "column": 0, "text": "Metric", "role": "column_header"},
        {"row": 0, "column": 1, "text": "Value", "role": "column_header"},
        {"row": 1, "column": 0, "text": "Margin", "role": "row_header"},
        {"row": 1, "column": 1, "text": "48.2%"},
    ]
    assert "cell_context" not in table

    assert main(["show", str(path), "/doclang/heading[2]", "--raw"]) == 0
    assert 'level="2">Results</heading>' in capsys.readouterr().out

    assert main(["outline", str(path), "--format", "json"]) == 0
    outline = json.loads(capsys.readouterr().out)
    assert [heading["xpaths"][0] for heading in outline] == [
        "/d:doclang/d:heading[1]",
        "/d:doclang/d:heading[2]",
        "/d:doclang/d:heading[3]",
        "/d:doclang/d:heading[4]",
    ]
    assert outline[1]["logical_type"] == "heading"
    assert outline[1]["depth"] == 2
    assert outline[1]["document"] == str(path)

    assert main(["outline", str(path)]) == 0
    outline_text = capsys.readouterr().out
    assert "/heading[1]" in outline_text
    assert "/d:" not in outline_text

    assert main(["select", str(path), "count(/heading)", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["value"] == 4.0
    assert main(["select", str(path), "count(descendant::heading)", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["value"] == 4.0
    assert main(["select", str(path), "/heading[1]", "--format", "json"]) == 0
    selected = json.loads(capsys.readouterr().out)["results"][0]
    assert selected["xpaths"] == ["/d:doclang/d:heading[1]"]
    assert "truncated" not in selected
    assert main(["select", str(path), "/heading[1]", "--format", "text", "--max-chars", "5"]) == 0
    assert capsys.readouterr().out == "Repo…\n"

    assert main(["inspect", str(path), "--format", "json"]) == 0
    inventory = json.loads(capsys.readouterr().out)[0]
    assert set(inventory) == {
        "document",
        "input_type",
        "page_count",
        "semantic_units",
        "elements",
        "metadata_elements",
    }
    assert inventory["semantic_units"] > 0
    assert inventory["elements"]["code"] == 1
    assert "bold" not in inventory["elements"]
    assert "fcel" not in inventory["elements"]
    assert inventory["metadata_elements"] == {}

    assert main(["grep", "Revenue", str(path), "--within-xpath", "/heading[2]", "--section", "-q"]) == 0


def test_text_output_collapses_formatting_and_marks_truncation(tmp_path: Path, capsys, monkeypatch) -> None:
    path = _document(tmp_path)

    assert main(["show", str(path), "/formula[1]", "-B", "1"]) == 0
    output = capsys.readouterr().out
    assert "public" in output
    assert "class Main {}" in output
    assert output.count("public") == 1
    assert "/d:" not in output
    assert "<bold>" not in output
    assert "\n- " not in output
    assert "\n--\n" not in output
    assert "Type:" not in output
    assert "Page:" not in output
    assert "Section:" not in output
    assert str(path) not in output

    assert main(["grep", "-F", "See this important article for details.", str(path)]) == 0
    output = capsys.readouterr().out
    assert "See this important article for details." in output
    assert "<bold>" not in output
    assert "**" not in output
    assert "Type:" not in output
    assert str(path) not in output

    assert main(["grep", "-F", "See this important article for details.", str(path), "-n"]) == 0
    output = capsys.readouterr().out
    assert "/text[4]:See this important article for details." in output
    assert "XPath:" not in output

    assert main(["grep", "Revenue", str(path), "--max-chars", "8"]) == 0
    output = capsys.readouterr().out
    assert output.rstrip().endswith("Revenue…")
    assert "[truncated]" not in output

    assert main(["grep", "Revenue", str(path), "--max-chars", "8", "--format", "json"]) == 0
    record = json.loads(capsys.readouterr().out)[0]
    assert record["text"] == "Revenue…"
    assert record["truncated"] is True

    assert main(["show", str(path), "/formula[1]", "-B", "1", "-n"]) == 0
    output = capsys.readouterr().out
    assert "/code[1]-public  class Main {}" in output
    assert "/formula[1]:$$E = mc^2$$" in output
    assert "/d:" not in output
    assert "\n--\n" not in output
    assert "XPath:" not in output

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("dclq.cli.sys.stdout.isatty", lambda: True)
    assert main(["grep", "Revenue", str(path)]) == 0
    assert "\033[1;31mRevenue\033[0m grew." in capsys.readouterr().out


def test_text_context_merges_overlapping_windows(tmp_path: Path, capsys) -> None:
    path = tmp_path / "overlap.dclg"
    path.write_text(
        """<doclang xmlns="https://www.doclang.ai/ns/v0">
  <text>Before first</text>
  <text>Information one</text>
  <text>Shared context</text>
  <text>Information two</text>
  <text>After second</text>
  <text>Omitted one</text>
  <text>Omitted two</text>
  <text>Before third</text>
  <text>Information three</text>
  <text>After third</text>
</doclang>
""",
        encoding="utf-8",
    )

    assert main(["grep", "Information", str(path), "-C", "1", "-n"]) == 0
    output = capsys.readouterr().out

    assert output.count("\n--\n") == 1
    assert output.count("Shared context") == 1
    assert "/text[2]:Information one" in output
    assert "/text[4]:Information two" in output
    assert "/text[9]:Information three" in output
    assert "/text[2]-" not in output
    assert "/text[4]-" not in output
    assert "Omitted one" not in output
    assert "Omitted two" not in output


def test_hierarchical_hits_use_context_span(tmp_path: Path, capsys) -> None:
    path = tmp_path / "nested.dclg"
    path.write_text(
        """<doclang xmlns="https://www.doclang.ai/ns/v0">
  <text>Early before</text>
  <formula>early</formula>
  <text>Early after</text>
  <list><ldiv/><text>Late item <formula>late</formula></text></list>
  <text>Late after</text>
</doclang>
""",
        encoding="utf-8",
    )

    assert main(["show", str(path), "//formula", "-C", "1", "-n"]) == 0
    output = capsys.readouterr().out
    assert "/list[1]/ldiv[1]-- Late item $late$" in output
    assert "/list[1]/text[1]/formula[1]:$$late$$" in output
    assert output.count("late") == 2
    assert "/text[2]-Early after" in output
    assert "/text[3]-Late after" in output

    assert main(["show", str(path), "/list[1]/text[1]/formula[1]", "-n"]) == 0
    assert capsys.readouterr().out == "/list[1]/text[1]/formula[1]:$$late$$\n"

    assert main(["show", str(path), "/list[1]/text[1]/formula[1]", "-C", "0", "-n"]) == 0
    assert capsys.readouterr().out == ("/list[1]/ldiv[1]-- Late item $late$\n/list[1]/text[1]/formula[1]:$$late$$\n")

    assert main(["grep", "late", str(path), "--type", "formula", "-n"]) == 0
    assert capsys.readouterr().out == "/list[1]/ldiv[1]:- Late item $late$\n"

    assert main(["grep", "late", str(path), "--all", "-n"]) == 0
    assert capsys.readouterr().out == "/list[1]/ldiv[1]:- Late item $late$\n"

    assert main(["show", str(path), "/list[1]", "-C", "1", "-n"]) == 0
    output = capsys.readouterr().out
    assert "/text[2]-Early after" in output
    assert "/list[1]:- Late item $late$" in output
    assert "/text[3]-Late after" in output


def test_text_views_use_core_serializer_modes_without_trimming(tmp_path: Path, capsys) -> None:
    path = tmp_path / "projections.dclg"
    path.write_text(PROJECTION_DOCUMENT, encoding="utf-8")

    assert main(["show", str(path), "/code[1]", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["text"] == "  SELECT 1;\n  "

    assert main(["grep", "-F", "Chart summary", str(path), "--type", "picture", "--view", "visible", "-q"]) == 1
    assert main(["grep", "-F", "Chart summary", str(path), "--type", "picture", "--view", "metadata", "-q"]) == 0

    assert (
        main(["grep", "-F", "Chart summary", str(path), "--type", "picture", "--view", "metadata", "--format", "json"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)[0]["text"] == "Chart summary"

    assert (
        main(["grep", "-F", "Chart summary", str(path), "--type", "picture", "--view", "all", "--format", "json"]) == 0
    )
    assert json.loads(capsys.readouterr().out)[0]["text"] == "Chart caption\n\nChart summary"


def test_archive_input_and_grep_exit_codes(tmp_path: Path, capsys) -> None:
    archive = tmp_path / "report.dclx"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("document.xml", DOCUMENT)

    assert main(["grep", "-F", "Termination notice", str(archive), "-q"]) == 0
    assert main(["grep", "Outside", str(archive), "--page", "2", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["pages"] == [2]
    assert main(["grep", "missing", str(archive), "-q"]) == 1
    assert main(["grep", "[", str(archive)]) == 2
    assert "invalid regular expression" in capsys.readouterr().err

    prefixed = tmp_path / "prefixed.xml"
    prefixed.write_text(
        '<d:doclang xmlns:d="https://www.doclang.ai/ns/v0"><d:text>prefixed</d:text></d:doclang>',
        encoding="utf-8",
    )
    assert main(["grep", "prefixed", str(prefixed), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["xpaths"] == ["/d:doclang/d:text[1]"]


def test_typer_help_version_and_validation(tmp_path: Path, capsys) -> None:
    path = _document(tmp_path)

    assert main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "inspect" in help_text
    assert "grep" in help_text
    assert "list" in help_text
    # the version tracks docling-core, so the help is the only maturity signal
    assert "EXPERIMENTAL" in help_text

    assert main(["grep", "--help"]) == 0
    assert "--class" not in capsys.readouterr().out

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__

    assert main(["grep", "Report", str(path), "--format", "yaml"]) == 2
    assert "Invalid value" in capsys.readouterr().err


def test_list_enumerates_units_without_a_pattern(tmp_path: Path, capsys) -> None:
    path = _document(tmp_path)

    assert main(["list", str(path), "--all", "--format", "json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert main(["grep", "", str(path), "--all", "--format", "json"]) == 0
    grepped = json.loads(capsys.readouterr().out)
    assert [record["xpaths"] for record in listed] == [record["xpaths"] for record in grepped]
    assert all("matches" not in record for record in listed)

    empty_patterns = tmp_path / "empty-patterns"
    empty_patterns.touch()
    assert main(["grep", "-f", str(empty_patterns), str(path), "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out) == []

    assert main(["list", str(path), "--offset", "1", "--limit", "2", "--format", "json"]) == 0
    window = json.loads(capsys.readouterr().out)
    assert [record["xpaths"] for record in window] == [record["xpaths"] for record in listed[1:3]]
    assert main(["list", str(path), "--offset", "1", "--limit", "2", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == window

    for bound in (["--limit", "0"], ["--offset", "999"]):
        assert main(["list", str(path), *bound, "--format", "json"]) == 1
        assert json.loads(capsys.readouterr().out) == []

    assert main(["list", str(path), "--type", "table_cell", "--page", "2", "-n"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "/table[1]/ched[1]:Metric",
        "/table[1]/ched[2]:Value",
        "/table[1]/rhed[1]:Margin",
        "/table[1]/fcel[1]:48.2%",
    ]

    assert main(["list", str(path), "--type", "formula", "--page", "2", "-q"]) == 1

    # pattern-specific options belong to grep only
    for option in (["-i"], ["-F"], ["-C", "1"], ["-e", "x"], ["--context-scope", "section"]):
        assert main(["list", str(path), *option]) == 2

    # a bare pattern is a usage error now that grep is explicit
    assert main(["Revenue", str(path)]) == 2
    assert "No such command" in capsys.readouterr().err


def test_context_scope_clips_at_its_boundary(tmp_path: Path, capsys) -> None:
    path = tmp_path / "scopes.dclg"
    path.write_text(SCOPE_DOCUMENT, encoding="utf-8")

    def scoped(pattern: str, scope: str, count: str = "2") -> list[str]:
        assert main(["grep", pattern, str(path), "-C", count, "-n", "--context-scope", scope]) == 0
        return capsys.readouterr().out.splitlines()

    # document: context crosses the heading boundary
    assert scoped("Beta", "document") == [
        "/heading[1]-First",
        "/text[1]-Alpha",
        "/text[2]:Beta",
        "/heading[2]-Second",
        "/text[3]-Gamma",
    ]
    # section: context stops at the heading boundary
    assert scoped("Beta", "section") == ["/heading[1]-First", "/text[1]-Alpha", "/text[2]:Beta"]
    # auto: a paragraph resolves to section
    assert scoped("Beta", "auto") == scoped("Beta", "section")

    # container: a table cell sees only cells of the same table
    assert scoped("Margin", "document", "3")[0] == "/list[1]/ldiv[2]-- Item two"
    assert scoped("Margin", "container", "3") == [
        "/table[1]/ched[1]-Metric",
        "/table[1]/ched[2]-Value",
        "/table[1]/rhed[1]:Margin",
        "/table[1]/fcel[1]-48.2%",
    ]
    # auto: a table cell resolves to container
    assert scoped("Margin", "auto", "3") == scoped("Margin", "container", "3")

    # container: a list item sees only siblings of the same list
    assert scoped("Item one", "container", "3") == [
        "/list[1]/ldiv[1]:- Item one",
        "/list[1]/ldiv[2]-- Item two",
    ]

    # page was removed: it severs threads across page breaks for a layout reason
    assert main(["grep", "Beta", str(path), "-C", "1", "--context-scope", "page"]) == 2
    assert "'page' is not one of" in capsys.readouterr().err
