import json
import zipfile
from pathlib import Path

from dlgrep.cli import main

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
  <table>
    <ched/>Metric<ched/>Value<nl/>
    <rhed/>Margin<fcel/>48.2%<nl/>
  </table>
  <page_break/>
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


def _document(tmp_path: Path) -> Path:
    path = tmp_path / "report.dclg"
    path.write_text(DOCUMENT, encoding="utf-8")
    return path


def test_search_returns_source_identity_and_structural_context(tmp_path: Path, capsys) -> None:
    path = _document(tmp_path)

    assert main(["48\\.2", str(path), "-C", "1", "-A", "0", "--format", "json"]) == 0
    result = json.loads(capsys.readouterr().out)[0]

    assert set(result) == {
        "filename",
        "chunk_index",
        "text",
        "raw_text",
        "num_tokens",
        "headings",
        "captions",
        "doc_items",
        "page_numbers",
        "metadata",
    }
    assert result["metadata"]["xpath"] == "/d:doclang/d:table[1]/d:fcel[1]"
    assert result["headings"] == ["Report", "Results"]
    assert result["metadata"]["context"]["row_headers"] == ["Margin"]
    assert result["metadata"]["context"]["column_headers"] == ["Value"]
    assert result["metadata"]["context"]["before"][0]["xpath"].endswith("/d:rhed[1]")
    assert "after" not in result["metadata"]["context"]

    assert main(["beta", str(path), "--format", "json"]) == 0
    threaded = json.loads(capsys.readouterr().out)[0]
    assert threaded["metadata"]["xpaths"] == ["/d:doclang/d:text[2]", "/d:doclang/d:text[3]"]


def test_retrieval_commands_share_xpath_addresses(tmp_path: Path, capsys) -> None:
    path = _document(tmp_path)

    assert main(["show", str(path), "/heading[2]", "--section", "--format", "json"]) == 0
    section = json.loads(capsys.readouterr().out)[0]
    assert "Termination notice" in section["raw_text"]
    assert "Outside the section" not in section["raw_text"]
    assert "| Metric" in section["raw_text"]
    assert "#/tables/0" in section["doc_items"]

    assert main(["show", str(path), "/doclang/heading[2]", "--raw"]) == 0
    assert 'level="2">Results</heading>' in capsys.readouterr().out

    assert main(["outline", str(path), "--format", "json"]) == 0
    outline = json.loads(capsys.readouterr().out)
    assert [heading["xpath"] for heading in outline["headings"]] == [
        "/d:doclang/d:heading[1]",
        "/d:doclang/d:heading[2]",
        "/d:doclang/d:heading[3]",
    ]

    assert main(["select", str(path), "count(/heading)", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["value"] == 3.0
    assert main(["select", str(path), "count(descendant::heading)", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["value"] == 3.0
    assert main(["select", str(path), "/heading[1]", "--format", "text", "--max-chars", "5"]) == 0
    assert capsys.readouterr().out == "Repo…\n"

    assert main(["inspect", str(path), "--format", "json"]) == 0
    inventory = json.loads(capsys.readouterr().out)[0]
    assert set(inventory) == {"document", "input_type", "pages", "semantic_units", "elements", "metadata"}
    assert inventory["semantic_units"] > 0
    assert inventory["elements"]["code"] == 1
    assert "bold" not in inventory["elements"]
    assert "fcel" not in inventory["elements"]
    assert inventory["metadata"] == {}

    assert main(["Revenue", str(path), "--within-xpath", "/heading[2]", "--section", "-q"]) == 0


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

    assert main(["-F", "See this important article for details.", str(path)]) == 0
    output = capsys.readouterr().out
    assert "See this important article for details." in output
    assert "<bold>" not in output
    assert "**" not in output
    assert "Type:" not in output
    assert str(path) not in output

    assert main(["-F", "See this important article for details.", str(path), "-n"]) == 0
    output = capsys.readouterr().out
    assert "/d:doclang/d:text[4]:See this important article for details." in output
    assert "XPath:" not in output

    assert main(["Revenue", str(path), "--max-chars", "8"]) == 0
    output = capsys.readouterr().out
    assert output.rstrip().endswith("Revenue…")
    assert "[truncated]" not in output

    assert main(["show", str(path), "/formula[1]", "-B", "1", "-n"]) == 0
    output = capsys.readouterr().out
    assert "/d:doclang/d:code[1]-public  class Main {}" in output
    assert "/d:doclang/d:formula[1]:$$E = mc^2$$" in output
    assert "\n--\n" not in output
    assert "XPath:" not in output

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("dlgrep.cli.sys.stdout.isatty", lambda: True)
    assert main(["Revenue", str(path)]) == 0
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

    assert main(["Information", str(path), "-C", "1", "-n"]) == 0
    output = capsys.readouterr().out

    assert output.count("\n--\n") == 1
    assert output.count("Shared context") == 1
    assert "/d:doclang/d:text[2]:Information one" in output
    assert "/d:doclang/d:text[4]:Information two" in output
    assert "/d:doclang/d:text[9]:Information three" in output
    assert "/d:doclang/d:text[2]-" not in output
    assert "/d:doclang/d:text[4]-" not in output
    assert "Omitted one" not in output
    assert "Omitted two" not in output


def test_text_views_use_core_serializer_modes_without_trimming(tmp_path: Path, capsys) -> None:
    path = tmp_path / "projections.dclg"
    path.write_text(PROJECTION_DOCUMENT, encoding="utf-8")

    assert main(["show", str(path), "/code[1]", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["raw_text"] == "  SELECT 1;\n  "

    assert main(["-F", "Chart summary", str(path), "--type", "picture", "--view", "visible", "-q"]) == 1
    assert main(["-F", "Chart summary", str(path), "--type", "picture", "--view", "metadata", "-q"]) == 0

    assert main(["-F", "Chart summary", str(path), "--type", "picture", "--view", "metadata", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["raw_text"] == "Chart summary"

    assert main(["-F", "Chart summary", str(path), "--type", "picture", "--view", "all", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["raw_text"] == "Chart caption\n\nChart summary"


def test_archive_input_and_grep_exit_codes(tmp_path: Path, capsys) -> None:
    archive = tmp_path / "report.dclx"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("document.xml", DOCUMENT)

    assert main(["-F", "Termination notice", str(archive), "-q"]) == 0
    assert main(["Outside", str(archive), "--page", "2", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["page_numbers"] == [2]
    assert main(["missing", str(archive), "-q"]) == 1
    assert main(["[", str(archive)]) == 2
    assert "invalid regular expression" in capsys.readouterr().err

    prefixed = tmp_path / "prefixed.xml"
    prefixed.write_text(
        '<d:doclang xmlns:d="https://www.doclang.ai/ns/v0"><d:text>prefixed</d:text></d:doclang>',
        encoding="utf-8",
    )
    assert main(["prefixed", str(prefixed), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["metadata"]["xpath"] == "/d:doclang/d:text[1]"


def test_typer_help_version_and_validation(tmp_path: Path, capsys) -> None:
    path = _document(tmp_path)

    assert main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "inspect" in help_text
    assert "search" in help_text

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.0.0"

    assert main(["Report", str(path), "--format", "yaml"]) == 2
    assert "Invalid value" in capsys.readouterr().err
