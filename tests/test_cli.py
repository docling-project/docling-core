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


def _document(tmp_path: Path) -> Path:
    path = tmp_path / "report.dclg"
    path.write_text(DOCUMENT, encoding="utf-8")
    return path


def test_search_returns_source_identity_and_structural_context(tmp_path: Path, capsys) -> None:
    path = _document(tmp_path)

    assert main(["48\\.2", str(path), "-C", "1", "-A", "0", "--format", "json"]) == 0
    result = json.loads(capsys.readouterr().out)[0]

    assert result["xpath"] == "/d:doclang/d:table[1]/d:fcel[1]"
    assert result["context"]["headings"] == ["Report", "Results"]
    assert result["context"]["row_headers"] == ["Margin"]
    assert result["context"]["column_headers"] == ["Value"]
    assert result["context"]["before"][0]["xpath"].endswith("/d:rhed[1]")
    assert result["context"]["after"] == []

    assert main(["beta", str(path), "--format", "json"]) == 0
    threaded = json.loads(capsys.readouterr().out)[0]
    assert threaded["cardinality"] == 2
    assert threaded["xpaths"] == ["/d:doclang/d:text[2]", "/d:doclang/d:text[3]"]


def test_retrieval_commands_share_xpath_addresses(tmp_path: Path, capsys) -> None:
    path = _document(tmp_path)

    assert main(["show", str(path), "/d:doclang/d:heading[2]", "--section", "--format", "json"]) == 0
    section = json.loads(capsys.readouterr().out)[0]
    assert "Termination notice" in section["text"]
    assert "Outside the section" not in section["text"]

    assert main(["show", str(path), "/d:doclang/d:heading[2]", "--raw"]) == 0
    assert 'level="2">Results</heading>' in capsys.readouterr().out

    assert main(["outline", str(path), "--format", "json"]) == 0
    outline = json.loads(capsys.readouterr().out)
    assert [heading["xpath"] for heading in outline["headings"]] == [
        "/d:doclang/d:heading[1]",
        "/d:doclang/d:heading[2]",
        "/d:doclang/d:heading[3]",
    ]

    assert main(["select", str(path), "count(/d:doclang/d:heading)", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["value"] == 3.0

    assert main(["inspect", str(path), "--format", "json"]) == 0
    inventory = json.loads(capsys.readouterr().out)[0]
    assert inventory["source_map"]["unbound_semantic_units"] == 0


def test_archive_input_and_grep_exit_codes(tmp_path: Path, capsys) -> None:
    archive = tmp_path / "report.dclx"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("document.xml", DOCUMENT)

    assert main(["-F", "Termination notice", str(archive), "-q"]) == 0
    assert main(["Outside", str(archive), "--page", "2", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["pages"] == [2]
    assert main(["missing", str(archive), "-q"]) == 1
    assert main(["[", str(archive)]) == 2
    assert "invalid regular expression" in capsys.readouterr().err

    prefixed = tmp_path / "prefixed.xml"
    prefixed.write_text(
        '<d:doclang xmlns:d="https://www.doclang.ai/ns/v0"><d:text>prefixed</d:text></d:doclang>',
        encoding="utf-8",
    )
    assert main(["prefixed", str(prefixed), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["xpath"] == "/d:doclang/d:text[1]"


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
