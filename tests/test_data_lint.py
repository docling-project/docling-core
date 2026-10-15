"""Tests of the private data lint in ``scripts/`` (``_data_lint.py``)."""

import sys
from pathlib import Path

import pytest

from docling_core.types.doc import DocItemLabel, DoclingDocument

_SCRIPTS = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def checks():
    sys.path.insert(0, str(_SCRIPTS))
    import _data_lint  # type: ignore[import-not-found]

    return _data_lint


def _status(results):
    return {r.check: r.status for r in results}


def test_check_json_flags_empty_list_wrapper(checks, tmp_path):
    doc = DoclingDocument(name="")
    lst = doc.add_list_group()
    doc.add_list_item(text="a", marker="1.", parent=lst)
    wrapper = doc.add_list_item(text="", marker="", parent=lst)
    sub = doc.add_list_group(parent=wrapper)
    doc.add_list_item(text="b", marker="a.", parent=sub)
    path = tmp_path / "doc.json"
    doc.save_as_json(path)

    status = _status(checks.check_json(path))
    assert status["json.load"] == "ok"
    assert status["json.rules"] == "ok"
    assert status["json.empty-list-wrapper"] == "warn"


def test_check_pair_roundtrip(checks, tmp_path):
    doc = DoclingDocument(name="")
    doc.add_text(label=DocItemLabel.TEXT, text="hello")
    json_path, dclx_path = tmp_path / "doc.json", tmp_path / "doc.dclx"
    doc.save_as_json(json_path)
    doc.save_as_doclang_archive(dclx_path)

    status = _status(checks.check_pair(json_path, dclx_path))
    assert status["dclx/deserialize"] == "ok"
    assert status["dclx/roundtrip"] == "ok"
    assert status["pair.json-to-dclx"] == "ok"
    assert status["pair.dclx-to-json"] == "ok"
    assert "fail" not in status.values()


def test_check_dclx_reports_a_failing_roundtrip(checks, tmp_path):
    # a field region in the furniture layer loses its layer on deserialization (known gap)
    from docling_core.types.doc.document import ContentLayer

    doc = DoclingDocument(name="")
    region = doc.add_field_region()
    region.content_layer = ContentLayer.FURNITURE
    item = doc.add_field_item(parent=region)
    item.content_layer = ContentLayer.FURNITURE
    doc.add_field_key(text="k", parent=item, content_layer=ContentLayer.FURNITURE)
    doc.add_field_value(text="v", parent=item, content_layer=ContentLayer.FURNITURE)
    path = tmp_path / "doc.dclx"
    doc.save_as_doclang_archive(path)

    status = _status(checks.check_dclx(path))
    assert status["dclx.deserialize"] == "ok"
    assert status["dclx.roundtrip"] == "fail"

    # the pair check sees the same loss from the JSON side: the layers differ after DCLX -> JSON
    doc.save_as_json(tmp_path / "doc.json")
    pair = {r.check: r for r in checks.check_pair(tmp_path / "doc.json", path)}
    assert pair["pair.json-to-dclx"].status == "ok"
    assert pair["pair.dclx-to-json"].status == "warn"  # a diagnostic, not a failure
    assert any("field_region [furniture]" in line for line in pair["pair.dclx-to-json"].details)


def test_cli_by_extension_and_check_pair(checks, tmp_path, capsys):
    doc = DoclingDocument(name="")
    doc.add_text(label=DocItemLabel.TEXT, text="hello")
    doc.save_as_json(tmp_path / "doc.json")
    doc.save_as_doclang_archive(tmp_path / "doc.dclx")

    assert checks.main([str(tmp_path / "doc.json")]) == 0
    assert checks.main([str(tmp_path / "doc.dclx"), "--pair", "-q"]) == 0
    assert "RESULT: OK" in capsys.readouterr().out
    checks.main([str(tmp_path / "doc.json"), "--pair"])
    assert "pair.json-to-dclx" in capsys.readouterr().out

    (tmp_path / "doc.json").rename(tmp_path / "other.json")
    with pytest.raises(SystemExit, match="counterpart not found"):
        checks.main([str(tmp_path / "doc.dclx"), "--pair"])
    with pytest.raises(SystemExit, match="unsupported extension"):
        checks.main([str(tmp_path / "x.txt")])


def test_cli_result_line_reports_warnings(checks, tmp_path, capsys):
    doc = DoclingDocument(name="")
    lst = doc.add_list_group()
    doc.add_list_item(text="a", marker="1.", parent=lst)
    wrapper = doc.add_list_item(text="", marker="", parent=lst)
    doc.add_list_item(text="b", marker="a.", parent=doc.add_list_group(parent=wrapper))
    doc.save_as_json(tmp_path / "doc.json")

    assert checks.main([str(tmp_path / "doc.json")]) == 0  # warnings do not change the exit code
    assert "RESULT: WARN (1 warning(s))" in capsys.readouterr().out


def test_selection_and_list_checks(checks, tmp_path, capsys):
    sel = checks.Selection(only=["dclx"], skip=["dclx.schematron"])
    assert sel.on("dclx.roundtrip") and sel.on("dclx.xsd")
    assert not sel.on("dclx.schematron") and not sel.on("json.load") and not sel.on("pair.json-to-dclx")
    assert checks.Selection(only=["json.empty-list-wrapper"]).on("json.empty-list-wrapper")
    assert not checks.Selection(only=["json.empty-list-wrapper"]).on("json.load")
    assert checks.Selection(only=["json"]).on("json.empty-list-wrapper")  # a group prefix selects its checks
    with pytest.raises(SystemExit, match=r"--only dclx\.roundtrip selects nothing"):
        checks.Selection(only=["dclx.roundtrip"], skip=["dclx"])
    with pytest.raises(SystemExit, match="--only json selects nothing"):
        checks.Selection(only=["json"], skip=["json"])
    with pytest.raises(SystemExit, match="unknown check: nope"):
        checks.Selection(only=["nope"])

    assert checks.main(["--list-checks"]) == 0
    out = capsys.readouterr().out
    assert all(slug in out for slug in checks.CHECKS)

    doc = DoclingDocument(name="")
    doc.add_text(label=DocItemLabel.TEXT, text="hello")
    doc.save_as_json(tmp_path / "doc.json")
    doc.save_as_doclang_archive(tmp_path / "doc.dclx")
    capsys.readouterr()
    checks.main([str(tmp_path / "doc.dclx"), "--pair", "--only", "dclx.roundtrip,pair"])
    printed = [line.split()[1] for line in capsys.readouterr().out.splitlines() if line[:2] in ("OK", "FA")]
    assert printed == ["dclx/roundtrip", "pair.json-to-dclx", "pair.dclx-to-json"]


def test_diff_normalizes_indentation_but_not_content_whitespace(checks):
    # indentation and blank lines outside <content> do not matter ...
    assert checks._diff("<text>\n  a\n</text>", "<text>\na\n\n</text>") == []
    # ... but <content> preserves whitespace, so any difference inside it is reported
    assert checks._diff("<text><content> a\nb</content></text>", "<text><content>a\nb</content></text>")
    assert checks._diff("<text><content>a\n b</content></text>", "<text><content>a\nb</content></text>")
    assert checks._diff("<text><content>a\nb </content></text>", "<text><content>a\nb</content></text>")
    assert checks._diff("<text><content>a\nb</content></text>", "<text><content>a\nb</content></text>") == []


def test_pair_accepts_dclg_markup(checks, tmp_path, capsys):
    doc = DoclingDocument(name="")
    doc.add_text(label=DocItemLabel.TEXT, text="hello")
    doc.save_as_json(tmp_path / "doc.json")
    xml = checks.serialize(doc, (512, 512))
    (tmp_path / "doc.dclg").write_text(xml)

    assert checks.main([str(tmp_path / "doc.dclg"), "--pair", "-q"]) == 0
    assert checks.main([str(tmp_path / "doc.json"), "--pair", "-q"]) == 0  # found via .dclg
    (tmp_path / "doc.dclg").rename(tmp_path / "doc.dclg.xml")
    assert checks.main([str(tmp_path / "doc.dclg.xml"), "--pair", "-q"]) == 0  # X.dclg.xml -> X.json
    capsys.readouterr()
    (tmp_path / "doc.dclg.xml").rename(tmp_path / "gone.xml")
    with pytest.raises(SystemExit, match="counterpart not found"):
        checks.main([str(tmp_path / "doc.json"), "--pair"])


# ---- failure paths -------------------------------------------------------------------------------------------------
def _simple_doc(text: str = "hello") -> DoclingDocument:
    doc = DoclingDocument(name="")
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    return doc


def test_json_load_failure(checks, tmp_path):
    (tmp_path / "bad.json").write_text('{"schema_name": "DoclingDocument", "texts": "not a list"}')
    results = checks.check_json(tmp_path / "bad.json")
    assert [(r.check, r.status) for r in results] == [("json.load", "fail")]

    (tmp_path / "broken.json").write_text("{not json")
    assert [(r.check, r.status) for r in checks.check_json(tmp_path / "broken.json")] == [("json.load", "fail")]


def test_json_rules_failure_on_legacy_key_value_item(checks, tmp_path):
    from docling_core.types.doc.document import GraphData

    doc = _simple_doc()
    doc.add_key_values(graph=GraphData(cells=[], links=[]))
    doc.save_as_json(tmp_path / "doc.json")

    status = {r.check: r for r in checks.check_json(tmp_path / "doc.json")}
    assert status["json.rules"].status == "fail"
    assert any("is to be migrated to a field region" in line for line in status["json.rules"].details)


def test_json_serialize_failure(checks, tmp_path, monkeypatch):
    _simple_doc().save_as_json(tmp_path / "doc.json")

    def boom(doc, resolution):
        raise RuntimeError("cannot serialize")

    monkeypatch.setattr(checks, "serialize", boom)
    status = {
        r.check: r for r in checks.check_json(tmp_path / "doc.json", sel=checks.Selection(only=["json.serialize"]))
    }
    assert status["json.serialize"].status == "fail"
    assert "cannot serialize" in status["json.serialize"].message


def test_dclx_xsd_failure(checks, tmp_path):
    (tmp_path / "bad.xml").write_text('<doclang version="0.7"><not_an_element/></doclang>')
    status = {r.check: r for r in checks.check_dclx(tmp_path / "bad.xml", sel=checks.Selection(only=["dclx.xsd"]))}
    assert status["dclx.xsd"].status == "fail"
    assert status["dclx.xsd"].details


def test_dclx_schematron_fails_without_the_backend(checks, tmp_path, monkeypatch):
    import doclang.validation

    _simple_doc().save_as_doclang_archive(tmp_path / "doc.dclx")
    real = doclang.validation.validate

    def validate(path, **kwargs):
        if kwargs.get("schematron_only"):
            raise ImportError("no saxon")
        return real(path, **kwargs)

    monkeypatch.setattr(doclang.validation, "validate", validate)
    status = {r.check: r for r in checks.check_dclx(tmp_path / "doc.dclx", sel=checks.Selection(only=["dclx"]))}
    assert status["dclx.schematron"].status == "fail"
    assert "--skip dclx.schematron" in status["dclx.schematron"].message
    assert status["dclx.xsd"].status == "ok"

    # skipping it explicitly runs without it
    skipped = {
        r.check: r.status
        for r in checks.check_dclx(tmp_path / "doc.dclx", sel=checks.Selection(skip=["dclx.schematron"]))
    }
    assert "dclx.schematron" not in skipped and "fail" not in skipped.values()


def test_dclx_deserialize_failure(checks, tmp_path):
    (tmp_path / "broken.xml").write_text("<doclang><text>")
    status = {r.check: r.status for r in checks.check_dclx(tmp_path / "broken.xml")}
    assert status["dclx.deserialize"] == "fail"
    assert "dclx.roundtrip" not in status  # nothing to round trip


def test_dclx_rules_failure_when_the_deserialized_document_fails_model_validation(checks, tmp_path, monkeypatch):
    # the deserializer builds documents through the API, so the lint re-validates the result through the model
    _simple_doc().save_as_doclang_archive(tmp_path / "doc.dclx")

    def invalid(*args, **kwargs):
        raise ValueError("not a valid document")

    monkeypatch.setattr(checks.DoclingDocument, "model_validate", invalid)
    status = {r.check: r for r in checks.check_dclx(tmp_path / "doc.dclx")}
    assert status["dclx.rules"].status == "fail"
    assert any("not a valid document" in line for line in status["dclx.rules"].details)


def test_pair_json_to_dclx_failure_when_the_files_do_not_match(checks, tmp_path):
    _simple_doc("one").save_as_json(tmp_path / "doc.json")
    _simple_doc("two").save_as_doclang_archive(tmp_path / "doc.dclx")

    status = {r.check: r for r in checks.check_pair(tmp_path / "doc.json", tmp_path / "doc.dclx")}
    assert status["pair.json-to-dclx"].status == "fail"
    assert status["pair.dclx-to-json"].status == "warn"
    assert status["dclx/roundtrip"].status == "ok"  # each side is fine on its own


def test_cli_output_json_and_quiet(checks, tmp_path, capsys):
    import json

    _simple_doc("one").save_as_json(tmp_path / "doc.json")
    _simple_doc("two").save_as_doclang_archive(tmp_path / "doc.dclx")

    assert checks.main([str(tmp_path / "doc.json"), "--pair", "--output", "json"]) == 1
    data = json.loads(capsys.readouterr().out)
    (label,) = data
    by_check = {r["check"]: r for r in data[label]}
    assert by_check["pair.json-to-dclx"]["status"] == "fail" and by_check["pair.json-to-dclx"]["details"]

    assert checks.main([str(tmp_path / "doc.json"), "--pair", "-q"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  pair.json-to-dclx" in out and "OK  " not in out and "WARN" not in out
    assert out.rstrip().endswith("RESULT: FAILED")


def test_json_load_failure_on_model_validation_error(checks, tmp_path):
    import json

    doc = _simple_doc()
    doc.add_text(label=DocItemLabel.TEXT, text="again")
    data = doc.model_dump(mode="json")
    data["texts"][1]["self_ref"] = data["texts"][0]["self_ref"]  # a duplicate ref is rejected by the model
    (tmp_path / "doc.json").write_text(json.dumps(data))

    results = checks.check_json(tmp_path / "doc.json")
    assert [(r.check, r.status) for r in results] == [("json.load", "fail")]
    assert "Duplicate ref" in results[0].message


def test_cli_report_modes(checks, tmp_path, capsys):
    import json

    from docling_core.types.doc.document import ContentLayer

    _simple_doc().save_as_json(tmp_path / "good.json")
    _simple_doc().save_as_doclang_archive(tmp_path / "good.dclx")
    bad = DoclingDocument(name="")
    region = bad.add_field_region()
    region.content_layer = ContentLayer.FURNITURE  # lost on deserialization: the round trip fails
    bad.save_as_json(tmp_path / "bad.json")
    bad.save_as_doclang_archive(tmp_path / "bad.dclx")
    files = [str(tmp_path / "good.dclx"), str(tmp_path / "bad.dclx")]

    assert checks.main([*files, "--pair", "--report", "checks", "--only", "dclx.roundtrip"]) == 1
    out = capsys.readouterr().out
    assert "PER CHECK (2 file(s))" in out
    row = next(line for line in out.splitlines() if line.startswith("dclx.roundtrip"))
    assert row.split()[1:] == ["1", "0", "1"]  # ok, warn, fail
    assert "bad.dclx" not in out  # the files are only listed on request

    checks.main([*files, "--pair", "--report", "checks-with-files", "--only", "dclx.roundtrip"])
    out = capsys.readouterr().out
    assert "FAIL dclx.roundtrip (1):" in out and "bad.json + bad.dclx" in out and "good.dclx" not in out

    checks.main([*files, "--pair", "--report", "checks-with-files", "--only", "dclx.roundtrip", "--output", "json"])
    data = json.loads(capsys.readouterr().out)
    assert data["files"] == 2 and data["checks"]["dclx.roundtrip"] == {"ok": 1, "fail": 1}
    assert data["flagged"]["dclx.roundtrip"] == [{"status": "fail", "file": "bad.json + bad.dclx"}]
