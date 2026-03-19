"""Test the pydantic models in package utils."""

import json
from pathlib import Path

from pydantic import Field
from requests import Response

from docling_core.utils.alias import AliasModel
from docling_core.utils.file import resolve_source_to_path, resolve_source_to_stream

from .test_data_gen_flag import GEN_TEST_DATA


def assert_or_generate_ground_truth(
    result_text: str,
    exp_path: Path,
    error_msg: str = "Serialized text should match expected output",
    is_json: bool = False,
) -> None:
    """Helper function to either generate ground truth or assert against it.

    This function supports the GEN_TEST_DATA pattern for test maintenance:
    - When GEN_TEST_DATA=1, it writes the result to the ground truth file
    - Otherwise, it reads the ground truth file and asserts equality

    Args:
        result_text: The serialized result text to compare or save
        exp_path: Path to the expected/ground truth file
        error_msg: Error message to display if assertion fails
        is_json: If True, compare as JSON objects instead of raw strings
    """
    if GEN_TEST_DATA:
        with open(exp_path, "w", encoding="utf-8") as f:
            f.write(result_text)
    else:
        with open(exp_path, encoding="utf-8") as f:
            if is_json:
                expected = json.load(f)
                actual = json.loads(result_text)
                assert actual == expected, error_msg
            else:
                expected = f.read()
                assert result_text == expected, error_msg


def assert_or_generate_json_ground_truth(
    result_data: dict | list,
    exp_path: Path | str,
    error_msg: str = "JSON data should match expected output",
) -> None:
    """Helper function to either generate JSON ground truth or assert against it.

    This is a convenience wrapper for JSON data that handles serialization with
    proper formatting (indent=4, trailing newline).

    Args:
        result_data: The data structure (dict or list) to compare or save
        exp_path: Path to the expected/ground truth JSON file
        error_msg: Error message to display if assertion fails
    """
    exp_path = Path(exp_path) if isinstance(exp_path, str) else exp_path

    if GEN_TEST_DATA:
        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4)
            f.write("\n")
    else:
        with open(exp_path, encoding="utf-8") as f:
            expected = json.load(f)
        assert result_data == expected, error_msg


def test_alias_model() -> None:
    """Test the functionality of AliasModel."""

    class AliasModelChild(AliasModel):
        foo: str = Field(alias="boo")

    data = {"foo": "lorem ipsum"}
    data_alias = {"boo": "lorem ipsum"}

    # data validated from dict, JSON, and constructor can use field names or aliases

    AliasModelChild.model_validate(data_alias)
    AliasModelChild.model_validate(data)

    AliasModelChild.model_validate_json(json.dumps(data_alias))
    AliasModelChild.model_validate_json(json.dumps(data))

    AliasModelChild(boo="lorem ipsum")  # type: ignore[call-arg]
    AliasModelChild(foo="lorem ipsum")

    # children classes will also inherite the populate_by_name

    class AliasModelGrandChild(AliasModelChild):
        var: int

    AliasModelGrandChild(boo="lorem ipsum", var=3)  # type: ignore[call-arg]
    AliasModelGrandChild(foo="lorem ipsum", var=3)

    # serialized data will always use aliases

    obj = AliasModelChild.model_validate(data_alias)
    assert obj.model_dump() == data_alias
    assert obj.model_dump() != data

    assert obj.model_dump_json() == json.dumps(data_alias, separators=(",", ":"))
    assert obj.model_dump_json() != json.dumps(data, separators=(",", ":"))


def test_resolve_source_to_path_url_wout_path(monkeypatch):
    expected_str = "foo"
    expected_bytes = bytes(expected_str, "utf-8")

    def get_dummy_response(*args, **kwargs):
        r = Response()
        r.status_code = 200
        r._content = expected_bytes
        return r

    monkeypatch.setattr("requests.get", get_dummy_response)
    monkeypatch.setattr(
        "requests.models.Response.iter_content",
        lambda *args, **kwargs: [expected_bytes],
    )
    path = resolve_source_to_path("https://pypi.org")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert text == expected_str


def test_resolve_source_to_stream_url_wout_path(monkeypatch):
    expected_str = "foo"
    expected_bytes = bytes(expected_str, "utf-8")

    def get_dummy_response(*args, **kwargs):
        r = Response()
        r.status_code = 200
        r._content = expected_bytes
        return r

    monkeypatch.setattr("requests.get", get_dummy_response)
    monkeypatch.setattr(
        "requests.models.Response.iter_content",
        lambda *args, **kwargs: [expected_bytes],
    )
    doc_stream = resolve_source_to_stream("https://pypi.org")
    assert doc_stream.name == "file"

    text = doc_stream.stream.read().decode("utf8")
    assert text == expected_str
