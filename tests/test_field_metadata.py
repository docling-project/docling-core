"""Document-level AcroForm field metadata on FieldItem."""

import json

import pytest

from docling_core.types.doc import DoclingDocument, FieldControl
from docling_core.types.doc.items.form import FieldItem


def _doc_with_field(**kwargs) -> tuple[DoclingDocument, FieldItem]:
    doc = DoclingDocument(name="form")
    region = doc.add_field_region()
    return doc, doc.add_field_item(parent=region, **kwargs)


def test_a_field_carries_no_metadata_until_it_is_given_some():
    _, item = _doc_with_field()
    assert item.control is None
    assert item.description is None
    assert item.required is False
    assert item.options == []


def test_the_helper_populates_every_slot():
    _, item = _doc_with_field(
        control=FieldControl.CHOICE,
        description="Country of residence",
        required=True,
        options=["US", "DE"],
    )
    assert item.control is FieldControl.CHOICE
    assert item.description == "Country of residence"
    assert item.required is True
    assert item.options == ["US", "DE"]


def test_options_are_copied_so_the_caller_cannot_mutate_the_document():
    caller_list = ["US", "DE"]
    _, item = _doc_with_field(control=FieldControl.CHOICE, options=caller_list)
    caller_list.append("FR")
    assert item.options == ["US", "DE"]


def test_metadata_survives_a_json_round_trip():
    doc, _ = _doc_with_field(
        control=FieldControl.RADIO,
        description="Delivery speed",
        required=True,
        options=["standard", "express"],
    )
    restored = DoclingDocument.model_validate(json.loads(doc.model_dump_json(by_alias=True)))
    (item,) = restored.field_items
    assert item.control is FieldControl.RADIO
    assert item.description == "Delivery speed"
    assert item.required is True
    assert item.options == ["standard", "express"]


def test_unset_optionals_stay_off_the_wire():
    """`export_to_dict` excludes None, so an unannotated field stays compact."""
    doc, _ = _doc_with_field()
    (item,) = doc.export_to_dict()["field_items"]
    assert "control" not in item
    assert "description" not in item
    # These two have concrete defaults rather than None, so they are always present.
    assert item["required"] is False
    assert item["options"] == []


@pytest.mark.parametrize("control", list(FieldControl))
def test_every_control_round_trips_as_its_string_value(control: FieldControl):
    doc, _ = _doc_with_field(control=control)
    (item,) = doc.export_to_dict()["field_items"]
    assert item["control"] == control.value
    restored = DoclingDocument.model_validate(json.loads(doc.model_dump_json(by_alias=True)))
    assert restored.field_items[0].control is control
