"""Unit tests for appflowy_mcp.fields (database field editing over CRDT)."""

from __future__ import annotations

import json

import pytest

from appflowy_mcp import fields
from tests.conftest import build_fields_doc

PRIMARY = {"name": "Name", "ty": 0, "is_primary": True}


def database(doc):
    return doc.get("data", type=fields.blocks.YMap)["database"]


def doc_with_type_field(views=None):
    return build_fields_doc(
        {"n": PRIMARY, "t": {"name": "Type", "ty": 3}},
        views
        if views is not None
        else {
            "V": {
                "field_orders": [{"id": "n"}, {"id": "t"}],
                "field_settings": {"n": {"w": 1}, "t": {"w": 1}},
                "filters": [{"id": "f1", "field_id": "t"}, {"id": "f2", "field_id": "n"}],
            }
        },
    )


def options_of(doc, field_id, key="3"):
    raw = database(doc)["fields"][field_id]["type_option"][key]["content"]
    return json.loads(raw)["options"]


# -- rename_field ----------------------------------------------------------
def test_rename_field_sets_name():
    doc = doc_with_type_field()
    fields.rename_field(doc, "t", "Kind")
    assert database(doc)["fields"]["t"]["name"] == "Kind"


def test_rename_field_unknown_raises():
    doc = doc_with_type_field()
    with pytest.raises(ValueError, match="not found"):
        fields.rename_field(doc, "missing", "x")


# -- delete_field ----------------------------------------------------------
def test_delete_field_removes_definition():
    doc = doc_with_type_field()
    fields.delete_field(doc, "t")
    assert "t" not in database(doc)["fields"].keys()


def test_delete_field_strips_field_orders():
    doc = doc_with_type_field()
    fields.delete_field(doc, "t")
    orders = database(doc)["views"]["V"]["field_orders"]
    assert [o["id"] for o in orders] == ["n"]


def test_delete_field_strips_field_settings():
    doc = doc_with_type_field()
    fields.delete_field(doc, "t")
    assert list(database(doc)["views"]["V"]["field_settings"].keys()) == ["n"]


def test_delete_field_strips_filters_on_that_field():
    doc = doc_with_type_field()
    fields.delete_field(doc, "t")
    filters = database(doc)["views"]["V"]["filters"]
    assert [f["id"] for f in filters] == ["f2"]


def test_delete_field_tolerates_view_without_field_keys():
    doc = doc_with_type_field(views={"V": {}})
    fields.delete_field(doc, "t")
    assert "t" not in database(doc)["fields"].keys()


def test_delete_field_tolerates_settings_without_that_field():
    doc = doc_with_type_field(views={"V": {"field_settings": {"n": {"w": 1}}}})
    fields.delete_field(doc, "t")
    assert list(database(doc)["views"]["V"]["field_settings"].keys()) == ["n"]


def test_delete_field_primary_raises():
    doc = doc_with_type_field()
    with pytest.raises(ValueError, match="primary"):
        fields.delete_field(doc, "n")


# -- add_select_options ----------------------------------------------------
def test_add_select_options_appends_options():
    doc = doc_with_type_field()
    fields.add_select_options(doc, "t", ["Revision", "Exam"])
    assert [o["name"] for o in options_of(doc, "t")] == ["Revision", "Exam"]


def test_add_select_options_returns_added():
    doc = doc_with_type_field()
    added = fields.add_select_options(doc, "t", ["Revision"])
    assert added[0]["name"] == "Revision"


def test_add_select_options_assigns_colors_in_order():
    doc = doc_with_type_field()
    added = fields.add_select_options(doc, "t", ["a", "b"])
    assert [o["color"] for o in added] == ["Purple", "Pink"]


def test_add_select_options_skips_existing_names():
    doc = doc_with_type_field()
    fields.add_select_options(doc, "t", ["Revision"])
    added = fields.add_select_options(doc, "t", ["Revision", "Exam"])
    assert [o["name"] for o in added] == ["Exam"]


def test_add_select_options_keeps_existing_content():
    content = json.dumps({"options": [{"id": "abcd", "name": "Old", "color": "Blue"}]})
    doc = build_fields_doc(
        {"t": {"name": "Tags", "ty": 4, "type_option": {"4": {"content": content}}}},
        {},
    )
    fields.add_select_options(doc, "t", ["New"])
    assert [o["name"] for o in options_of(doc, "t", "4")] == ["Old", "New"]


def test_add_select_options_fills_type_option_without_content():
    doc = build_fields_doc({"t": {"name": "S", "ty": 3, "type_option": {"3": {}}}}, {})
    fields.add_select_options(doc, "t", ["x"])
    assert [o["name"] for o in options_of(doc, "t")] == ["x"]


def test_add_select_options_rejects_non_select_field():
    doc = doc_with_type_field()
    with pytest.raises(ValueError, match="not a select field"):
        fields.add_select_options(doc, "n", ["x"])


def test_new_option_id_retries_on_collision(monkeypatch):
    picks = iter("aaaa" + "bbbb")
    monkeypatch.setattr(fields.secrets, "choice", lambda _alphabet: next(picks))
    assert fields._new_option_id({"aaaa"}) == "bbbb"
