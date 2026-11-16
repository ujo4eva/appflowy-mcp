"""Unit tests for appflowy_mcp.blocks."""

from __future__ import annotations

import json

import pytest
from pycrdt import Array, Doc, Map, Text

from appflowy_mcp import blocks
from tests.conftest import build_collab


def raw_document(block_defs, children_defs, page_id="root"):
    """Build a collab from explicit block + children-map specs.

    Lets tests create shapes the high-level builder cannot, such as dangling
    child references and a page root whose children array is missing.
    """
    doc = Doc()
    document = Map()
    doc.get("data", type=Map)["document"] = document
    bm = Map()
    document["blocks"] = bm
    meta = Map()
    document["meta"] = meta
    meta["text_map"] = Map()
    cm = Map()
    meta["children_map"] = cm
    document["page_id"] = page_id
    for bid, spec in block_defs.items():
        bm[bid] = Map(
            {
                "id": bid,
                "ty": "paragraph",
                "data": "{}",
                "parent": spec.get("parent", ""),
                "children": spec.get("children", ""),
                "external_id": "",
                "external_type": "text",
            }
        )
    for cid, kids in children_defs.items():
        cm[cid] = Array()
        for k in kids:
            cm[cid].append(k)
    return doc.get_update()


def integrated_text(initial=""):
    """A Y.Text wired into a live document, as the editing code requires."""
    doc = Doc()
    tmap = doc.get("m", type=Map)
    tmap["t"] = Text()
    t = tmap["t"]
    if initial:
        t.insert(0, initial)
    return t


def test_new_id_default_length():
    assert len(blocks.new_id()) == 10


def test_new_id_custom_length():
    assert len(blocks.new_id(5)) == 5


def test_new_id_uses_url_safe_alphabet():
    assert set(blocks.new_id(50)) <= set(blocks._ID_ALPHABET)


def test_ykeys_lists_map_keys():
    _doc, document = blocks.load_document(build_collab())
    assert "blocks" in blocks.ykeys(document)


def test_apply_delta_replaces_existing_content():
    t = integrated_text("old")
    blocks.apply_delta_to_text(t, [{"insert": "new"}])
    assert str(t) == "new"


def test_apply_delta_none_clears_text():
    t = integrated_text("old")
    blocks.apply_delta_to_text(t, None)
    assert str(t) == ""


def test_apply_delta_skips_non_string_insert():
    t = integrated_text()
    blocks.apply_delta_to_text(t, [{"insert": 5}, {"insert": "ok"}])
    assert str(t) == "ok"


def test_apply_delta_skips_empty_string():
    t = integrated_text()
    blocks.apply_delta_to_text(t, [{"insert": ""}, {"insert": "x"}])
    assert str(t) == "x"


def test_apply_delta_with_attributes():
    t = integrated_text()
    blocks.apply_delta_to_text(
        t, [{"insert": "bold", "attributes": {"bold": True}}, {"insert": " plain"}]
    )
    assert str(t) == "bold plain"


def test_block_data_json_heading_uses_level():
    assert json.loads(blocks.block_data_json("heading", 2, False)) == {"level": 2}


def test_block_data_json_heading_defaults_level_one():
    assert json.loads(blocks.block_data_json("heading", 0, False)) == {"level": 1}


def test_block_data_json_todo_list_checked():
    assert json.loads(blocks.block_data_json("todo_list", 1, True)) == {"checked": True}


def test_block_data_json_image_uses_url():
    assert json.loads(blocks.block_data_json("image", 1, False, "http://x/i.png")) == {
        "url": "http://x/i.png"
    }


def test_block_data_json_other_is_empty_object():
    assert blocks.block_data_json("paragraph", 1, False) == "{}"


def test_load_document_returns_doc_and_document_map():
    doc, document = blocks.load_document(build_collab())
    assert "blocks" in blocks.ykeys(document)
    assert doc is not None


def test_block_text_returns_content():
    _doc, document = blocks.load_document(
        build_collab(blocks=[{"id": "b1", "text": "hello"}])
    )
    assert blocks.block_text(document, "b1") == "hello"


def test_block_text_none_when_no_external_id():
    _doc, document = blocks.load_document(
        build_collab(blocks=[{"id": "b1", "text": None}])
    )
    assert blocks.block_text(document, "b1") is None


def test_ordered_blocks_walks_tree_with_depth():
    collab = build_collab(
        blocks=[
            {"id": "b1", "text": "top"},
            {"id": "b2", "text": "child", "parent": "b1"},
        ]
    )
    _doc, document = blocks.load_document(collab)
    ordered = blocks.ordered_blocks(document)
    by_id = {b["id"]: b for b in ordered}
    assert by_id["b1"]["depth"] == 0
    assert by_id["b2"]["depth"] == 1


def test_ordered_blocks_includes_type_and_text():
    _doc, document = blocks.load_document(
        build_collab(blocks=[{"id": "b1", "ty": "heading", "text": "H"}])
    )
    entry = next(b for b in blocks.ordered_blocks(document) if b["id"] == "b1")
    assert entry["type"] == "heading"
    assert entry["text"] == "H"


def test_ordered_blocks_fallback_without_children_map():
    collab = build_collab(
        blocks=[{"id": "b1", "text": "x"}], with_children_map=False
    )
    _doc, document = blocks.load_document(collab)
    ids = {b["id"] for b in blocks.ordered_blocks(document)}
    assert "b1" in ids


def test_ordered_blocks_fallback_without_page_id():
    collab = build_collab(blocks=[{"id": "b1", "text": "x"}], page_id=None)
    _doc, document = blocks.load_document(collab)
    ordered = blocks.ordered_blocks(document)
    assert all(b["depth"] == 0 for b in ordered)


def test_ordered_blocks_ignores_dangling_child_references():
    collab = raw_document(
        block_defs={
            "root": {"children": "ch-root"},
            "A": {"parent": "root", "children": "ch-A"},
            "B": {"parent": "A", "children": ""},
        },
        children_defs={"ch-root": ["A", "dangling1"], "ch-A": ["B", "dangling2"]},
    )
    _doc, document = blocks.load_document(collab)
    ids = [b["id"] for b in blocks.ordered_blocks(document)]
    assert ids == ["A", "B"]


def test_ordered_blocks_fallback_when_root_has_no_children_array():
    collab = raw_document(
        block_defs={"root": {"children": ""}, "X": {"parent": "root", "children": ""}},
        children_defs={},
    )
    _doc, document = blocks.load_document(collab)
    ids = {b["id"] for b in blocks.ordered_blocks(document)}
    assert ids == {"X"}


# -- insert_tree -----------------------------------------------------------
def loaded(blocks_spec=None, **kw):
    return blocks.load_document(build_collab(blocks=blocks_spec, **kw))


def root_children(document):
    meta = document["meta"]
    return [str(x) for x in meta["children_map"][document["blocks"]["root"]["children"]]]


def child_ids(document, block_id):
    meta = document["meta"]
    return [str(x) for x in meta["children_map"][document["blocks"][block_id]["children"]]]


def test_insert_tree_appends_to_page_root_by_default():
    _doc, document = loaded([{"id": "a", "text": "a"}])
    new = blocks.insert_tree(document, {"text": "x"})
    assert root_children(document) == ["a", new]


def test_insert_tree_honours_index():
    _doc, document = loaded([{"id": "a", "text": "a"}])
    new = blocks.insert_tree(document, {"text": "x"}, index=0)
    assert root_children(document) == [new, "a"]


def test_insert_tree_nests_under_parent_id():
    _doc, document = loaded([{"id": "a", "text": "a"}])
    new = blocks.insert_tree(document, {"text": "x"}, parent_id="a")
    assert child_ids(document, "a") == [new]


def test_insert_tree_keeps_children_in_list_order():
    _doc, document = loaded()
    node = {
        "type": "bulleted_list",
        "children": [
            {"text": "1", "children": [{"text": "1a"}]},
            {"text": "2"},
        ],
    }
    new = blocks.insert_tree(document, node)
    kids = child_ids(document, new)
    assert [blocks.block_text(document, k) for k in kids] == ["1", "2"]


def test_insert_tree_uses_delta_over_text():
    _doc, document = loaded()
    new = blocks.insert_tree(document, {"text": "plain", "data": {"delta": [{"insert": "rich"}]}})
    assert blocks.block_text(document, new) == "rich"


def test_insert_tree_stores_data_without_delta():
    _doc, document = loaded()
    new = blocks.insert_tree(document, {"type": "heading", "data": {"level": 2, "delta": []}})
    assert json.loads(document["blocks"][new]["data"]) == {"level": 2}


def test_insert_tree_gives_container_blocks_no_text():
    _doc, document = loaded()
    new = blocks.insert_tree(document, {"type": "simple_table"})
    assert document["blocks"][new]["external_id"] == ""


def test_insert_tree_text_block_without_content_is_empty():
    _doc, document = loaded()
    new = blocks.insert_tree(document, {"type": "paragraph"})
    assert blocks.block_text(document, new) == ""


def test_insert_tree_without_children_map_raises():
    _doc, document = loaded(with_children_map=False)
    with pytest.raises(ValueError, match="no children_map"):
        blocks.insert_tree(document, {"text": "x"})


def test_insert_tree_unknown_parent_raises():
    _doc, document = loaded()
    with pytest.raises(ValueError, match="not found on this page"):
        blocks.insert_tree(document, {"text": "x"}, parent_id="ghost")


def test_insert_tree_without_page_root_raises():
    _doc, document = loaded(page_id=None)
    with pytest.raises(ValueError, match="not found on this page"):
        blocks.insert_tree(document, {"text": "x"})


def test_insert_tree_parent_without_children_array_raises():
    _doc, document = loaded([{"id": "a", "text": "a", "no_children": True}])
    with pytest.raises(ValueError, match="has no children array"):
        blocks.insert_tree(document, {"text": "x"}, parent_id="a")


# -- table_node ------------------------------------------------------------
def test_table_node_pads_short_rows():
    node = blocks.table_node([["a", "b"], ["c"]])
    assert [len(r["children"]) for r in node["children"]] == [2, 2]


def test_table_node_sets_header_flag():
    node = blocks.table_node([["a"]], header_row=False)
    assert node["data"]["enable_header_row"] is False


def test_table_node_cell_text_lands_in_paragraph():
    node = blocks.table_node([["hello"]])
    assert node["children"][0]["children"][0]["children"][0] == {
        "type": "paragraph",
        "text": "hello",
    }


def test_table_node_accepts_delta_cells():
    node = blocks.table_node([[[{"insert": "b", "attributes": {"bold": True}}]]])
    para = node["children"][0]["children"][0]["children"][0]
    assert para["data"]["delta"][0]["insert"] == "b"


def test_table_node_empty_cell_has_no_text():
    node = blocks.table_node([["a", ""]])
    assert node["children"][0]["children"][1]["children"][0] == {"type": "paragraph"}


def test_table_node_without_rows_raises():
    with pytest.raises(ValueError, match="at least one row"):
        blocks.table_node([[]])


def test_table_node_rejects_non_list_rows():
    with pytest.raises(ValueError, match="list of lists"):
        blocks.table_node(["a,b"])
