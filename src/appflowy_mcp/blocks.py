"""In-place block editing of AppFlowy documents via Yjs/CRDT.

AppFlowy stores a document as a Yjs collab. A block's editable rich text lives
in ``document.meta.text_map`` as a ``Y.Text`` keyed by the block's
``external_id``; the block tree/order lives in ``document.meta.children_map``.
The web client mutates these locally and ships a Yrs *update* to the
``web-update`` endpoint. We do exactly the same — read the collab, edit the Yjs
structure with :mod:`pycrdt`, and POST one update. The REST API exposes no
per-block replace endpoint, so this is the only way to edit existing content.
"""

from __future__ import annotations

import json
import secrets
import string

try:
    from pycrdt import Array as YArray
    from pycrdt import Doc as YDoc
    from pycrdt import Map as YMap
    from pycrdt import Text as YText

    HAS_PYCRDT = True
except Exception:  # noqa: BLE001  # pragma: no cover - optional import fallback
    HAS_PYCRDT = False

# AppFlowy block/text ids are 10-char nanoids over this URL-safe alphabet.
_ID_ALPHABET = string.ascii_letters + string.digits + "_-"


def new_id(length: int = 10) -> str:
    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(length))


def ykeys(node) -> list:
    """List keys of a pycrdt Map (stable across pycrdt versions)."""
    return list(node.keys())


def apply_delta_to_text(t, delta) -> None:
    """Replace a Y.Text's content from an AppFlowy delta.

    Each op looks like ``{"insert": "Hi", "attributes": {"bold": true}}``.
    Supported attributes: bold, italic, underline, strikethrough, code, color,
    bg_color, href. We pass the union of all attributes on every op (unused ones
    set to None) to avoid Yjs formatting bleed into following plain text.
    """
    t.clear()
    all_keys: set = set()
    for op in delta or []:
        all_keys |= set((op.get("attributes") or {}).keys())
    idx = 0
    for op in delta or []:
        s = op.get("insert")
        if not isinstance(s, str) or s == "":
            continue
        op_attrs = op.get("attributes") or {}
        attrs = {k: op_attrs.get(k, None) for k in all_keys} if all_keys else None
        t.insert(idx, s, attrs)
        idx += len(s)


def block_data_json(block_type: str, heading_level: int, checked: bool, url: str = "") -> str:
    """Build the per-type ``data`` JSON string stored on a block."""
    if block_type == "heading":
        return json.dumps({"level": heading_level or 1})
    if block_type == "todo_list":
        return json.dumps({"checked": bool(checked)})
    if block_type == "image":
        return json.dumps({"url": url})
    return "{}"


def load_document(encoded_collab: bytes):
    """Decode an ``encoded_collab`` blob into ``(doc, document_map)``."""
    doc = YDoc()
    doc.apply_update(encoded_collab)
    document = doc.get("data", type=YMap)["document"]
    return doc, document


def block_text(document, block_id: str):
    blocks = document["blocks"]
    text_map = document["meta"]["text_map"]
    ext = blocks[block_id]["external_id"]
    if ext and ext in ykeys(text_map):
        return str(text_map[ext])
    return None


def ordered_blocks(document) -> list:
    """Return the page's blocks as a flat, document-ordered list with depth."""
    blocks = document["blocks"]
    meta = document["meta"]
    children_map = meta["children_map"] if "children_map" in ykeys(meta) else None
    page_id = document["page_id"] if "page_id" in ykeys(document) else None
    block_ids = ykeys(blocks)

    out: list = []

    def entry(bid: str, depth: int) -> dict:
        b = blocks[bid]
        return {"id": bid, "type": str(b["ty"]), "text": block_text(document, bid), "depth": depth}

    def walk(bid: str, depth: int) -> None:
        out.append(entry(bid, depth))
        ch = blocks[bid]["children"]
        if children_map is not None and ch and ch in ykeys(children_map):
            for cid in list(children_map[ch]):
                if str(cid) in block_ids:
                    walk(str(cid), depth + 1)

    if page_id and page_id in block_ids and children_map is not None:
        root_children = blocks[page_id]["children"]
        if root_children and root_children in ykeys(children_map):
            for cid in list(children_map[root_children]):
                if str(cid) in block_ids:
                    walk(str(cid), 0)
            return out
    for bid in block_ids:
        if bid != page_id:
            out.append(entry(bid, 0))
    return out


# Container and embed blocks carry no rich text of their own.
NON_TEXT_TYPES = frozenset(
    {
        "page",
        "simple_table",
        "simple_table_row",
        "simple_table_cell",
        "table",
        "table/cell",
        "divider",
        "image",
    }
)


def _node_delta(node: dict):
    data = node.get("data") or {}
    if "delta" in data:
        return data["delta"]
    if node.get("text"):
        return [{"insert": node["text"]}]
    return None


def _create_tree(block_map, text_map, children_map, parent: str, node: dict) -> str:
    """Create ``node`` and its descendants; returns the new block id.

    The caller links the returned id into the parent's children array. Children
    are linked here in list order, which is what AppFlowy's append-block endpoint
    gets wrong for nested blocks.
    """
    block_type = node.get("type") or "paragraph"
    data = {k: v for k, v in (node.get("data") or {}).items() if k != "delta"}
    block_id = new_id()
    children_id = new_id()
    children_map[children_id] = YArray()
    text_id = ""
    if block_type not in NON_TEXT_TYPES:
        text_id = new_id()
        text_map[text_id] = YText()
        delta = _node_delta(node)
        if delta:
            apply_delta_to_text(text_map[text_id], delta)
    block_map[block_id] = YMap(
        {
            "id": block_id,
            "ty": block_type,
            "data": json.dumps(data),
            "parent": parent,
            "children": children_id,
            "external_id": text_id,
            "external_type": "text" if text_id else "",
        }
    )
    for child in node.get("children") or []:
        child_id = _create_tree(block_map, text_map, children_map, block_id, child)
        children_map[children_id].append(child_id)
    return block_id


def insert_tree(document, node: dict, parent_id: str = "", index: int = -1) -> str:
    """Insert a block tree under ``parent_id`` ("" = page root) at ``index``.

    ``node`` is ``{"type", "data": {..., "delta": [...]}, "children": [...]}``
    (``"text"`` is accepted as plain-text shorthand). Raises ValueError when the
    parent cannot take children. Call inside a transaction.
    """
    block_map = document["blocks"]
    meta = document["meta"]
    if "children_map" not in ykeys(meta):
        raise ValueError("document has no children_map; cannot position blocks")
    children_map = meta["children_map"]
    page_root = str(document["page_id"]) if "page_id" in ykeys(document) else None
    parent = parent_id or page_root
    if not parent or parent not in ykeys(block_map):
        raise ValueError(f"parent block {parent!r} not found on this page")
    parent_children_id = block_map[parent]["children"]
    if not parent_children_id or parent_children_id not in ykeys(children_map):
        raise ValueError(f"parent block {parent!r} has no children array")
    new_block = _create_tree(block_map, meta["text_map"], children_map, parent, node)
    arr = children_map[parent_children_id]
    n = len(arr)
    arr.insert(n if index < 0 or index > n else index, new_block)
    return new_block


def table_node(rows: list, header_row: bool = True) -> dict:
    """Build a ``simple_table`` tree (the table type current AppFlowy clients
    create) from rows of cell text; short rows are padded with empty cells."""
    if not all(isinstance(r, list) for r in rows):
        raise ValueError("rows must be a list of lists of cells")
    width = max((len(r) for r in rows), default=0)
    if not rows or width == 0:
        raise ValueError("a table needs at least one row with one cell")

    def cell(value) -> dict:
        para: dict = {"type": "paragraph"}
        if isinstance(value, list):
            para["data"] = {"delta": value}
        elif value not in (None, ""):
            para["text"] = str(value)
        return {"type": "simple_table_cell", "children": [para]}

    return {
        "type": "simple_table",
        "data": {"enable_header_row": bool(header_row), "enable_header_column": False},
        "children": [
            {
                "type": "simple_table_row",
                "children": [cell(r[i] if i < len(r) else "") for i in range(width)],
            }
            for r in rows
        ],
    }


__all__ = [
    "HAS_PYCRDT",
    "NON_TEXT_TYPES",
    "YArray",
    "YMap",
    "YText",
    "apply_delta_to_text",
    "block_data_json",
    "block_text",
    "insert_tree",
    "load_document",
    "new_id",
    "ordered_blocks",
    "table_node",
    "ykeys",
]
