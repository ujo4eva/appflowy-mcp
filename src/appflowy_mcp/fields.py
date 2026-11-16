"""In-place database field (column) editing via Yjs/CRDT.

The REST API can list and add fields but not rename or delete them, and it cannot
add options to a select field. Fields live in the database collab (``collab_type``
1) under ``data.database.fields.<field id>``; every view keeps its own
``field_orders`` array plus per-field ``field_settings``, and may reference a field
from ``filters``, ``sorts`` and ``groups``. Deleting a field strips all of those,
like ``Database::delete_field`` upstream.
"""

from __future__ import annotations

import json
import secrets
import string
import time

from . import blocks

SINGLE_SELECT = 3
MULTI_SELECT = 4

# The colour names AppFlowy's select options cycle through.
OPTION_COLORS = (
    "Purple",
    "Pink",
    "LightPink",
    "Orange",
    "Yellow",
    "Lime",
    "Green",
    "Aqua",
    "Blue",
)

_ID_ALPHABET = string.ascii_letters + string.digits


def _database(doc):
    return doc.get("data", type=blocks.YMap)["database"]


def _field_map(doc, field_id: str):
    fields = _database(doc)["fields"]
    if field_id not in blocks.ykeys(fields):
        raise ValueError(f"field {field_id!r} not found")
    return fields[field_id]


def rename_field(doc, field_id: str, name: str) -> None:
    field = _field_map(doc, field_id)
    field["name"] = name
    field["last_modified"] = int(time.time())


def _strip_field_refs(array, field_id: str) -> None:
    for i in range(len(array) - 1, -1, -1):
        entry = array[i]
        # ponytail: order/filter/sort/group entries are maps -> dicts.
        if entry.get("id") == field_id or entry.get("field_id") == field_id:
            del array[i]


def delete_field(doc, field_id: str) -> None:
    """Remove a field from the database and every view that references it."""
    field = _field_map(doc, field_id)
    if field.get("is_primary"):
        raise ValueError(f"field {field_id!r} is the primary field and cannot be deleted")
    database = _database(doc)
    views = database["views"]
    for view_id in blocks.ykeys(views):
        view = views[view_id]
        keys = blocks.ykeys(view)
        for key in ("field_orders", "filters", "sorts", "groups"):
            if key in keys:
                _strip_field_refs(view[key], field_id)
        if "field_settings" in keys and field_id in blocks.ykeys(view["field_settings"]):
            del view["field_settings"][field_id]
    del database["fields"][field_id]


def _new_option_id(taken: set) -> str:
    while True:
        candidate = "".join(secrets.choice(_ID_ALPHABET) for _ in range(4))
        if candidate not in taken:
            return candidate


def add_select_options(doc, field_id: str, names: list) -> list:
    """Append options to a Single/MultiSelect field; returns the options added.

    Names that already exist are skipped, so the call is safe to repeat.
    """
    field = _field_map(doc, field_id)
    field_type = field.get("ty")
    if field_type not in (SINGLE_SELECT, MULTI_SELECT):
        raise ValueError(f"field {field_id!r} is not a select field")
    type_options = field["type_option"]
    # Numbers can decode as floats; the type_option map is keyed "3", not "3.0".
    key = str(int(field_type))
    if key not in blocks.ykeys(type_options):
        type_options[key] = blocks.YMap()
    type_option = type_options[key]
    raw = type_option.get("content") if "content" in blocks.ykeys(type_option) else None
    content = json.loads(raw) if raw else {}
    options = content.setdefault("options", [])
    content.setdefault("disable_color", False)
    existing = {o.get("name") for o in options}
    taken = {o.get("id") for o in options}
    added = []
    for name in names:
        if name in existing:
            continue
        option = {
            "id": _new_option_id(taken),
            "name": name,
            "color": OPTION_COLORS[len(options) % len(OPTION_COLORS)],
        }
        options.append(option)
        added.append(option)
        existing.add(name)
        taken.add(option["id"])
    type_option["content"] = json.dumps(content)
    field["last_modified"] = int(time.time())
    return added
