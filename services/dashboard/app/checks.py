"""Prove that every callback's component IDs actually exist in a layout.

Why this file exists
--------------------
The app runs with ``suppress_callback_exceptions=True``. That setting is
required for multi-page Dash -- a callback may reference a component on a page
the user has not opened yet -- but it carries a nasty edge:

    If a layout stops rendering a component that a callback references, Dash
    does not raise. The callback simply never fires. The page loads, looks
    perfect, and one card stays empty forever.

That is a false success, and a false success is worse than a visible failure.
It is the same shape as the JDBC jar incident, where the DAG reported SUCCESS
while writing zero rows. The fix is the same: make the failure loud.

This module walks every registered page layout, collects every component ID
actually rendered, then reads Dash's global callback registry and checks that
every ID the callbacks depend on is present.

It is a migration tool. Run it after every layout rewrite:

    docker compose exec dashboard python -m app.checks

Exit 0 = layouts and callbacks agree. Exit 1 = something is broken.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Iterator

import dash
import dash._callback as dash_callback
from dash.development.base_component import Component

from app import ids

# Importing main builds the app, registers the pages, and -- via its final
# import of app.callbacks -- populates the global callback registry.
from app.main import app

# Dash's own page-router internals. They are created by page_container and
# driven by Dash's clientside callbacks, which never appear in the Python
# callback registry -- so they look like orphans when they are not ours at all.
DASH_INTERNAL_PREFIX = "_pages_"

# ============================================================
# Walking layouts
# ============================================================

def _walk(node: Any) -> Iterator[Component]:
    """Yield every Dash component in a tree, depth-first."""
    if isinstance(node, Component):
        yield node
        children = getattr(node, "children", None)
        if children is not None:
            yield from _walk(children)
        return

    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _walk(child)


def collect_layout_ids() -> tuple[set[str], set[str]]:
    """Return (flat IDs, pattern types) rendered by app.layout and every page.

    Flat IDs are plain strings. Pattern types are the "type" key of dict IDs.
    Most pattern components are created at runtime by callbacks rather than
    sitting in a static layout, so all that can be verified statically is that
    the type name is one the contract in ids.py declares.
    """
    flat: set[str] = set()
    types: set[str] = set()

    roots: list[Any] = [app.layout]
    for page in dash.page_registry.values():
        layout = page.get("layout")
        roots.append(layout() if callable(layout) else layout)

    for root in roots:
        for component in _walk(root):
            component_id = getattr(component, "id", None)
            if isinstance(component_id, str):
                flat.add(component_id)
            elif isinstance(component_id, dict) and "type" in component_id:
                types.add(str(component_id["type"]))

    return flat, types


# ============================================================
# Reading the callback registry
# ============================================================

def _parse_id(raw: str) -> str | dict[str, Any]:
    """Dict IDs arrive from the registry as JSON strings. Decode them."""
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _split_output(output: str) -> list[str]:
    """Split a callback's output key into individual 'id.property' pairs.

    Dash encodes outputs three ways, and this has to survive all of them:
        single           "pred-kpis.children"
        multiple         "..pred-kpis.children...pred-table.children.."
        allow_duplicate  "doc-list.children@<hash>"
    """
    if output.startswith("..") and output.endswith(".."):
        return [part for part in output[2:-2].split("...") if part]
    return [output]


def _output_id(part: str) -> str | dict[str, Any]:
    """Pull the component ID out of one 'id.property' pair."""
    if part.startswith("{"):
        # A dict ID is JSON and may itself contain dots. Cut at the brace,
        # not at the last dot.
        close = part.rindex("}")
        return _parse_id(part[: close + 1])

    component_id, _, _property = part.rpartition(".")
    return component_id


def collect_callback_ids() -> tuple[set[str], set[str]]:
    """Return (flat IDs, pattern types) that the callbacks depend on."""
    flat: set[str] = set()
    types: set[str] = set()

    def record(component_id: str | dict[str, Any]) -> None:
        if isinstance(component_id, str):
            flat.add(component_id)
        elif isinstance(component_id, dict) and "type" in component_id:
            types.add(str(component_id["type"]))

    for spec in dash_callback.GLOBAL_CALLBACK_LIST:
        for part in _split_output(spec["output"]):
            record(_output_id(part))
        for dep in list(spec.get("inputs", [])) + list(spec.get("state", [])):
            record(_parse_id(dep["id"]))

    return flat, types


# ============================================================
# The check
# ============================================================

def verify() -> list[str]:
    """Return a list of problems. An empty list means the contract holds."""
    layout_flat, _layout_types = collect_layout_ids()
    callback_flat, callback_types = collect_callback_ids()

    problems: list[str] = []

    # 1. THE CRITICAL ONE. A callback depends on an ID no layout renders.
    #    This is the silent death: the callback never fires, nothing warns.
    for component_id in sorted(callback_flat - layout_flat):
        problems.append(
            f"SILENT DEATH  '{component_id}' is used by a callback but is not "
            f"rendered by any layout. That callback will never fire."
        )

    # 2. A callback uses a pattern type the contract does not declare, which
    #    means ids.PATTERN_TYPES has drifted from the builder functions.
    for pattern_type in sorted(callback_types - ids.PATTERN_TYPES):
        problems.append(
            f"UNDECLARED    pattern type '{pattern_type}' is used by a callback "
            f"but is missing from ids.PATTERN_TYPES."
        )

    # 3. A layout renders an ID nothing listens to. Usually a typo. Sometimes
    #    deliberate -- those are declared in ids.CALLBACK_FREE.
    #    Dash's own page-router internals are excluded: they are wired by
    #    clientside callbacks, which never reach the Python registry, so they
    #    would otherwise show up as orphans forever.
    orphans = {
        component_id
        for component_id in layout_flat - callback_flat - ids.CALLBACK_FREE
        if not component_id.startswith(DASH_INTERNAL_PREFIX)
    }
    for component_id in sorted(orphans):
        problems.append(
            f"ORPHAN        '{component_id}' is rendered but no callback uses "
            f"it. A typo, or add it to ids.CALLBACK_FREE if deliberate."
        )

    return problems

def main() -> int:
    layout_flat, layout_types = collect_layout_ids()
    callback_flat, callback_types = collect_callback_ids()
    problems = verify()

    print(
        f"layouts:   {len(layout_flat):3d} ids, {len(layout_types)} pattern types\n"
        f"callbacks: {len(callback_flat):3d} ids, {len(callback_types)} pattern types "
        f"across {len(dash_callback.GLOBAL_CALLBACK_LIST)} registered callbacks"
    )

    if not problems:
        print("\nOK  layouts and callbacks agree.")
        return 0

    print(f"\nFAIL  {len(problems)} problem(s):\n")
    for problem in problems:
        print(f"  {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())