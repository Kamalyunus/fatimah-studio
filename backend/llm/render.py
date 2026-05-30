"""Pure helpers that shape LLM plan output into canon/cast/location strings and dicts.
No I/O — safe to call anywhere."""
from __future__ import annotations

import re

def render_canon(canon: dict | None) -> str:
    """Render a single character canon dict as a descriptive clause used verbatim in
    Flux prompts. Empty/missing parts are skipped so we don't leak placeholder text."""
    if not isinstance(canon, dict):
        return ""
    parts: list[str] = []
    for key in ("species", "colors", "features", "clothing", "accessories"):
        v = (canon.get(key) or "").strip()
        if v:
            parts.append(v)
    if not parts:
        return ""
    name = (canon.get("name") or "").strip()
    body = ", ".join(parts)
    return f"{name} ({body})" if name else body


def render_cast(characters: list, names: list[str] | None = None) -> str:
    """Render multiple character canons as a single combined clause. If `names` is
    given, only render the characters whose `name` matches one in the list. Joined
    with semicolons so Flux/Kontext parses each canon distinctly."""
    if not isinstance(characters, list) or not characters:
        return ""
    if names is not None:
        wanted = {n.strip().lower() for n in names if n}
        characters = [c for c in characters if isinstance(c, dict)
                      and (c.get("name") or "").strip().lower() in wanted]
    clauses = [render_canon(c) for c in characters]
    return "; ".join(c for c in clauses if c)


def coerce_characters(plan: dict) -> list[dict]:
    """Normalise the `characters` list from a plan dict. Returns a list with the
    protagonist first; every entry has a role of 'protagonist' or 'supporting'.
    Empty list if the plan has no characters."""
    chars = plan.get("characters")
    if not isinstance(chars, list) or not chars:
        return []
    out: list[dict] = []
    for c in chars:
        if isinstance(c, dict) and (c.get("name") or "").strip():
            role = (c.get("role") or "").strip().lower()
            if role not in ("protagonist", "supporting"):
                role = "protagonist" if not out else "supporting"
            out.append({**c, "role": role})
    # Exactly one protagonist (first one wins; others demoted).
    seen_protagonist = False
    for c in out:
        if c["role"] == "protagonist":
            if seen_protagonist:
                c["role"] = "supporting"
            seen_protagonist = True
    return out


def protagonist_of(characters: list[dict]) -> dict | None:
    """Return the protagonist character (first one with role='protagonist') or None."""
    for c in characters:
        if isinstance(c, dict) and c.get("role") == "protagonist":
            return c
    return characters[0] if characters else None


def _slugify_location_id(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "place"


def coerce_locations(plan: dict) -> list[dict]:
    """Normalise the `locations` list. Ensures every entry has non-empty id/name/description
    and ids are unique slugified strings. Returns at least one fallback entry if empty."""
    raw = plan.get("locations")
    out: list[dict] = []
    seen_ids: set[str] = set()
    if isinstance(raw, list):
        for loc in raw:
            if not isinstance(loc, dict):
                continue
            lid = _slugify_location_id(loc.get("id") or loc.get("name") or "")
            if not lid or lid in seen_ids:
                # de-dup by suffixing
                base = lid or "place"
                i = 2
                while f"{base}-{i}" in seen_ids:
                    i += 1
                lid = f"{base}-{i}" if base else f"place-{i}"
            seen_ids.add(lid)
            out.append({
                "id":          lid,
                "name":        (loc.get("name") or lid.replace("-", " ")).strip(),
                "description": (loc.get("description") or "").strip(),
            })
    if not out:
        out.append({"id": "scene", "name": "the scene", "description": ""})
    return out


def location_by_id(locations: list[dict], lid: str) -> dict | None:
    """Find a location dict by id; case-insensitive, slug-normalised."""
    if not lid:
        return None
    target = _slugify_location_id(lid)
    for loc in locations:
        if isinstance(loc, dict) and _slugify_location_id(loc.get("id") or "") == target:
            return loc
    return None


def render_objects(objs: list | None) -> str:
    """Render a list of object names as a short clause for the Flux/Wan prompts.
    Empty list → empty string (callers can branch on that)."""
    if not isinstance(objs, list) or not objs:
        return ""
    cleaned = [str(o).strip() for o in objs if str(o or "").strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def render_location(loc: dict | None) -> str:
    """Render a location as a descriptive clause for prompts. Empty parts skipped."""
    if not isinstance(loc, dict):
        return ""
    name = (loc.get("name") or "").strip()
    desc = (loc.get("description") or "").strip()
    if name and desc:
        return f"{name} — {desc}"
    return name or desc


