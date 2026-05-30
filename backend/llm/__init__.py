"""Local LLM helpers (Ollama) for prompt improvement + story decomposition.

Re-exports the public API so callers keep using `import llm; llm.plan_storybook(...)`."""
from llm.client import LLM_MODEL, OLLAMA_URL, is_available, unload
from llm.planning import improve_prompt, plan_storybook
from llm.render import (
    coerce_characters,
    coerce_locations,
    location_by_id,
    protagonist_of,
    render_canon,
    render_cast,
    render_location,
    render_objects,
)

__all__ = [
    "LLM_MODEL", "OLLAMA_URL", "is_available", "unload",
    "improve_prompt", "plan_storybook",
    "coerce_characters", "coerce_locations", "location_by_id", "protagonist_of",
    "render_canon", "render_cast", "render_location", "render_objects",
]
