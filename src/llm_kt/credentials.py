"""Read credentials separately from serializable experiment configuration."""
import os
from pathlib import Path


def resolve_api_key(model):
    if model["adapter"] == "mock":
        return ""
    value = os.environ.get(model["api_key_env"], "").strip()
    if value:
        return value
    path = model.get("api_key_file")
    if path:
        try:
            value = Path(path).read_text(encoding="utf-8-sig").strip()
        except OSError:
            raise ValueError("Cannot read API key file; check model.api_key_file") from None
        if value and not any(c.isspace() for c in value):
            return value
        raise ValueError("API key file must contain one nonempty key without whitespace")
    raise ValueError(f"Set {model['api_key_env']} or configure model.api_key_file")
