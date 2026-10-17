"""Validated, dated offline pricing snapshots. Never sent to model APIs."""
import math
from .common import assert_keys, read_json

RATE_KEYS = {"input_per_million_usd", "output_per_million_usd", "cache_read_per_million_usd", "cache_write_per_million_usd"}


def load_pricing(path, model_id):
    data = read_json(path)
    assert_keys(data, {"model_id", "currency", "unit", "verified_at", "source", "endpoints"}, "pricing")
    if data.get("model_id") != model_id or data.get("currency") != "USD" or data.get("unit") != "per_million_tokens":
        raise ValueError("Pricing model/currency/unit mismatch")
    if not data.get("verified_at") or not data.get("source") or not data.get("endpoints"):
        raise ValueError("Pricing requires source, date and endpoints")
    seen = set()
    for endpoint in data["endpoints"]:
        assert_keys(endpoint, RATE_KEYS | {"provider", "tag", "supports_implicit_caching"}, "pricing.endpoint")
        if not isinstance(endpoint.get("provider"), str) or not endpoint["provider"] or not isinstance(endpoint.get("tag"), str) or not endpoint["tag"] or endpoint["tag"] in seen:
            raise ValueError("Pricing endpoints need unique tags and provider names")
        seen.add(endpoint.get("tag"))
        for name in RATE_KEYS:
            value = endpoint.get(name)
            if value is None and name.startswith("cache_"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid or missing price: {name}")
    return data
