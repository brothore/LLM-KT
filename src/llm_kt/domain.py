from dataclasses import dataclass, asdict
import math
from typing import Protocol


@dataclass(frozen=True)
class ModelInput:
    sample_id: str
    history: list[dict]
    current_item: dict
    input_version: str = "history_ids_v1"

    def state(self):
        return {"history": self.history, "current_item": self.current_item}


class Predictor(Protocol):
    def request(self, item: ModelInput) -> dict: ...
    def probability(self, response: dict) -> float: ...


def checked_probability(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Probability must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability must be finite and in [0,1]")
    return float(value)
