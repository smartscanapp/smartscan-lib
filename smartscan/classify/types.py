from dataclasses import dataclass

@dataclass
class ClassificationResult:
    item_id: str
    label: str | None = None
    similarity: float = 0.0
