from dataclasses import dataclass
from numpy import ndarray

@dataclass
class ClassPrototype:
    class_id: str
    prototype_embedding: ndarray
    cohesion_score: float

@dataclass
class ClassificationResult:
    item_id: str
    label: str | None = None
    similarity: float = 0.0
