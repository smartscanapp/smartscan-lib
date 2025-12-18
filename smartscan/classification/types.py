from dataclasses import dataclass
import numpy as np

@dataclass
class ClassPrototype:
    class_id: str
    prototype_embedding: np.ndarray
    cohesion_score: float

@dataclass
class ClassificationResult:
    item_id: str
    label: str | None
    similarity: float

@dataclass
class ClassificationInput:
    item_id: str
    embedding: np.ndarray


