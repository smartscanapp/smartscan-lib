from dataclasses import dataclass

@dataclass(frozen=True)
class ClusterAccuracy:
    per_label: dict[str, float]
    mean_accuracy: float

@dataclass(frozen=True)
class ClusteredItem:
    item_id: str
    cluster_id: str
