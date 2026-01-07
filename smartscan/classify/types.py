from dataclasses import dataclass
from numpy import ndarray
from typing import Optional, NewType, Dict, TypeAlias, List

__all__ = [
    "ClassificationResult",
    "ClusterAccuracy",
    "ClusterMetrics",
    "ClusterMetadata",
    "BaseCluster",
    "LabelledCluster",
    "UnLabelledCluster",
    "ItemId",
    "ClusterId",
    "Assignments",
    "MergeId",
    "TargetClusters",
    "ClusterMerges"

]

@dataclass(frozen=True)
class ClassificationResult:
    item_id: str
    label: str | None = None
    similarity: float = 0.0


@dataclass(frozen=True)
class ClusterAccuracy:
    per_label: Dict[str, float]
    mean_accuracy: float

@dataclass(frozen=True)
class ClusterMetrics():
    prototype_size: int
    mean_similarity: float = 0
    min_similarity: float = 0
    max_similarity: float = 0
    nearest_other_similarity: Optional[float] = None
    separation_margin: Optional[float] = None

ClusterMetadata: TypeAlias = ClusterMetrics

@dataclass
class BaseCluster:
    prototype_id: str
    embedding: ndarray
    metadata: ClusterMetadata
    label: Optional[str] = None

@dataclass
class LabelledCluster(BaseCluster):
    label: str

UnLabelledCluster: TypeAlias = BaseCluster


ItemId = NewType("ItemId", str)
ClusterId = NewType("ClusterId", str)

Assignments = Dict[ItemId, ClusterId]


MergeId = NewType("MergeId", str)
TargetClusters = NewType("TargetClusters", List[str])

ClusterMerges = Dict[MergeId, TargetClusters]



