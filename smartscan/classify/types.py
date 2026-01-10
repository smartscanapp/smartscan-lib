from dataclasses import dataclass
from numpy import ndarray
from typing import Optional, NewType, Dict, List
from pydantic import Field, BaseModel

__all__ = [
    "ClassificationResult",
    "ClusterAccuracy",
    "ClusterMetrics",
    "ClusterMetadata",
    "Cluster",
    "ItemId",
    "ClusterId",
    "Assignments",
    "MergeId",
    "TargetClusters",
    "ClusterMerges",
    "ClusterResult"
]

@dataclass(frozen=True)
class ClusterAccuracy:
    per_label: Dict[str, float]
    mean_accuracy: float

class ClusterMetrics(BaseModel):
    prototype_size: int
    mean_similarity: float = 0
    min_similarity: float = 0
    max_similarity: float = 0
    nearest_other_similarity: Optional[float] = None
    separation_margin: Optional[float] = None

class ClusterMetadata(ClusterMetrics):
    label: str


@dataclass
class Cluster:
    UNLABELLED = "unlabelled"
    prototype_id: str
    embedding: ndarray
    metadata: ClusterMetadata
    label: str

ItemId = NewType("ItemId", str)
ClusterId = NewType("ClusterId", str)
Assignments = Dict[ItemId, ClusterId]

MergeId = NewType("MergeId", str)
TargetClusters = NewType("TargetClusters", List[ClusterId])
ClusterMerges = Dict[MergeId, TargetClusters]

class ClassificationResult(BaseModel):
    item_id: str
    label: Optional[str] = Field(default=None)

@dataclass(frozen=True)
class ClusterResult:
    clusters:  Dict[ClusterId, Cluster]
    assignments: Assignments
    merges: ClusterMerges




