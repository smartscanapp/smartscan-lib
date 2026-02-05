from dataclasses import dataclass
from numpy import ndarray
from typing import NewType, Dict, List, ClassVar
from pydantic import  BaseModel

__all__ = [
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
    "ClusterResult",
    "ClusterNoEmbeddings",
]

class ClusterMetrics(BaseModel):
    prototype_size: int
    mean_similarity: float = 0
    std_similarity: float = 0

class ClusterMetadata(ClusterMetrics):
    label: str


class ClusterNoEmbeddings(BaseModel):
    UNLABELLED:ClassVar[str] = "unlabelled"
    prototype_id: str
    metadata: ClusterMetadata
    label: str
    

ItemId = NewType("ItemId", str)
ClusterId = NewType("ClusterId", str)
Assignments = Dict[ItemId, ClusterId]

MergeId = NewType("MergeId", str)
TargetClusters = NewType("TargetClusters", List[ClusterId])
ClusterMerges = Dict[MergeId, TargetClusters]

@dataclass
class Cluster:
    UNLABELLED = "unlabelled"
    prototype_id: str
    embedding: ndarray
    metadata: ClusterMetadata
    label: str

@dataclass(frozen=True)
class ClusterAccuracy:
    per_label: Dict[str, float]
    mean_accuracy: float

@dataclass(frozen=True)
class ClusterResult:
    clusters:  Dict[ClusterId, Cluster]
    assignments: Assignments
    merges: ClusterMerges




