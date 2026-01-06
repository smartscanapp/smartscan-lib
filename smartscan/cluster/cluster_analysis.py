import numpy as np

from smartscan import Prototype
from smartscan.cluster.types import ClusterAccuracy

def compare_clusters(cluster_prototypes: dict[str, Prototype],merge_threshold: float = 0.9,verbose: bool = False) -> dict[str, list[str]]:
    """
    Efficient row-wise clustering comparison for large prototype sets.
    Avoids full n x n similarity matrix to save memory.

    Args:
        cluster_prototypes: dict mapping prototype_id -> Prototype (with .embedding)
        merge_threshold: similarity threshold to consider a merge
        verbose: whether to print debug info

    Returns:
        dict mapping prototype_id -> list of prototype_ids to merge with
    """
    cluster_merges: dict[str, list[str]] = {}
    cluster_ids = list(cluster_prototypes.keys())
    cluster_embeds = np.array([p.embedding for p in cluster_prototypes.values()], dtype=np.float32)

    n = len(cluster_ids)

    for idx, emb in enumerate(cluster_embeds):
        sims = np.dot(cluster_embeds, emb)
        # Mask self-similarity
        sims[idx] = 0.0
        merge_indices = np.flatnonzero(sims > merge_threshold)

        if merge_indices.size > 0:
            cluster_merges[cluster_ids[idx]] = [cluster_ids[j] for j in merge_indices]
            if verbose:
                print(f"Prototype {cluster_ids[idx]} merges with: {[cluster_ids[j] for j in merge_indices]}")

    return cluster_merges

def merge_clusters(cluster_merges: dict[str, list[str]], assignments: dict[str, str]) -> dict[str, str]:
    flat_merge_map = {
        old_id: merged_id
        for merged_id, cluster_ids in cluster_merges.items()
        for old_id in cluster_ids
    }

    return {item_id: flat_merge_map.get(cluster_id, cluster_id)
            for item_id, cluster_id in assignments.items()}


def count_predicted_labels(assignments: dict[str, str], labels: list[str]) -> dict[str, int]:
    counts: dict[str, dict[str, int]] = {}

    for item_id, cluster_id in assignments.items():
        for label in labels:
            if item_id.startswith(label):
                counts.setdefault(label, {})
                counts[label][cluster_id] = counts[label].get(cluster_id, 0) + 1
                break

    return {label: max(cluster_counts.values()) for label, cluster_counts in counts.items()}


def calculate_cluster_accuracy(label_counts: dict[str, int], predicted_label_counts: dict[str, int]) -> ClusterAccuracy:
    per_label_acc = {label: 0 for label in label_counts}
    label_counts = dict(sorted(label_counts.items()))
    predicted_label_counts = dict(sorted(predicted_label_counts.items()))
    running_acc = 0.0

    for ((label, count), (_, predict_count)) in zip(label_counts.items(), predicted_label_counts.items()):
        prediction_acc = float(predict_count / count)
        per_label_acc[label] = prediction_acc
        running_acc += prediction_acc
        
    return ClusterAccuracy(per_label_acc, running_acc/len(per_label_acc))
