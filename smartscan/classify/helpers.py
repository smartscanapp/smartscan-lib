import numpy as np

from smartscan.classify.types import BaseCluster


def merge_similar_clusters(cluster_prototypes: dict[str, BaseCluster], merge_threshold: float = 0.9,verbose: bool = False) -> dict[str, str]:
    cluster_merges: dict[str, list[str]] = {}
    cluster_ids = list(cluster_prototypes.keys())
    cluster_embeds = [p.embedding for p in cluster_prototypes.values()]

    for idx, emb in enumerate(cluster_embeds):
        sims = np.dot(cluster_embeds, emb)
        # Mask self-similarity
        sims[idx] = 0.0
        merge_indices = np.flatnonzero(sims > merge_threshold)

        if merge_indices.size > 0:
            cluster_merges[cluster_ids[idx]] = [cluster_ids[j] for j in merge_indices]
            if verbose:
                print(f"Cluster {cluster_ids[idx]} merges with: {[cluster_ids[j] for j in merge_indices]}")

    return cluster_merges

