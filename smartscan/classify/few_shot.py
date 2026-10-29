import numpy as np
from smartscan.classify.types import  ClassificationResult
from smartscan.cluster.types import  Cluster
from smartscan.embeds.types import StoredEmbedding

def few_shot_classify(item: StoredEmbedding, labelled_clusters: list[Cluster]) -> ClassificationResult:
        label = None
        best_sim = 0.0
        
        for cluster in labelled_clusters:
            try:
                similarity = np.dot(item.embedding, cluster.embedding)
            except Exception as e:
                continue
            threshold = cluster.metadata.mean_similarity - cluster.metadata.std_similarity
            if similarity > best_sim and similarity >= threshold:
                label = cluster.metadata.label
                best_sim = similarity
        return ClassificationResult(item_id=item.item_id, label=label, similarity=float(best_sim))
