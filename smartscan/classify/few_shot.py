import numpy as np
from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.classify.types import ClassificationResult, LabelledCluster
from smartscan.embeds.types import ItemEmbedding

class FewShotClassifier(BatchProcessor[ItemEmbedding, ClassificationResult]):
    def __init__(self, 
                labelled_clusters: list[LabelledCluster],
                listener: ProcessorListener[ItemEmbedding, ClassificationResult],
                sim_factor: float = 1.0,
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.labelled_clusters = labelled_clusters
        self.sim_factor = sim_factor

    def on_process(self, item: ItemEmbedding) -> ClassificationResult:
        return few_shot_classify(item, self.labelled_clusters, self.sim_factor)
    
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)


def few_shot_classify(item: ItemEmbedding, labelled_clusters: list[LabelledCluster], sim_factor: float = 1.0) -> ClassificationResult:
        label = None
        best_sim = 0.0
        
        for cluster in labelled_clusters:
            try:
                similarity = np.dot(item.embedding, cluster.embedding)
            except Exception as e:
                continue
            if similarity > best_sim and similarity >= sim_factor * cluster.metadata.mean_similarity:
                label = cluster.label
                best_sim = similarity
        return ClassificationResult(item_id=item.item_id, label=label, similarity=float(best_sim))
