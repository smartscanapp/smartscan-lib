import numpy as np
from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.types import ClassificationResult, ItemEmbedding, Prototype

class FewShotClassifier(BatchProcessor[ItemEmbedding, ClassificationResult]):
    def __init__(self, 
                class_prototypes: list[Prototype],
                listener: ProcessorListener[ItemEmbedding, ClassificationResult],
                sim_factor: float = 1.0,
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.class_prototypes = class_prototypes
        self.sim_factor = sim_factor

    def on_process(self, item: ItemEmbedding) -> ClassificationResult:
        return few_shot_classify(item, self.class_prototypes, self.sim_factor)
    
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)


def few_shot_classify(item: ItemEmbedding, class_prototypes: list[Prototype], sim_factor: float = 1.0) -> ClassificationResult:
        label = None
        best_sim = 0.0
        
        for class_prototype in class_prototypes:
            try:
                similarity = np.dot(item.embedding, class_prototype.prototype_embedding)
            except Exception as e:
                continue
            if similarity > best_sim and similarity >= sim_factor * class_prototype.cohesion_score:
                label = class_prototype.class_id
                best_sim = similarity
        return ClassificationResult(item_id=item.item_id, label=label, similarity=float(best_sim))
