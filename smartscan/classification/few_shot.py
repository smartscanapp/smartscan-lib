import numpy as np
from smartscan.classification.types import ClassPrototype, ClassificationResult
from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.classification.types import ClassificationResult, ClassPrototype
from smartscan.types import ItemEmbedding

class FewShotClassifier(BatchProcessor[ItemEmbedding, ClassificationResult]):
    def __init__(self, 
                class_prototypes: list[ClassPrototype],
                listener: ProcessorListener[ItemEmbedding, ClassificationResult],
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.class_prototypes = class_prototypes

    def on_process(self, item: ItemEmbedding) -> ClassificationResult:
        return few_shot_classify(item, self.class_prototypes)
    
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)


def few_shot_classify(item: ItemEmbedding, class_prototypes: list[ClassPrototype]) -> ClassificationResult:
        label = None
        best_sim = 0.0
        
        for class_prototype in class_prototypes:
            try:
                similarity = np.dot(item.embedding, class_prototype.prototype_embedding)
            except Exception as e:
                continue
            if similarity > best_sim and similarity >= class_prototype.cohesion_score:
                label = class_prototype.class_id
                best_sim = similarity

        return ClassificationResult(item_id=item.item_id, label=label, similarity=float(best_sim))


def calculate_cohesion_score(prototype: np.ndarray, class_sample_batch: np.ndarray) -> float:
    return np.mean(np.dot(class_sample_batch, prototype))
