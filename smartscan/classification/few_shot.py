import numpy as np
from smartscan.classification.types import ClassPrototype, ClassificationResult, ClassificationInput

def few_shot_classification(item: ClassificationInput, class_prototypes: list[ClassPrototype]) -> ClassificationResult:
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
