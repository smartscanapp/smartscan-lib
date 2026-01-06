from abc import abstractmethod, ABC
from typing import Any
import numpy as np

class DetectorProvider(ABC):
    @abstractmethod
    def detect(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        pass
  
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def is_initialized(self) -> bool: 
        pass
    
    @abstractmethod
    def close_session(self):
        pass