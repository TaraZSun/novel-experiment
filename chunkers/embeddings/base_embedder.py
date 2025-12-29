from __future__ import annotations
from typing import Protocol, Sequence
import numpy as np
class BaseEmbedder(Protocol):
    def embed(self, texts: Sequence[str])->np.ndarray:...
        
    