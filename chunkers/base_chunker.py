import abc
from utils import schema

class BaseChunker(abc.ABC):
    @abc.abstractmethod
    def chunk(self, text:str)->list[schema.Chunk]:
        pass
