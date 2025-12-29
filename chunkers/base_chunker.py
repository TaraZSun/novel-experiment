import abc
from utils import schema
from settings import ChunkerConfig
from utils.clean_text import preprocess_text
import tiktoken


class BaseChunker(abc.ABC):
    def __init__(self, *, config: ChunkerConfig):
        self.config = config
        self.tokenizer = tiktoken.get_encoding(config.chunk.encoding_name)

    def count_tokens(self, text:str)->int:
        return len(self.tokenizer.encode(text))

    def preprocess(self, text:str)->str:
        return preprocess_text(text, self.config)
    
    @abc.abstractmethod
    def chunk(self, text:str)->list[schema.Chunk]:
        pass
