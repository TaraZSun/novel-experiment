import tiktoken
import base_chunker
from utils import clean_text
from utils import  schema
from constants import ChunkerConfig
import abc

class BaseTextChunker(base_chunker.BaseChunker):
    def __init__(self, config: ChunkerConfig):
        self.config = config
        self.tokenizer = tiktoken.get_encoding(self.config.encoding_name)

    def chunk(self, text)-> list[schema.Chunk]:
        cleaned_text = self._preprocess(text)
        return self._split_text(cleaned_text)


    def _preprocess(self, text:str)->str:
        return clean_text(
            text,
            clean_html = self.config.clean_html,
            clean_markdown = self.config.clean_markdown
                                    )
    
    @abc.abstractmethod
    def _split_text(self, text)-> list[schema.Chunk]:
        pass
