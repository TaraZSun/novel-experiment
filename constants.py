# constants.py
import pathlib
MARKDOWN_LINK_REGEX = r'\[([^\]]+)\]\([^)]+\)'
MARKDOWN_BOLD_REGEX = r'(\*\*|__)(.*?)\1'
MARKDOWN_ITALIC_REGEX = r'(\*|_)(.*?)\1'
MARKDOWN_CODE_REGEX = r'`(.*?)`'
MARKDOWN_HEADER_REGEX = r'#+\s*(.*)'

# Fixed-size chunker parameters
CHUNK_SIZE = 500  # Number of tokens per chunk
OVERLAP = 0      # Number of overlapping tokens between chunks
BATCH_SIZE = 5    # Number of chunks per batch
NOVEL_FILE_PATH = pathlib.Path("data/alice.txt")  # Path to the novel text file