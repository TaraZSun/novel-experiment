from utils import write_chunks
import pathlib
import logging
import argparse
from constants import CHUNK_SIZE, NOVEL_FILE_PATH

from chunkers.fixed_size_chunker import FixedSizeChunker
from chunkers.llm_chunker import LLMChunker
from chunkers.semantic_chunker import SemanticChunker
from chunkers.paragraph_chunker import ParagraphChunker

CHUNKER_MAP = {
    "fixed_size": FixedSizeChunker,
    "llm": LLMChunker,
    "semantic": SemanticChunker,
    "paragraph": ParagraphChunker
}

logging.basicConfig(level=logging.INFO)

def process_text(chunker_class, text, chunk_size)-> tuple:
    """process the text using the given chunker class and parameters."""
    chunker = chunker_class(chunk_size=chunk_size)
    chunks = chunker.chunk(text)
    logging.info(f"Total chunks created: {len(chunks)}")

    for c in chunks[:5]:
        logging.info(f"Chunk {c.id}: {c.token_count} tokens, preview: {c.text[:100]}...")

    return chunks

def main(selected_chunker: str = "fixed_size", 
         chunk_size: int = CHUNK_SIZE, 
     ) -> None:
    """Main function to process the novel text and save chunks to individual JSON files."""
    file_path = NOVEL_FILE_PATH
    with file_path.open("r", encoding="utf-8") as f:
        novel_text = f.read()

    chunker_class = CHUNKER_MAP[selected_chunker]

    chunks = process_text(
        chunker_class,
        novel_text,
        chunk_size=chunk_size,
        
    )

    chunks_folder = pathlib.Path("chunks")
    chunker_output_dir = chunks_folder / selected_chunker
    write_chunks.write_chunks_to_json(chunks, output_dir=str(chunker_output_dir))
    logging.info(f"Wrote {len(chunks)} chunk files to {chunker_output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a novel text file into chunks.")
    parser.add_argument(
        "--chunker",
        type=str,
        choices=CHUNKER_MAP.keys(),
        default="fixed_size",
        help="The chunker to use. Options are 'fixed_size', 'llm', 'semantic', 'paragraph'. Default is 'fixed_size'."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Number of tokens per chunk. Default is {CHUNK_SIZE}."
    )
   

    args = parser.parse_args()
    main(selected_chunker=args.chunker, chunk_size=args.chunk_size)

