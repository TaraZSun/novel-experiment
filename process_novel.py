import json
import pathlib
import logging
import argparse
from constants import CHUNK_SIZE, OVERLAP, BATCH_SIZE, NOVEL_FILE_PATH

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

def process_text(chunker_class, text, chunk_size, overlap, batch_size=BATCH_SIZE)-> tuple:
    """process the text using the given chunker class and parameters."""
    chunker = chunker_class(chunk_size=chunk_size, overlap=overlap, batch_size=batch_size)
    chunks = chunker.chunk(text)
    logging.info(f"Total chunks created: {len(chunks)}")

    for c in chunks[:5]:
        logging.info(f"Chunk {c['id']}: {c['token_count']} tokens, preview: {c['text'][:100]}...")

    # Demonstrate chunk_batches
    batches = chunker.chunk_batches(text, batch_size=batch_size)
    for i, batch in enumerate(batches, 1):
        logging.info(f"Batch {i} has {len(batch)} chunks.")

    return chunks, batches

def main(selected_chunker:str="fixed_size",chunk_size:int=CHUNK_SIZE,overlap:int=OVERLAP,batch_size:int=BATCH_SIZE)->None:
    """Main function to process the novel text and save chunks to a JSON file.
    
    Args:
        selected_chunker (str): The chunker to use. Options are "fixed_size", "llm", "semantic", "paragraph".
    """
    file_path = NOVEL_FILE_PATH
    with file_path.open("r", encoding="utf-8") as f:
        novel_text = f.read()

    chunker_class = CHUNKER_MAP[selected_chunker]

    chunks, batches = process_text(
        chunker_class,
        novel_text,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
    )


    chunks_folder = pathlib.Path("chunks")
    chunks_folder.mkdir(exist_ok=True)  

    output_batch_path = chunks_folder / f"{selected_chunker}_batches.json"
    with pathlib.Path(output_batch_path).open("w", encoding="utf-8") as f:
        json.dump(batches, f, ensure_ascii=False, indent=2)
    logging.info(f"Batches saved to {output_batch_path}")


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
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help=f"Number of overlapping tokens between chunks. Default is {OVERLAP}."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of chunks per batch. Default is {BATCH_SIZE}."
    )

    args = parser.parse_args()
    main(selected_chunker=args.chunker, chunk_size=args.chunk_size, overlap=args.overlap, batch_size=args.batch_size)

