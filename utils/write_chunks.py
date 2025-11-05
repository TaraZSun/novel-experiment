# Description: This module contains a function to write chunk objects to individual JSON files.
import uuid
import pathlib
from utils import schema
import logging
logger = logging.getLogger(__name__)

def write_chunks_to_json(chunks: list[schema.Chunk], output_dir: str) -> None:
    """Write each chunk to a separate JSON file in the specified directory.

    Args:
        chunks (list[schema.Chunk]): List of chunk objects to write.
        output_dir (str): Directory where JSON files will be saved.
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        chunk_file = output_path / f"chunk_{chunk.id}_{uuid.uuid4().hex}.json"
        with chunk_file.open("w", encoding="utf-8") as f:
            f.write(chunk.model_dump_json(indent=2))
    logger.info(f"Wrote {len(chunks)} chunks to {output_dir}")