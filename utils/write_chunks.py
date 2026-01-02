"""
Utility functions for writing text chunks to files.
"""
import json
from pathlib import Path
from utils import schema

def write_chunks_to_json(chunks:list[schema.Chunk], output_dir:str):
    """
    Write chunks to a JSON file in the specified output directory.
    Each chunk is represented as a dictionary with its id, text, and token count.

    Args:
        chunks (list[schema.Chunk]): List of Chunk objects to write.
        output_dir (Path): Directory where the output file will be saved.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        file_name = f"chunk_{chunk.id}.json"
        file_path = output_dir / file_name

        chunk_dict={
            "id": chunk.id,
            "text": chunk.text,
            "token_count": chunk.token_count,}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_dict, f, ensure_ascii=False, indent=2)
    print(f"✓ Chunks written to {output_dir}")


def write_chunks_to_single_json(chunks:list[schema.Chunk], output_file:str):
    """
    Write all chunks to a single JSON file in the specified output directory.
    Each chunk is represented as a dictionary with its id, text, and token count.

    Args:
        chunks (list[schema.Chunk]): List of Chunk objects to write.
        output_dir (Path): Directory where the output file will be saved.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_list = []
    for chunk in chunks:
        chunk_dict={
            "id": chunk.id,
            "text": chunk.text,
            "token_count": chunk.token_count,}
        chunks_list.append(chunk_dict)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_list, f, ensure_ascii=False, indent=2)

def write_chunks_to_txt(chunks:list[schema.Chunk], output_dir:str):
    """
    Write chunks to a TXT file in the specified output directory.
    Each chunk is separated by two newlines.

    Args:
        chunks (list[schema.Chunk]): List of Chunk objects to write.
        output_dir (Path): Directory where the output file will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunks in enumerate(chunks):
            if i>0:
                f.write("\n\n")

            f.write(f"--- Chunk {chunks.id} (Tokens: {chunks.token_count}) ---\n")
            f.write(chunks.text)
            f.write("\n")
    print(f"✓ Chunks written to {output_file}")

def read_chunks_from_json(input_file:Path)->list[schema.Chunk]:
    """
    Read chunks from a JSON file.

    Args:
        input_file (Path): Path to the input JSON file.

    Returns:
        list[schema.Chunk]: List of Chunk objects read from the file.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    chunks = []
    for filepath in sorted(input_file.parent.glob("chunk_*.json")):
        with open(filepath, 'r', encoding='utf-8') as f:
            chunk_dict = json.load(f)
            chunk = schema.Chunk(
                id=chunk_dict["id"],
                text=chunk_dict["text"],
                token_count=chunk_dict["token_count"],
            )
            chunks.append(chunk)
    chunks.sort(key=lambda c: c.id)
    return chunks

def get_chunks_summary(chunks:list[schema.Chunk])->dict[str, int | float]:
    """
    Generate a summary string for a list of chunks.

    Args:
        chunks (list[schema.Chunk]): List of Chunk objects.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_tokens": 0,
            "avg_tokens": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
        }
    
    counts = [c.token_count for c in chunks]
    total_tokens = sum(counts)
    return {
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens/len(counts),
        "min_tokens": min(counts),
        "max_tokens": max(counts),
    }

