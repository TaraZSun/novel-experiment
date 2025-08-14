import os
import json
from datetime import datetime
from typing import Dict, Any, List
import argparse

from .fixed_size_chunker import FixedSizeChunker             
from .paragraph_chunker import ParagraphChunker  
from .semantic_chunker import SemanticChunker       
from .llm_chunker import LLMChunker               


# registry: strategy_name -> callable
STRATEGIES = {
    "fixed":     lambda *, text, chunk_size=500, overlap=0, **_: FixedSizeChunker(chunk_size=chunk_size, overlap=overlap).chunk(text),
    "paragraph": lambda *, text, chunk_size=500, **_: ParagraphChunker(chunk_size=chunk_size).chunk(text),
    "semantic":  lambda *, text, chunk_size=500, **_: SemanticChunker(chunk_size=chunk_size).chunk(text),
    "llm": lambda *, text, size=500, model=None, prompt=None, overlap_sentences=1, llm_batch_chars=3000, **_:
    LLMChunker(
        model=model or "llama-3.1-8b-instant",
        size=size,
        prompt_path=prompt or "prompts/llm_chunker_prompt.yaml",
        overlap_sentences=overlap_sentences,
        llm_batch_chars=llm_batch_chars,
    ).chunk(text),
}


def run_chunking(
    strategy: str,
    input_path: str,
    output_dir: str = "chunks",
    output_name: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Run a chunking strategy by name, save JSONL to output_dir, and return the output path.

    Parameters
    ----------
    strategy : {"fixed","paragraph","semantic","llm"}
        Which chunker to use.
    input_path : str
        Path to the cleaned text file.
    output_dir : str, default "chunks"
        Directory to save the JSONL file.
    output_name : Optional[str]
        File name (e.g., "alice_fixed.jsonl"). If None, it will be auto-generated.
    **kwargs :
        Extra args passed to the strategy function (e.g., size=500, overlap=50, threshold=0.35, model="...").

    Returns
    -------
    str : the full path of the saved JSONL file.
    """
    strategy = strategy.lower().strip()
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose one of: {', '.join(STRATEGIES.keys())}"
        )

    # read input text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # run the strategy
    chunk_fn = STRATEGIES[strategy]
    chunks: List[Dict[str, Any]] = chunk_fn(text=text, **kwargs)

    # ensure output folder
    os.makedirs(output_dir, exist_ok=True)

    # default file name
    if output_name is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # include a couple of common params in the name if present
        size = kwargs.get("size")
        overlap = kwargs.get("overlap")
        suffix_bits = [strategy]
        if size: 
            suffix_bits.append(f"s{size}")
        if overlap: 
            suffix_bits.append(f"o{overlap}")
        suffix_bits.append(stamp)
        output_name = f"{base}_{'_'.join(suffix_bits)}.jsonl"

    output_path = os.path.join(output_dir, output_name)

    # write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks -> {output_path}")
    return output_path


# Optional: small CLI wrapper so you can call it directly
if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Run a chunking strategy and save JSONL.")
    parser.add_argument("-s", "--strategy", required=True, choices=list(STRATEGIES.keys()))
    parser.add_argument("-i", "--input", required=True, help="Path to cleaned text file")
    parser.add_argument("-d", "--outdir", default="chunks", help="Output directory")
    parser.add_argument("-o", "--output", default=None, help="Output file name (optional)")

    # common optional params
    parser.add_argument("--size", type=int, default=None, help="Max tokens per chunk")
    parser.add_argument("--overlap", type=int, default=None, help="Token overlap (fixed)")
    parser.add_argument("--threshold", type=float, default=None, help="Semantic split threshold")
    parser.add_argument("--model", type=str, default=None, help="Model name for LLM chunking")
    # argparse: add prompt + overlap_sentences
    parser.add_argument("--prompt", type=str, default=None, help="Path to LLM chunker prompt YAML")
    parser.add_argument("--overlap-sentences", type=int, default=1, dest="overlap_sentences",
                    help="Approx sentence overlap for LLM chunking")
    args = parser.parse_args()
    # include in extra kwargs collection
    extra = {k: v for k, v in vars(args).items()
         if k in ("size", "overlap", "threshold", "model", "prompt", "overlap_sentences") and v is not None}
    

    run_chunking(
        strategy=args.strategy,
        input_path=args.input,
        output_dir=args.outdir,
        output_name=args.output,
        **extra,
    )
