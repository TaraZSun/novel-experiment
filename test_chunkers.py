import argparse
from pathlib import Path
import time
from utils import write_chunks
import consts
from chunkers.embeddings.factory import build_embedder
from settings import EmbeddingConfig
from settings import ChunkerConfig, ChunkConfig
from chunkers.cluster_semantic_chunker import ClusterSemanticChunker
from chunkers.greedy_semantic_chunker import GreedySemanticChunker
novel =consts.NOVEL_PATH

def load_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def test_chunkers(chunker, text: str, method_name:str, output_dir:Path):

    start_time = time.time()
    chunks = chunker.chunk(text)
    elapsed_time = time.time() - start_time

    stats = write_chunks.get_chunks_summary(chunks)
    print(f"\n✓ Chunking completed in {elapsed_time:.2f} seconds")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"  Avg tokens/chunk: {stats['avg_tokens']:.1f}")
    print(f"  Min tokens: {stats['min_tokens']}")
    print(f"  Max tokens: {stats['max_tokens']}")
    
    # Save outputs
    method_output_dir = f"{output_dir}/{method_name}"
    write_chunks.write_chunks_to_json(chunks, method_output_dir)
    write_chunks.write_chunks_to_single_json(chunks, f"{method_output_dir}/all_chunks.json")
    write_chunks.write_chunks_to_txt(chunks, f"{method_output_dir}/chunks.txt")
    
    print(f"\n✓ Output saved to: {method_output_dir}/")
    return {
        'method': method_name,
        'chunks': chunks,
        'stats': stats,
        'time': elapsed_time
    }
def compare_results(results: list):
    """Print comparison table of results."""
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Chunks':<10} {'Avg Tokens':<12} {'Time (s)':<10}")
    print(f"{'-'*70}")
    
    for result in results:
        print(f"{result['method']:<15} "
              f"{result['stats']['total_chunks']:<10} "
              f"{result['stats']['avg_tokens']:<12.1f} "
              f"{result['time']:<10.2f}")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test semantic chunkers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test greedy chunker
  python test_chunkers.py --method greedy --input data/sample.txt
  
  # Test cluster chunker
  python test_chunkers.py --method cluster --input data/sample.txt
  
  # Test both and compare
  python test_chunkers.py --method both --input data/sample.txt
  
  # Customize chunk size
  python test_chunkers.py --method both --input data/sample.txt --chunk-size 1024
  
  # Use different embedding model
  python test_chunkers.py --method greedy --input data/sample.txt --model all-mpnet-base-v2
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['greedy', 'cluster', 'both'],
        default='both',
        help='Which chunking method to test (default: both)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=novel,
        help='Path to input text file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output_dir/test_results',
        help='Output directory for results (default: output_dir/test_results)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=512,
        help='Maximum chunk size in tokens (default: 512)'
    )
    
    parser.add_argument(
        '--segment-size',
        type=int,
        default=50,
        help='Segment size for cluster chunker (default: 50)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='Similarity threshold for greedy chunker (default: 0.8)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--precollapse_min_tokens',
        type=int,
        default=80,
        help='Min tokens for precollapse (default: 80)'
    )
    
    parser.add_argument(
        '--fill_floor',
        type=float,
        default=0.95,
        help='Pack fill ratio (default: 0.95)'
    )
    
    args = parser.parse_args()
    
    # Load input text
    print(f"\nLoading text from: {args.input}")
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return
    
    text = load_text(args.input)
    print(f"✓ Loaded {len(text)} characters")
    
    # Build embedder
    print(f"\nInitializing embedder with model: {args.model}")
    embedding_config = EmbeddingConfig(
        local_embedding=True,
        provider='st',
        st_model=args.model,
        normalize=True
    )
    embedder = build_embedder(embedding_config)
    print("✓ Embedder initialized")
    # Configure chunker
    chunker_config = ChunkerConfig(
        chunk=ChunkConfig(
            chunk_size=args.chunk_size,)
    )
    
    # Test chunkers
    results = []
    
    if args.method in ['greedy', 'both']:
        chunker = GreedySemanticChunker(
            config=chunker_config,
            embedder=embedder,
            similarity_threshold=args.threshold,
            precollapse_min_tokens=args.precollapse_min_tokens,
            fill_floor=args.fill_floor,
        )
        result = test_chunkers(chunker, text, 'greedy', args.output_dir)
        results.append(result)
    
    if args.method in ['cluster', 'both']:
        chunker = ClusterSemanticChunker(
            config=chunker_config,
            embedder=embedder,
            segment_size=args.segment_size,
        )
        result = test_chunkers(chunker, text, 'cluster', args.output_dir)
        results.append(result)
    
    # Compare if both were tested
    if len(results) > 1:
        compare_results(results)
    
    print(f"\n✓ All tests completed!")
    print(f"✓ Results saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main()