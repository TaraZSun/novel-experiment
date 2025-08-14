## Text Chunkers

This project provides 4 different chunking strategies for splitting long text into smaller, coherent pieces:
```bash
1. fixed – splits by fixed token/character windows
2. paragraph – preserves paragraph boundaries
3. semantic – splits based on semantic similarity (requires embeddings)
4. llm – uses a Large Language Model (LLM) to produce natural, context-preserving chunks (supports Groq & OpenAI)
```
The LLMChunker is designed to handle Groq’s free tier limits by batching and enforcing strict JSON/base64-safe output.

## Choosing a Chunker
```bash 
fixed – fastest, most predictable
paragraph – keeps paragraph integrity
semantic – good for unstructured text
llm – highest quality, preserves meaning, adds optional notes
```

## Chunker Structure
```bash
project/
├─ chunkers/
│  ├─ run_chunking.py          # CLI entrypoint (registers all 4 strategies)
│  ├─ fixed_size_chunker.py
│  ├─ paragraph_chunker.py
│  ├─ semantic_chunker.py
│  ├─ llm_chunker.py           # LLMChunker (batching + JSON/base64 + fallback)
│  └─ llm_chunker_debug.py     # Minimal debug runner (optional)
├─ prompts/
│  └─ llm_chunker_prompt.yaml  
```

## Environment Variables

Create a .env file in the project root:
```bash
# Prefer Groq for free-tier models
GROQ_API_KEY="gsk_xxx"
# Optional fallback:
# OPENAI_API_KEY="sk-xxx"
```
`Notes`:

.env is loaded by Python via load_dotenv() — it will not automatically appear in your terminal environment.

If you want to use it in the shell (echo $GROQ_API_KEY), run:
```bash
set -a; source .env; set +a
```

## CLI Usage

Run `run_chunking.py` to use any of the 4 strategies:
```bash
python -m chunkers.run_chunking \
  -s {fixed|paragraph|semantic|llm} \
  -i INPUT_FILE \
  [--size 500] [--overlap 50] \
  [--threshold 0.35] \
  [--model llama-3.1-8b-instant] \
  [--prompt prompts/llm_chunker_prompt.yaml] \
  [--overlap-sentences 1]
```
### Examples
#### Fixed size:
```bash
python -m chunkers.run_chunking \
  -s fixed \
  -i data/alice_cleaned.txt \
  --size 500 --overlap 50
```

#### Paragraph:
```bash
python -m chunkers.run_chunking \
  -s paragraph \
  -i data/alice_cleaned.txt \
  --size 500
```

#### Semantic:
```bash
python -m chunkers.run_chunking \
  -s semantic \
  -i data/alice_cleaned.txt \
  --size 500 --threshold 0.35
```

#### LLM (Groq):
```bash
python -m chunkers.run_chunking \
  -s llm \
  -i data/alice_cleaned.txt \
  --size 100 \
  --model llama-3.1-8b-instant \
  --prompt prompts/llm_chunker_prompt.yaml \
  --overlap-sentences 1
```

## Groq Limits & Batching
```bash
Groq free tier has token-per-request and TPM limits.
LLMChunker splits input into batches (llm_batch_chars default: ~2000 characters).
If you hit 413 errors (“Request too large”), reduce:
    a.llm_batch_chars (e.g., 1500)
    b.--size (target chunk size)
```
