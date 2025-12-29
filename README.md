# RAG Chunking Experiments

## Objective

Evaluate how different text chunking strategies affect Retrieval-Augmented Generation (RAG) performance on the same dataset.

## Dataset

* **data/alice.txt** — plain text dataset used for retrieval and generation evaluation.

## Hypothesis

Different chunking strategies will lead to varying levels of RAG performance, with some strategies yielding better retrieval and generation results than others.

## Chunking Strategies

1. **Fixed Size Chunks** — Divide the text into chunks of a fixed number of tokens or characters.
2. **Semantic Chunks** — Use NLP techniques to create chunks based on semantic boundaries (sentences or paragraphs).
3. **Paragraph Chunks** — Use entire paragraphs as chunks to preserve context.
4. **LLM-based Chunks** — Use a language model to identify optimal chunk boundaries based on content relevance.

## Experimental Control

To isolate the impact of chunking, **all other components remain constant** across experiments:

* **Embedding model:** same for all runs
* **Retriever / Vector database:** same implementation and parameters
* **LLM generator:** same model, temperature, and prompt template
* **Evaluation dataset and queries:** identical for every run

Only the **chunking strategy** changes between experiments.

## Evaluation Metrics

* **Retrieval Accuracy** — Recall@k, Precision@k, F1@k
* **Generation Quality** — BLEU, ROUGE, or human evaluation
* **Latency** — Average time for retrieval and generation
* **Baseline** — Direct QA without retrieval for reference

## Experiment Setup

1. Apply each chunking strategy to preprocess the dataset.
2. Build a retriever using the same embedding model and settings.
3. Run the same query set through the RAG pipeline.
4. Evaluate retrieval and generation metrics.
5. Compare results across strategies to identify which chunking method performs best.

## Expected Outcomes

* Identification of the chunking strategy that yields the best RAG performance.
* Insights into how chunk size and structure impact retrieval and generation quality.
* Recommendations for optimal chunking strategies for future RAG applications.

## Future Work

* Explore hybrid or adaptive chunking methods.
* Extend evaluation to diverse datasets (technical, conversational, narrative).
* Include user satisfaction and trust metrics in interactive settings.

## Notes

No fine-tuning is performed; all experiments use pretrained models with frozen weights.
Embedding, retriever, and LLM selections remain fixed throughout all experiments to ensure comparability.
