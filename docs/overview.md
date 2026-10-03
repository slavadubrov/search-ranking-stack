# Project Overview

Build a modern search ranking stack from BM25 to LLM reranking, measuring improvement at every stage on the Amazon ESCI product search benchmark.

## What This Project Does

This project implements a **5-stage search ranking pipeline** and benchmarks each stage on a real product search dataset. It demonstrates the industry-standard approach to search: start with fast, cheap retrieval methods, then progressively apply more expensive models to fewer candidates.

**Why it matters:** Most search tutorials show a single model in isolation. Real search systems use multiple models in a cascade. This project shows how each stage contributes — and proves it with metrics.

## Architecture

```mermaid
flowchart TB
    subgraph "Stage 1: First-Pass Retrieval"
        Q[Query] --> BM25["BM25 Sparse<br/><i>rank_bm25</i>"]
        Q --> Dense["Dense Bi-Encoder<br/><i>all-MiniLM-L6-v2</i>"]
        BM25 --> |"Top-100"| RRF["Hybrid RRF Fusion<br/><i>k=60</i>"]
        Dense --> |"Top-100"| RRF
    end

    subgraph "Stage 2: Reranking"
        RRF --> |"Top-50"| CE["Cross-Encoder<br/><i>ms-marco-MiniLM-L12-v2</i>"]
        CE --> |"Top-10"| LLM["LLM Reranker<br/><i>RankGPT-style listwise</i>"]
    end

    LLM --> Results["Final Top-10 Results"]

    style RRF fill:#3498db,color:white
    style CE fill:#e67e22,color:white
    style LLM fill:#9b59b6,color:white
```

## The Retrieval-Reranking Funnel

Each stage narrows the candidate set while improving precision:

```mermaid
flowchart LR
    Corpus["Full Corpus<br/>~9,900 products"] --> S1["BM25 + Dense<br/>Top-100 each"]
    S1 --> S1c["RRF Fusion<br/>100 candidates"]
    S1c --> S2["Cross-Encoder<br/>Top-50 reranked"]
    S2 --> S3["LLM Reranker<br/>Top-10 reranked"]
    S3 --> Final["Final Results<br/>10 products"]

    style Corpus fill:#ecf0f1,color:#2c3e50
    style S1 fill:#3498db,color:white
    style S1c fill:#2980b9,color:white
    style S2 fill:#e67e22,color:white
    style S3 fill:#9b59b6,color:white
```

**Why this funnel shape?** Cost and latency increase dramatically at each stage. BM25 scores 9,870 documents in milliseconds. The cross-encoder scores 50 candidates per query, and the LLM reads 10. By filtering aggressively, we get the best model's quality at a fraction of the cost.

## Key Insights

See the [results table in the README](../README.md#results) for the measured numbers.

| Insight | Evidence |
|---------|----------|
| **Hybrid search outperforms either method alone** | RRF NDCG (0.628) > max(BM25 0.585, Dense 0.611) |
| **Dense beats BM25 on this dataset** | Dense NDCG 0.611 vs BM25 0.585 — semantic matching helps with product search vocabulary mismatch |
| **Recall is set at retrieval** | Recall@100 stays at 0.842 through both reranking stages |
| **The cross-encoder provides the largest single jump** | +0.071 NDCG@10 over hybrid RRF (0.628 → 0.699) |
| **A small local LLM adds nothing on top** | `llama3.2:3b` leaves NDCG@10 at 0.699, with 13.4% of answers falling back to cross-encoder order |
| **Graded relevance reveals quality differences** | Label distribution plots show progression from Complement to Exact in top positions |

## Project Structure

```
search-ranking-stack/
├── src/search_ranking_stack/
│   ├── config.py                    # All hyperparameters and model names
│   ├── data_loader.py               # ESCIData schema and JSONL loading
│   ├── evaluate.py                  # NDCG, MRR, Recall via pytrec_eval
│   ├── data/
│   │   └── download.py              # HuggingFace download and sampling
│   └── stages/
│       ├── s01_bm25.py              # BM25 sparse retrieval
│       ├── s02_dense.py             # Dense bi-encoder retrieval
│       ├── s03_hybrid_rrf.py        # Reciprocal Rank Fusion
│       ├── s04_cross_encoder.py     # Cross-encoder reranking
│       └── s05_llm_rerank.py        # LLM listwise reranking
├── data/esci_sample/                # Downloaded dataset (gitignored)
├── results/
│   ├── metrics.json                 # Benchmark numbers
│   ├── metrics_comparison.png       # NDCG/MRR/Recall bar chart
│   └── label_distribution.png       # ESCI label distribution per stage
└── docs/                            # You are here
    ├── overview.md
    ├── dataset.md
    ├── methods.md
    └── metrics.md
```

## How to Use This for Learning

1. **Read the docs in order:** [Dataset](dataset.md) → [Methods](methods.md) → [Metrics](metrics.md)
2. **Run the pipeline:** `uv sync && uv run download-data && uv run run-all`
3. **Read the stage files** in order (`s01` through `s05`) — each is self-contained with docstrings explaining the approach
4. **Modify and experiment:**
   - Change `TOP_K_RETRIEVAL` in `config.py` to see how candidate pool size affects recall
   - Swap `BI_ENCODER_MODEL` for a larger model to see if dense retrieval improves
   - Adjust `RRF_K` to see how fusion sensitivity changes
   - Try different LLM models via `--llm-mode` to compare reranking quality

---

*Back to [README](../README.md) | Docs: [Dataset](dataset.md) | [Methods](methods.md) | [Metrics](metrics.md)*
