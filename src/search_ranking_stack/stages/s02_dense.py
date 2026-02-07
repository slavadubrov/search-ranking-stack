"""
Stage 1b: Dense Bi-Encoder Retrieval

Semantic retrieval using sentence embeddings.
Catches what BM25 misses via vocabulary mismatch.

Model: all-MiniLM-L6-v2 (22M params, fast, no GPU needed)
Blog Section: 3.2 - Dense Retrievers (Semantic)
"""

import time

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer

from ..config import BI_ENCODER_MODEL, CORPUS_EMBEDDINGS_PATH, TOP_K_RETRIEVAL
from ..data_loader import ESCIData

console = Console()


def run_dense(
    data: ESCIData,
    top_k: int = TOP_K_RETRIEVAL,
    use_cache: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Run dense bi-encoder retrieval.

    Args:
        data: BEIRData containing corpus, queries, qrels
        top_k: Number of documents to retrieve per query
        use_cache: Whether to cache/load corpus embeddings

    Returns:
        results: {query_id: {doc_id: cosine_similarity_score}}
    """
    console.print(
        f"\n[bold cyan]Stage 1b: Dense Bi-Encoder Retrieval ({BI_ENCODER_MODEL})[/bold cyan]"
    )

    # Load model
    model = SentenceTransformer(BI_ENCODER_MODEL)

    doc_ids = list(data.corpus.keys())
    doc_texts = [data.corpus[doc_id] for doc_id in doc_ids]

    # Encode documents (or load from cache)
    if use_cache and CORPUS_EMBEDDINGS_PATH.exists():
        console.print(f"  Loading cached embeddings from {CORPUS_EMBEDDINGS_PATH.name}...")
        corpus_embeddings = np.load(CORPUS_EMBEDDINGS_PATH)
    else:
        console.print(f"  Encoding {len(data.corpus):,} documents...", end=" ")
        start = time.time()

        corpus_embeddings = model.encode(
            doc_texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,  # Enables cosine sim via dot product
            convert_to_numpy=True,
        )

        encode_time = time.time() - start
        console.print(f"done ({encode_time:.1f}s)")

        # Cache embeddings
        if use_cache:
            np.save(CORPUS_EMBEDDINGS_PATH, corpus_embeddings)
            console.print(f"  [dim]Cached embeddings to {CORPUS_EMBEDDINGS_PATH.name}[/dim]")

    # Encode queries
    console.print(f"  Encoding {len(data.queries):,} queries...", end=" ")
    start = time.time()

    query_texts = list(data.queries.values())
    query_ids = list(data.queries.keys())

    query_embeddings = model.encode(
        query_texts,
        batch_size=128,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_time = time.time() - start
    console.print(f"done ({query_time:.1f}s)")

    # Compute similarities
    console.print("  Computing similarities...", end=" ")
    start = time.time()

    # With normalized embeddings, cosine similarity = dot product
    similarity_matrix = np.dot(query_embeddings, corpus_embeddings.T)

    sim_time = time.time() - start
    console.print(f"done ({sim_time:.2f}s)")

    # Extract top-k for each query
    results: dict[str, dict[str, float]] = {}

    for i, query_id in enumerate(query_ids):
        scores = similarity_matrix[i]
        top_indices = np.argsort(scores)[::-1][:top_k]
        results[query_id] = {doc_ids[idx]: float(scores[idx]) for idx in top_indices}

    return results
