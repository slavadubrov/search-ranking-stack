"""
Stage 2: Cross-Encoder Reranking

Reranks the hybrid top-K candidates with a cross-encoder.
This is typically the biggest ROI jump in search quality.

Model: ms-marco-MiniLM-L-12-v2 (33M params, ~12ms per pair on CPU)
Blog Section: 11 - Rerankers
"""

import time

from rich.console import Console
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from ..config import CROSS_ENCODER_MODEL, TOP_K_RERANK_CE, TOP_K_RETRIEVAL
from ..data_loader import BEIRData

console = Console()


def run_cross_encoder(
    data: BEIRData,
    hybrid_results: dict[str, dict[str, float]],
    top_k_rerank: int = TOP_K_RERANK_CE,
    top_k_output: int = TOP_K_RETRIEVAL,
) -> dict[str, dict[str, float]]:
    """
    Rerank hybrid results with a cross-encoder.

    Cross-encoders score (query, document) pairs jointly, providing
    more accurate relevance scores than bi-encoders at the cost of
    not being able to pre-compute document embeddings.

    Args:
        data: BEIRData containing corpus
        hybrid_results: Results from hybrid RRF fusion
        top_k_rerank: Number of candidates to rerank per query
        top_k_output: Number of results to return per query

    Returns:
        reranked_results: {query_id: {doc_id: cross_encoder_score}}
    """
    console.print(
        f"\n[bold cyan]Stage 2: Cross-Encoder Reranking ({CROSS_ENCODER_MODEL})[/bold cyan]"
    )

    # Load model
    console.print("  Loading cross-encoder model...")
    model = CrossEncoder(CROSS_ENCODER_MODEL)

    console.print(f"  Reranking top-{top_k_rerank} candidates per query...")

    reranked_results: dict[str, dict[str, float]] = {}
    query_times: list[float] = []

    for query_id, query_text in tqdm(data.queries.items(), desc="  Reranking"):
        if query_id not in hybrid_results:
            continue

        start = time.time()

        # Get top-k candidates for reranking
        candidates = list(hybrid_results[query_id].items())[:top_k_rerank]

        # Form (query, document) pairs
        pairs = []
        doc_ids = []
        for doc_id, _ in candidates:
            doc_text = data.corpus.get(doc_id, "")
            # Truncate to ~512 tokens (rough estimate: 4 chars per token)
            doc_text = doc_text[:2048]
            pairs.append([query_text, doc_text])
            doc_ids.append(doc_id)

        # Score all pairs
        if pairs:
            scores = model.predict(pairs, batch_size=64, show_progress_bar=False)

            # Create reranked results
            scored_docs = list(zip(doc_ids, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Take reranked top-k, then append remaining from original list
            reranked = {doc_id: float(score) for doc_id, score in scored_docs[:top_k_output]}

            # Add remaining docs from original results (positions beyond top_k_rerank)
            remaining = list(hybrid_results[query_id].items())[top_k_rerank:top_k_output]
            for doc_id, score in remaining:
                if doc_id not in reranked:
                    reranked[doc_id] = float(score) * 0.01  # Penalize unreranked docs

            reranked_results[query_id] = reranked

        query_times.append(time.time() - start)

    avg_time_ms = sum(query_times) / len(query_times) * 1000
    console.print(
        f"  {len(data.queries):,}/{len(data.queries):,} queries reranked "
        f"(avg {avg_time_ms:.0f}ms/query)"
    )

    return reranked_results
