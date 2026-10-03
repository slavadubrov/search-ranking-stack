"""
Stage 2: Cross-Encoder Reranking

Scores the first `top_k_rerank` hybrid candidates with a cross-encoder, puts them
in score order, and appends the unscored candidates in their hybrid order.

Model: cross-encoder/ms-marco-MiniLM-L12-v2
Article: https://slavadubrov.github.io/blog/2026/02/08/search-ranking-stack/
Section: Cross-encoder reranking
"""

import time
from math import isfinite

from rich.console import Console
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from ..config import CROSS_ENCODER_MODEL, TOP_K_RERANK_CE, TOP_K_RETRIEVAL
from ..data_loader import ESCIData

console = Console()


def rank_scores(ordered_ids: list[str]) -> dict[str, float]:
    """Assign strictly decreasing scores (n, n-1, ..., 1) to an ordered list of IDs.

    Evaluation sorts by score, so the scores must encode the order. These are rank
    scores, not calibrated relevance estimates.
    """
    return {doc_id: float(len(ordered_ids) - rank) for rank, doc_id in enumerate(ordered_ids)}


def sorted_candidates(results: dict[str, float], limit: int) -> list[tuple[str, float]]:
    """Return (doc_id, score) pairs by descending score, ties broken by doc_id."""
    return sorted(results.items(), key=lambda item: (-item[1], item[0]))[:limit]


def merge_head_and_tail(
    original: list[tuple[str, float]], scored_head: list[tuple[str, float]]
) -> dict[str, float]:
    """Order the scored head by its new scores, then append the unscored tail in its old order.

    Cross-encoder scores can be negative, so they must not be mixed with the tail's
    RRF scores: a positive RRF score would sort an unscored tail document above a
    negatively scored head document.
    """
    head_ids = [doc_id for doc_id, _ in sorted(scored_head, key=lambda item: (-item[1], item[0]))]
    tail_ids = [doc_id for doc_id, _ in original[len(scored_head) :]]
    return rank_scores(head_ids + tail_ids)


def run_cross_encoder(
    data: ESCIData,
    hybrid_results: dict[str, dict[str, float]],
    top_k_rerank: int = TOP_K_RERANK_CE,
    top_k_output: int = TOP_K_RETRIEVAL,
) -> dict[str, dict[str, float]]:
    """Rerank hybrid results with a cross-encoder.

    Returns:
        {query_id: {doc_id: rank_score}}, with the scored head ahead of the unscored tail.
    """
    if top_k_rerank < 0:
        raise ValueError("top_k_rerank must be non-negative")

    console.print(
        f"\n[bold cyan]Stage 2: Cross-Encoder Reranking ({CROSS_ENCODER_MODEL})[/bold cyan]"
    )
    console.print("  Loading cross-encoder model...")
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    console.print(f"  Reranking top-{top_k_rerank} candidates per query...")

    reranked_results: dict[str, dict[str, float]] = {}
    start = time.time()

    for query_id, query_text in tqdm(data.queries.items(), desc="  Reranking"):
        if query_id not in hybrid_results:
            continue

        original = sorted_candidates(hybrid_results[query_id], top_k_output)
        candidates = original[:top_k_rerank]

        # Form (query, document) pairs for joint encoding
        doc_ids = [doc_id for doc_id, _ in candidates]
        pairs = [[query_text, data.corpus[doc_id][:2048]] for doc_id in doc_ids]

        scores = model.predict(pairs, batch_size=64, show_progress_bar=False) if pairs else []
        if len(scores) != len(doc_ids) or not all(isfinite(s) for s in scores):
            raise ValueError("Expected one finite score per candidate")

        reranked_results[query_id] = merge_head_and_tail(original, list(zip(doc_ids, scores)))

    elapsed = time.time() - start
    console.print(
        f"  {len(reranked_results):,} queries reranked in {elapsed:.1f}s "
        f"(avg {elapsed / max(len(reranked_results), 1) * 1000:.0f}ms/query)"
    )
    return reranked_results
