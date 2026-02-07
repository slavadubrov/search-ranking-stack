"""
Stage 1c: Hybrid Search with Reciprocal Rank Fusion (RRF)

Combines BM25 and Dense retrieval results using RRF.
This is the core blog thesis — hybrid search outperforms either method alone.

Algorithm: RRF_score(d) = Σ 1/(k + rank(d, r)) for each ranking r
Where k = 60 (Cormack et al. 2009)

Blog Section: 5 - Hybrid Retrieval
"""

import time
from collections import defaultdict

from rich.console import Console

from ..config import RRF_K, TOP_K_RETRIEVAL

console = Console()


def reciprocal_rank_fusion(
    ranked_lists: list[dict[str, dict[str, float]]],
    k: int = RRF_K,
    top_k: int = TOP_K_RETRIEVAL,
) -> dict[str, dict[str, float]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF is rank-based, not score-based. This makes it robust to
    different score scales across retrieval methods.

    Args:
        ranked_lists: List of result dicts, each {query_id: {doc_id: score}}
        k: RRF constant (default 60 from original paper)
        top_k: Number of results to return per query

    Returns:
        fused_results: {query_id: {doc_id: rrf_score}}
    """
    # Get all query IDs
    all_query_ids = set()
    for results in ranked_lists:
        all_query_ids.update(results.keys())

    fused_results: dict[str, dict[str, float]] = {}

    for query_id in all_query_ids:
        # Collect RRF scores for this query
        rrf_scores: dict[str, float] = defaultdict(float)

        for results in ranked_lists:
            if query_id not in results:
                continue

            # Convert scores to ranks (1-indexed)
            query_results = results[query_id]
            sorted_docs = sorted(query_results.items(), key=lambda x: x[1], reverse=True)

            for rank, (doc_id, _score) in enumerate(sorted_docs, start=1):
                # RRF formula: 1 / (k + rank)
                rrf_scores[doc_id] += 1.0 / (k + rank)

        # Sort by RRF score and take top-k
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        fused_results[query_id] = dict(sorted_rrf)

    return fused_results


def run_hybrid_rrf(
    bm25_results: dict[str, dict[str, float]],
    dense_results: dict[str, dict[str, float]],
    k: int = RRF_K,
    top_k: int = TOP_K_RETRIEVAL,
) -> dict[str, dict[str, float]]:
    """
    Run hybrid search by fusing BM25 and Dense results with RRF.

    Args:
        bm25_results: Results from BM25 retrieval
        dense_results: Results from dense bi-encoder retrieval
        k: RRF constant
        top_k: Number of results per query

    Returns:
        hybrid_results: {query_id: {doc_id: rrf_score}}
    """
    console.print(f"\n[bold cyan]Stage 1c: Hybrid Search (RRF Fusion, k={k})[/bold cyan]")

    console.print(f"  Fusing BM25 + Dense results for {len(bm25_results):,} queries...", end=" ")
    start = time.time()

    hybrid_results = reciprocal_rank_fusion([bm25_results, dense_results], k=k, top_k=top_k)

    fusion_time = time.time() - start
    console.print(f"done ({fusion_time:.2f}s)")

    # Show example of RRF benefit
    _log_rrf_example(bm25_results, dense_results, hybrid_results)

    return hybrid_results


def _log_rrf_example(
    bm25_results: dict[str, dict[str, float]],
    dense_results: dict[str, dict[str, float]],
    hybrid_results: dict[str, dict[str, float]],
):
    """Log an example showing RRF rescuing documents."""
    # Find a query where hybrid found docs that weren't in top-10 of either method
    for query_id in list(hybrid_results.keys())[:50]:
        bm25_top10 = set(list(bm25_results.get(query_id, {}).keys())[:10])
        dense_top10 = set(list(dense_results.get(query_id, {}).keys())[:10])
        hybrid_top10 = set(list(hybrid_results[query_id].keys())[:10])

        # Find docs unique to hybrid top-10
        rescued = hybrid_top10 - (bm25_top10 | dense_top10)

        if rescued:
            console.print(f"\n  [dim]Example — Query {query_id}:[/dim]")
            console.print(
                f"    [dim]Hybrid rescued {len(rescued)} doc(s) into top-10 "
                f"that weren't in either individual top-10[/dim]"
            )
            break
