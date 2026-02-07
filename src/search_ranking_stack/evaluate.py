"""
Evaluation module for search ranking.

Computes standard IR metrics using pytrec_eval:
- NDCG@10: Normalized Discounted Cumulative Gain (primary metric)
- MRR@10: Mean Reciprocal Rank (how fast we find first relevant doc)
- Recall@100: Coverage of retrieval stage

ESCI uses graded relevance: Exact=3, Substitute=2, Complement=1, Irrelevant=0

Blog Section: Measuring Search Quality
"""

import numpy as np
import pytrec_eval
from rich.console import Console

from .config import ESCI_LABEL_MAP, EVAL_METRICS

console = Console()

# Reverse mapping for display
ESCI_LABEL_NAMES = {v: k for k, v in ESCI_LABEL_MAP.items()}


def evaluate(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    metrics: set[str] | None = None,
) -> dict[str, float]:
    """
    Evaluate search results against relevance judgments.

    Args:
        qrels: Ground truth relevance judgments
               {query_id: {doc_id: relevance_score}}
        results: Search results to evaluate
                 {query_id: {doc_id: score}}
        metrics: Set of pytrec_eval metric names (default: NDCG@10, MRR, Recall@100)

    Returns:
        Dictionary of metric_name → average_score
    """
    if metrics is None:
        metrics = EVAL_METRICS

    # Filter qrels to only queries that have results
    filtered_qrels = {qid: rels for qid, rels in qrels.items() if qid in results}

    # Create evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(filtered_qrels, metrics)

    # Evaluate
    per_query_scores = evaluator.evaluate(results)

    # Compute averages
    avg_scores: dict[str, float] = {}
    for metric in metrics:
        # pytrec_eval uses slightly different key names
        # e.g., ndcg_cut_10 stays as ndcg_cut_10
        scores = [s[metric] for s in per_query_scores.values() if metric in s]
        if scores:
            avg_scores[metric] = float(np.mean(scores))
        else:
            avg_scores[metric] = 0.0

    return avg_scores


def analyze_label_ranking(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    top_k: int = 10,
) -> dict[str, float]:
    """
    Analyze what fraction of top-K results are Exact/Substitute/Complement/Irrelevant.

    This is an ESCI-specific analysis that shows how each stage progressively
    pushes Exact matches to the top.

    Args:
        qrels: Ground truth with graded relevance (E=3, S=2, C=1, I=0)
        results: Search results {query_id: {doc_id: score}}
        top_k: Number of top results to analyze per query

    Returns:
        Dictionary with label names as keys and fractions as values
    """
    counts = {"Exact": 0, "Substitute": 0, "Complement": 0, "Irrelevant": 0}
    total = 0

    for qid, ranking in results.items():
        if qid not in qrels:
            continue

        # Sort by score descending and take top-k
        sorted_docs = sorted(ranking.items(), key=lambda x: -x[1])[:top_k]

        for pid, _ in sorted_docs:
            rel = qrels.get(qid, {}).get(pid, 0)
            label_name = ESCI_LABEL_NAMES.get(rel, "Irrelevant")
            counts[label_name] += 1
            total += 1

    if total == 0:
        return {name: 0.0 for name in counts}

    return {name: count / total for name, count in counts.items()}


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics for console output."""
    parts = []
    for name, value in metrics.items():
        # Clean up metric name for display
        display_name = name.replace("_cut_", "@").replace("_", " ").upper()
        if display_name == "RECIP RANK":
            display_name = "MRR@10"
        elif display_name == "RECALL 100":
            display_name = "Recall@100"
        parts.append(f"{display_name}: {value:.4f}")
    return " | ".join(parts)


def print_metrics(stage_name: str, metrics: dict[str, float]):
    """Print metrics with formatting."""
    console.print(f"  [green]→[/green] {format_metrics(metrics)}")
