"""
Orchestrator: Run the full search ranking pipeline.

Entry point: `uv run run-all`

Runs all stages sequentially:
1a. BM25 retrieval
1b. Dense bi-encoder retrieval
1c. Hybrid RRF fusion
2.  Cross-encoder reranking
3.  LLM listwise reranking (optional)

Then evaluates and visualizes results.
"""

from rich.console import Console
from rich.panel import Panel

from .data_loader import load_scifact
from .evaluate import evaluate, print_metrics
from .stages import (
    llm_available,
    run_bm25,
    run_cross_encoder,
    run_dense,
    run_hybrid_rrf,
    run_llm_rerank,
)
from .visualize import plot_comparison, print_table, save_metrics

console = Console()


def main():
    """Run the complete search ranking pipeline."""
    console.print(
        Panel.fit(
            "[bold cyan]Search Ranking Stack[/bold cyan]\n"
            "Multi-stage search ranking demo on SciFact benchmark",
            border_style="cyan",
        )
    )

    # 1. Load data
    data = load_scifact()

    # 2. Run stages sequentially
    console.print("\n[bold]Running retrieval and reranking stages...[/bold]")

    # Stage 1a: BM25
    bm25_results = run_bm25(data)
    bm25_metrics = evaluate(data.qrels, bm25_results)
    print_metrics("BM25", bm25_metrics)

    # Stage 1b: Dense
    dense_results = run_dense(data)
    dense_metrics = evaluate(data.qrels, dense_results)
    print_metrics("Dense", dense_metrics)

    # Stage 1c: Hybrid RRF
    hybrid_results = run_hybrid_rrf(bm25_results, dense_results)
    hybrid_metrics = evaluate(data.qrels, hybrid_results)
    print_metrics("Hybrid RRF", hybrid_metrics)

    # Stage 2: Cross-encoder
    ce_results = run_cross_encoder(data, hybrid_results)
    ce_metrics = evaluate(data.qrels, ce_results)
    print_metrics("Cross-Encoder", ce_metrics)

    # Collect all results
    all_results = {
        "BM25": bm25_results,
        "Dense Bi-Encoder": dense_results,
        "Hybrid (RRF)": hybrid_results,
        "+ Cross-Encoder": ce_results,
    }
    all_metrics = {
        "BM25": bm25_metrics,
        "Dense Bi-Encoder": dense_metrics,
        "Hybrid (RRF)": hybrid_metrics,
        "+ Cross-Encoder": ce_metrics,
    }

    # Stage 3: LLM (optional)
    if llm_available():
        llm_results = run_llm_rerank(data, ce_results)
        if llm_results:
            llm_metrics = evaluate(data.qrels, llm_results)
            print_metrics("LLM Reranker", llm_metrics)
            all_results["+ LLM Reranker"] = llm_results
            all_metrics["+ LLM Reranker"] = llm_metrics

    # 3. Visualize and save
    console.print("\n[bold]Generating results...[/bold]")

    save_metrics(all_metrics)
    plot_comparison(all_metrics)
    print_table(all_metrics)

    # 4. Summary
    console.print("\n[bold green]✓ Pipeline complete![/bold green]")

    # Show key insights
    console.print("\n[bold]Key Insights:[/bold]")

    # Hybrid vs individuals
    hybrid_ndcg = hybrid_metrics.get("ndcg_cut_10", 0)
    bm25_ndcg = bm25_metrics.get("ndcg_cut_10", 0)
    dense_ndcg = dense_metrics.get("ndcg_cut_10", 0)
    max_individual = max(bm25_ndcg, dense_ndcg)

    if hybrid_ndcg > max_individual:
        gain = (hybrid_ndcg - max_individual) * 100
        console.print(
            f"  • Hybrid RRF outperforms best individual method by "
            f"[bold]+{gain:.1f}%[/bold] NDCG@10"
        )

    # Cross-encoder improvement
    ce_ndcg = ce_metrics.get("ndcg_cut_10", 0)
    ce_gain = (ce_ndcg - hybrid_ndcg) * 100
    console.print(f"  • Cross-encoder reranking adds [bold]+{ce_gain:.1f}%[/bold] NDCG@10")

    # Recall stays flat after retrieval
    ce_recall = ce_metrics.get("recall_100", 0)
    console.print(
        f"  • Recall@100 stays flat at [bold]{ce_recall:.1%}[/bold] — "
        "reranking can't fix retrieval misses"
    )

    console.print()


if __name__ == "__main__":
    main()
