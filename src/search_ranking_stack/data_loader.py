"""
Data loader for SciFact dataset.
Loads corpus, queries, and relevance judgments from HuggingFace MTEB collection.

Note: Uses mteb/scifact instead of BeIR/scifact since the latter uses
deprecated dataset scripts that are no longer supported by datasets>=4.5.
"""

from dataclasses import dataclass

from datasets import load_dataset
from rich.console import Console
from tqdm import tqdm

from .config import SCIFACT_DIR

console = Console()


@dataclass
class BEIRData:
    """Container for BEIR dataset components."""

    corpus: dict[str, str]  # doc_id → full text (title + text)
    queries: dict[str, str]  # query_id → query text
    qrels: dict[str, dict[str, int]]  # query_id → {doc_id: relevance}


def load_scifact(cache_dir: str | None = None) -> BEIRData:
    """
    Load the SciFact dataset from HuggingFace MTEB collection.

    Uses the MTEB version which provides Parquet format compatible with
    modern datasets library versions.

    Args:
        cache_dir: Optional cache directory for datasets

    Returns:
        BEIRData containing corpus, queries, and qrels
    """
    cache_dir = cache_dir or str(SCIFACT_DIR)

    console.print("[bold blue]Loading SciFact dataset...[/bold blue]")

    # Load corpus from mteb/scifact
    console.print("  Loading corpus...")
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus", cache_dir=cache_dir)

    corpus = {}
    for doc in tqdm(corpus_ds, desc="  Processing documents"):
        doc_id = str(doc["_id"])
        # Concatenate title and text
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        corpus[doc_id] = f"{title} {text}".strip()

    console.print(f"  [green]✓[/green] Loaded {len(corpus):,} documents")

    # Load test queries - mteb/scifact has queries in a different structure
    console.print("  Loading queries...")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries", cache_dir=cache_dir)

    queries = {}
    for q in queries_ds:
        query_id = str(q["_id"])
        queries[query_id] = q["text"]

    console.print(f"  [green]✓[/green] Loaded {len(queries):,} queries")

    # Load qrels - the default split contains test qrels
    console.print("  Loading relevance judgments...")
    qrels_ds = load_dataset("mteb/scifact", "default", split="test", cache_dir=cache_dir)

    qrels: dict[str, dict[str, int]] = {}
    for item in qrels_ds:
        query_id = str(item["query-id"])
        corpus_id = str(item["corpus-id"])
        score = int(item["score"])

        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][corpus_id] = score

    console.print(f"  [green]✓[/green] Loaded relevance judgments for {len(qrels):,} queries")

    # Filter queries to only those with relevance judgments
    test_queries = {qid: text for qid, text in queries.items() if qid in qrels}
    console.print(f"  [green]✓[/green] Using {len(test_queries):,} test queries with judgments")

    return BEIRData(corpus=corpus, queries=test_queries, qrels=qrels)
