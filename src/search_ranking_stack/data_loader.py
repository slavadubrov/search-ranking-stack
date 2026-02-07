"""
Data loader for Amazon ESCI dataset.

Loads corpus, queries, and relevance judgments from sampled JSONL files.
The ESCI dataset uses 4-level graded relevance:
  - Exact (E) = 3: Product satisfies all query requirements
  - Substitute (S) = 2: Functional alternative
  - Complement (C) = 1: Related but not what user wants
  - Irrelevant (I) = 0: No relevance to query

Blog Section: Data Pipeline
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from .config import ESCI_DIR

console = Console()


@dataclass
class ESCIData:
    """Container for ESCI dataset components."""

    corpus: dict[str, str]  # product_id → product_text
    corpus_meta: dict[str, dict]  # product_id → {title, brand, color, ...}
    queries: dict[str, str]  # query_id → query text
    qrels: dict[str, dict[str, int]]  # query_id → {product_id: graded_relevance}


def load_esci(data_dir: str | Path | None = None) -> ESCIData:
    """
    Load the sampled ESCI dataset from JSONL files.

    Args:
        data_dir: Path to ESCI sample directory (default: data/esci_sample)

    Returns:
        ESCIData containing corpus, corpus_meta, queries, and qrels

    Raises:
        FileNotFoundError: If dataset not downloaded yet
    """
    data_dir = Path(data_dir) if data_dir else ESCI_DIR

    if not (data_dir / "corpus.jsonl").exists():
        raise FileNotFoundError(f"Dataset not found at {data_dir}. Run: uv run download-data")

    console.print("[bold blue]Loading ESCI dataset...[/bold blue]")

    # Load corpus
    console.print("  Loading corpus...")
    corpus: dict[str, str] = {}
    corpus_meta: dict[str, dict] = {}

    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            pid = str(row["product_id"])
            corpus[pid] = row.get("product_text", "")
            corpus_meta[pid] = {
                k: row.get(k) for k in ["product_title", "product_brand", "product_color"]
            }

    console.print(f"  [green]✓[/green] Loaded {len(corpus):,} products")

    # Load queries
    console.print("  Loading queries...")
    queries: dict[str, str] = {}

    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["query_id"])] = row["query"]

    console.print(f"  [green]✓[/green] Loaded {len(queries):,} queries")

    # Load qrels (graded: E=3, S=2, C=1, I=0)
    console.print("  Loading relevance judgments...")
    qrels: dict[str, dict[str, int]] = {}

    with open(data_dir / "qrels.jsonl") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row["query_id"])
            pid = str(row["product_id"])
            rel = int(row["relevance"])
            qrels.setdefault(qid, {})[pid] = rel

    total_judgments = sum(len(v) for v in qrels.values())
    console.print(
        f"  [green]✓[/green] Loaded {total_judgments:,} judgments for {len(qrels):,} queries"
    )

    return ESCIData(corpus=corpus, corpus_meta=corpus_meta, queries=queries, qrels=qrels)
