"""
ESCI Dataset Download & Sampling Script.

Entry point: `uv run download-data`

Downloads the Amazon ESCI (Shopping Queries Dataset) from HuggingFace and samples
a laptop-friendly subset (~500 queries, ~8.5K products, ~12K judgments).

Source: HuggingFace `tasksource/esci` — pre-joined version of Amazon's Shopping
Queries Dataset that merges examples + product metadata.

Blog Section: Data Pipeline
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from rich.console import Console

from ..config import (
    ESCI_DIR,
    ESCI_HF_DATASET,
    ESCI_LABEL_MAP,
    ESCI_LOCALE,
    ESCI_SEED,
    ESCI_TARGET_QUERIES,
)

console = Console()


def build_product_text(row: pd.Series) -> str:
    """Build composite product text for search indexing."""
    parts = []
    if pd.notna(row.get("product_title")):
        parts.append(str(row["product_title"]))
    if pd.notna(row.get("product_brand")):
        parts.append(f"Brand: {row['product_brand']}")
    if pd.notna(row.get("product_bullet_point")):
        parts.append(str(row["product_bullet_point"]))
    if pd.notna(row.get("product_description")):
        parts.append(str(row["product_description"])[:500])  # Truncate long descriptions
    return " ".join(parts)


def main():
    """Download and sample the ESCI dataset."""
    console.print("[bold]Downloading Amazon ESCI dataset...[/bold]")
    console.print("This will download from HuggingFace and sample a subset.\n")

    output_dir = Path(ESCI_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load from HuggingFace (streams ~2.5GB, caches locally)
    console.print("Loading Amazon ESCI dataset from HuggingFace...")
    ds = load_dataset(ESCI_HF_DATASET, split="train")  # tasksource/esci has single split
    df = ds.to_pandas()

    # 2. Filter to English locale
    df = df[df["product_locale"] == ESCI_LOCALE].copy()
    console.print(f"  English-locale rows: {len(df):,}")

    # 3. Filter to "hard" queries (small_version flag)
    if "small_version" in df.columns:
        df = df[df["small_version"] == 1].copy()
        console.print(f"  After small_version filter: {len(df):,}")

    # 4. Sample queries
    unique_queries = df["query_id"].unique()
    n_sample = min(ESCI_TARGET_QUERIES, len(unique_queries))

    rng = np.random.default_rng(ESCI_SEED)
    sampled_query_ids = rng.choice(unique_queries, size=n_sample, replace=False)

    df_sample = df[df["query_id"].isin(sampled_query_ids)].copy()
    console.print(f"  Sampled {n_sample} queries → {len(df_sample):,} (query, product) pairs")

    # 5. Build corpus (unique products)
    product_cols = [
        "product_id",
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
    ]
    available_cols = [c for c in product_cols if c in df_sample.columns]
    corpus_df = df_sample[available_cols].drop_duplicates(subset=["product_id"])

    # Build composite product_text for search
    corpus_df = corpus_df.copy()
    corpus_df["product_text"] = corpus_df.apply(build_product_text, axis=1)
    console.print(f"  Unique products in sample: {len(corpus_df):,}")

    # 6. Build queries
    queries_df = df_sample[["query_id", "query"]].drop_duplicates(subset=["query_id"])

    # 7. Build qrels with graded relevance
    qrels_df = df_sample[["query_id", "product_id", "esci_label"]].copy()
    qrels_df["relevance"] = qrels_df["esci_label"].map(ESCI_LABEL_MAP)

    # 8. Print label distribution (sanity check)
    label_dist = qrels_df["esci_label"].value_counts()
    console.print("\n  Label distribution in sample:")
    for label, count in label_dist.items():
        pct = count / len(qrels_df) * 100
        gain = ESCI_LABEL_MAP.get(label, 0)
        console.print(f"    {label:12s} (gain={gain}): {count:6,} ({pct:.1f}%)")

    # 9. Save as JSONL
    corpus_df.to_json(output_dir / "corpus.jsonl", orient="records", lines=True)
    queries_df.to_json(output_dir / "queries.jsonl", orient="records", lines=True)
    qrels_df[["query_id", "product_id", "relevance"]].to_json(
        output_dir / "qrels.jsonl", orient="records", lines=True
    )

    console.print(f"\n[bold green]✓ Saved to {output_dir}/[/bold green]")
    console.print(f"  corpus.jsonl:  {len(corpus_df):,} products")
    console.print(f"  queries.jsonl: {len(queries_df):,} queries")
    console.print(f"  qrels.jsonl:   {len(qrels_df):,} judgments")


if __name__ == "__main__":
    main()
