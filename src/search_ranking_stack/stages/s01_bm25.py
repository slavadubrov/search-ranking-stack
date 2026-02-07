"""
Stage 1a: BM25 Sparse Retrieval

Establishes the lexical baseline using the Okapi BM25 algorithm.
Shows where keyword matching works and where it fails.

Blog Section: 3.1 - Sparse Retrievers (Term-based)
"""

import time

import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console
from tqdm import tqdm

from ..config import TOP_K_RETRIEVAL
from ..data_loader import BEIRData

console = Console()


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace tokenizer.

    Note: Production systems use more sophisticated tokenization
    (e.g., NLTK, spaCy, or language-specific stemmers).
    Simple whitespace + lowercase is sufficient for this demo.
    """
    return text.lower().split()


def run_bm25(data: BEIRData, top_k: int = TOP_K_RETRIEVAL) -> dict[str, dict[str, float]]:
    """
    Run BM25 retrieval over the corpus.

    Args:
        data: BEIRData containing corpus, queries, qrels
        top_k: Number of documents to retrieve per query

    Returns:
        results: {query_id: {doc_id: bm25_score}}
    """
    console.print("\n[bold cyan]Stage 1a: BM25 Retrieval[/bold cyan]")

    # Build index
    console.print(f"  Indexing {len(data.corpus):,} documents...", end=" ")
    start = time.time()

    doc_ids = list(data.corpus.keys())
    doc_texts = [data.corpus[doc_id] for doc_id in doc_ids]
    tokenized_corpus = [tokenize(text) for text in doc_texts]

    bm25 = BM25Okapi(tokenized_corpus)
    index_time = time.time() - start
    console.print(f"done ({index_time:.1f}s)")

    # Retrieve for each query
    console.print(f"  Retrieving for {len(data.queries):,} queries...", end=" ")
    start = time.time()

    results: dict[str, dict[str, float]] = {}

    for query_id, query_text in tqdm(data.queries.items(), desc="  Querying", leave=False):
        tokenized_query = tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)

        # Get top-k documents
        top_indices = np.argsort(scores)[::-1][:top_k]
        results[query_id] = {doc_ids[idx]: float(scores[idx]) for idx in top_indices}

    retrieval_time = time.time() - start
    console.print(f"done ({retrieval_time:.1f}s)")

    return results
