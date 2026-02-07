"""
Stage 3: LLM Listwise Reranking

Uses an LLM to perform listwise reranking of the top-10 results.
Shows the frontier — LLM-as-a-reranker for maximum precision.

Three modes:
- Mode A: Local model (Qwen2.5-1.5B-Instruct) - requires [llm] extras
- Mode B: Ollama (any local model) - requires Ollama running
- Mode C: Claude API - requires [api] extras

Blog Section: 11.3 - LLM Rerankers
"""

import re

from rich.console import Console

from ..config import (
    LLM_MODEL_API,
    LLM_MODEL_LOCAL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K_RERANK_LLM,
    TOP_K_RETRIEVAL,
)
from ..data_loader import BEIRData

console = Console()


def llm_available() -> str | None:
    """
    Check if LLM reranking is available.

    Returns:
        "local" if local LLM (transformers) is available
        "ollama" if Ollama is running
        "api" if Anthropic API is available
        None if none are available
    """
    # Check for local LLM support (HuggingFace transformers)
    try:
        import accelerate  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return "local"
    except ImportError:
        pass

    # Check for Ollama
    try:
        import httpx

        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        if response.status_code == 200:
            return "ollama"
    except Exception:
        pass

    # Check for API support
    try:
        import anthropic  # noqa: F401

        return "api"
    except ImportError:
        pass

    return None


def _create_listwise_prompt(
    query: str, documents: list[tuple[str, str]], max_words: int = 200
) -> str:
    """
    Create a listwise ranking prompt inspired by RankGPT.

    Args:
        query: The search query
        documents: List of (doc_id, doc_text) tuples
        max_words: Max words per document

    Returns:
        The formatted prompt
    """
    n = len(documents)

    doc_texts = []
    for i, (doc_id, doc_text) in enumerate(documents, start=1):
        # Truncate to max_words
        words = doc_text.split()[:max_words]
        truncated = " ".join(words)
        doc_texts.append(f"[{i}] {truncated}")

    docs_formatted = "\n\n".join(doc_texts)

    prompt = f"""I will provide you with {n} passages, each indicated by a numerical identifier \
[1] to [{n}]. Rank the passages based on their relevance to the search query: "{query}"

{docs_formatted}

Rank the passages from most relevant to least relevant.
Output ONLY a comma-separated list of passage identifiers, e.g.: [3], [1], [2], ...
Do not explain your reasoning. Only output the ranking."""

    return prompt


def _parse_ranking(output: str, n: int) -> list[int] | None:
    """
    Parse LLM output to extract ranking order.

    Args:
        output: Raw LLM output
        n: Expected number of items

    Returns:
        List of 0-indexed positions, or None if parsing fails
    """
    # Extract all [N] patterns
    matches = re.findall(r"\[(\d+)\]", output)

    if not matches:
        return None

    try:
        # Convert to 0-indexed positions
        positions = [int(m) - 1 for m in matches]

        # Validate
        if len(positions) < n:
            # Pad with remaining positions in order
            seen = set(positions)
            for i in range(n):
                if i not in seen:
                    positions.append(i)

        return positions[:n]

    except (ValueError, IndexError):
        return None


def _run_local_llm(
    data: BEIRData,
    ce_results: dict[str, dict[str, float]],
    top_k_rerank: int,
    top_k_output: int,
) -> dict[str, dict[str, float]]:
    """Run reranking with local Qwen model."""
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print(f"  Loading local model: {LLM_MODEL_LOCAL}...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_LOCAL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_LOCAL,
        torch_dtype="auto",
        device_map="auto",
    )

    reranked_results: dict[str, dict[str, float]] = {}
    parse_failures = 0

    for query_id, query_text in tqdm(data.queries.items(), desc="  LLM Reranking"):
        if query_id not in ce_results:
            continue

        # Get top-k for LLM reranking
        candidates = list(ce_results[query_id].items())[:top_k_rerank]
        documents = [(doc_id, data.corpus.get(doc_id, "")) for doc_id, _ in candidates]

        # Generate prompt
        prompt = _create_listwise_prompt(query_text, documents)

        # Generate
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Parse ranking
        ranking = _parse_ranking(response, len(documents))

        if ranking is None:
            parse_failures += 1
            # Fall back to cross-encoder order
            reranked = dict(candidates[:top_k_output])
        else:
            # Apply new ranking
            reranked = {}
            for new_rank, old_idx in enumerate(ranking):
                if old_idx < len(candidates):
                    doc_id, _ = candidates[old_idx]
                    # Score: higher rank = higher score
                    reranked[doc_id] = float(top_k_rerank - new_rank)

        # Add remaining docs from original results
        remaining = list(ce_results[query_id].items())[top_k_rerank:top_k_output]
        for doc_id, score in remaining:
            if doc_id not in reranked:
                reranked[doc_id] = float(score) * 0.01

        reranked_results[query_id] = reranked

    console.print(
        f"  {len(data.queries):,}/{len(data.queries):,} queries processed "
        f"({len(data.queries) - parse_failures} parsed successfully, "
        f"{parse_failures} fell back to CE order)"
    )

    return reranked_results


def _run_api_llm(
    data: BEIRData,
    ce_results: dict[str, dict[str, float]],
    top_k_rerank: int,
    top_k_output: int,
) -> dict[str, dict[str, float]]:
    """Run reranking with Claude API."""
    import anthropic
    from tqdm import tqdm

    console.print(f"  Using Claude API: {LLM_MODEL_API}...")

    client = anthropic.Anthropic()

    reranked_results: dict[str, dict[str, float]] = {}
    parse_failures = 0

    for query_id, query_text in tqdm(data.queries.items(), desc="  LLM Reranking"):
        if query_id not in ce_results:
            continue

        # Get top-k for LLM reranking
        candidates = list(ce_results[query_id].items())[:top_k_rerank]
        documents = [(doc_id, data.corpus.get(doc_id, "")) for doc_id, _ in candidates]

        # Generate prompt
        prompt = _create_listwise_prompt(query_text, documents)

        try:
            response = client.messages.create(
                model=LLM_MODEL_API,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            output = response.content[0].text

            # Parse ranking
            ranking = _parse_ranking(output, len(documents))

            if ranking is None:
                parse_failures += 1
                reranked = dict(candidates[:top_k_output])
            else:
                reranked = {}
                for new_rank, old_idx in enumerate(ranking):
                    if old_idx < len(candidates):
                        doc_id, _ = candidates[old_idx]
                        reranked[doc_id] = float(top_k_rerank - new_rank)

        except Exception as e:
            console.print(f"    [yellow]API error for query {query_id}: {e}[/yellow]")
            parse_failures += 1
            reranked = dict(candidates[:top_k_output])

        # Add remaining docs
        remaining = list(ce_results[query_id].items())[top_k_rerank:top_k_output]
        for doc_id, score in remaining:
            if doc_id not in reranked:
                reranked[doc_id] = float(score) * 0.01

        reranked_results[query_id] = reranked

    console.print(
        f"  {len(data.queries):,}/{len(data.queries):,} queries processed "
        f"({len(data.queries) - parse_failures} parsed successfully, "
        f"{parse_failures} fell back to CE order)"
    )

    return reranked_results


def _run_ollama_llm(
    data: BEIRData,
    ce_results: dict[str, dict[str, float]],
    top_k_rerank: int,
    top_k_output: int,
) -> dict[str, dict[str, float]]:
    """Run reranking with Ollama local model."""
    import httpx
    from tqdm import tqdm

    console.print(f"  Using Ollama model: {OLLAMA_MODEL}...")

    reranked_results: dict[str, dict[str, float]] = {}
    parse_failures = 0

    for query_id, query_text in tqdm(data.queries.items(), desc="  LLM Reranking"):
        if query_id not in ce_results:
            continue

        # Get top-k for LLM reranking
        candidates = list(ce_results[query_id].items())[:top_k_rerank]
        documents = [(doc_id, data.corpus.get(doc_id, "")) for doc_id, _ in candidates]

        # Generate prompt
        prompt = _create_listwise_prompt(query_text, documents)

        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 100,
                    },
                },
                timeout=60.0,
            )
            response.raise_for_status()
            output = response.json().get("response", "")

            # Parse ranking
            ranking = _parse_ranking(output, len(documents))

            if ranking is None:
                parse_failures += 1
                reranked = dict(candidates[:top_k_output])
            else:
                reranked = {}
                for new_rank, old_idx in enumerate(ranking):
                    if old_idx < len(candidates):
                        doc_id, _ = candidates[old_idx]
                        reranked[doc_id] = float(top_k_rerank - new_rank)

        except Exception as e:
            console.print(f"    [yellow]Ollama error for query {query_id}: {e}[/yellow]")
            parse_failures += 1
            reranked = dict(candidates[:top_k_output])

        # Add remaining docs
        remaining = list(ce_results[query_id].items())[top_k_rerank:top_k_output]
        for doc_id, score in remaining:
            if doc_id not in reranked:
                reranked[doc_id] = float(score) * 0.01

        reranked_results[query_id] = reranked

    console.print(
        f"  {len(data.queries):,}/{len(data.queries):,} queries processed "
        f"({len(data.queries) - parse_failures} parsed successfully, "
        f"{parse_failures} fell back to CE order)"
    )

    return reranked_results


def run_llm_rerank(
    data: BEIRData,
    ce_results: dict[str, dict[str, float]],
    top_k_rerank: int = TOP_K_RERANK_LLM,
    top_k_output: int = TOP_K_RETRIEVAL,
) -> dict[str, dict[str, float]]:
    """
    Rerank top results using an LLM with listwise ranking.

    Args:
        data: BEIRData containing corpus
        ce_results: Results from cross-encoder reranking
        top_k_rerank: Number of candidates for LLM to rerank (default 10)
        top_k_output: Number of results to return per query

    Returns:
        reranked_results: {query_id: {doc_id: llm_score}}
    """
    mode = llm_available()

    if mode == "local":
        console.print(
            f"\n[bold cyan]Stage 3: LLM Listwise Reranking ({LLM_MODEL_LOCAL})[/bold cyan]"
        )
        return _run_local_llm(data, ce_results, top_k_rerank, top_k_output)

    elif mode == "ollama":
        console.print(f"\n[bold cyan]Stage 3: LLM Listwise Reranking ({OLLAMA_MODEL})[/bold cyan]")
        return _run_ollama_llm(data, ce_results, top_k_rerank, top_k_output)

    elif mode == "api":
        console.print(f"\n[bold cyan]Stage 3: LLM Listwise Reranking ({LLM_MODEL_API})[/bold cyan]")
        return _run_api_llm(data, ce_results, top_k_rerank, top_k_output)

    else:
        console.print("\n[yellow]Stage 3: LLM Reranking skipped[/yellow]")
        console.print("  Enable LLM reranking with one of:")
        console.print("    uv sync --extra llm      # Local HuggingFace model")
        console.print("    ollama pull qwen2.5:7b   # Ollama local model")
        console.print("    uv sync --extra api      # Claude API")
        return {}
