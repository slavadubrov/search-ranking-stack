"""
Stage 3: LLM Listwise Reranking

Asks an LLM to order the cross-encoder's top 10 (inspired by RankGPT). The demo
accepts an answer only if it names every candidate exactly once; otherwise it keeps
the cross-encoder order for that query and counts a fallback.

Three modes (selected via --llm-mode):
- local: HuggingFace model (Qwen2.5-1.5B-Instruct) - requires `uv sync --extra llm`
- ollama: Any Ollama model - requires Ollama running
- api: Claude API - requires `uv sync --extra api` and ANTHROPIC_API_KEY

Article: https://slavadubrov.github.io/blog/2026/02/08/search-ranking-stack/
Section: LLM listwise reranking
"""

import re
from collections.abc import Callable

from rich.console import Console
from tqdm import tqdm

from ..config import (
    LLM_MODEL_API,
    LLM_MODEL_LOCAL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K_RERANK_LLM,
    TOP_K_RETRIEVAL,
)
from ..data_loader import ESCIData
from .s04_cross_encoder import rank_scores, sorted_candidates

console = Console()


def _create_listwise_prompt(
    query: str, documents: list[tuple[str, str]], max_words: int = 200
) -> str:
    """Create a listwise ranking prompt inspired by RankGPT."""
    n = len(documents)

    doc_texts = []
    for i, (_doc_id, doc_text) in enumerate(documents, start=1):
        words = doc_text.split()[:max_words]
        doc_texts.append(f"[{i}] {' '.join(words)}")

    docs_formatted = "\n\n".join(doc_texts)

    return (
        f"I will provide you with {n} product listings, each indicated by a numerical "
        f"identifier [1] to [{n}]. Rank the products based on their relevance to the "
        f'search query: "{query}"\n\n'
        "Consider:\n"
        "- Exact matches (product satisfies all query requirements) should rank highest\n"
        "- Substitutes (functional alternatives) should rank above complements\n"
        "- Completely irrelevant products should rank lowest\n\n"
        f"{docs_formatted}\n\n"
        "Rank the products from most relevant to least relevant.\n"
        "Output ONLY a comma-separated list of product identifiers, e.g.: [3], [1], [2], ...\n"
        "Do not explain your reasoning. Only output the ranking."
    )


def _parse_ranking(output: str, n: int) -> list[int] | None:
    """Return 0-based positions if the output ranks IDs 1..n exactly once."""
    positions = [int(m) - 1 for m in re.findall(r"\[(\d+)\]", output)]

    # Reject missing, duplicate, and out-of-range IDs.
    if sorted(positions) != list(range(n)):
        return None

    return positions


def _rerank(
    data: ESCIData,
    ce_results: dict[str, dict[str, float]],
    generate: Callable[[str], str],
    top_k_rerank: int,
    top_k_output: int,
) -> tuple[dict[str, dict[str, float]], int]:
    """Run listwise reranking with one backend.

    Returns:
        ({query_id: {doc_id: rank_score}}, number of queries that fell back to CE order)
    """
    reranked_results: dict[str, dict[str, float]] = {}
    fallbacks = 0

    for query_id, query_text in tqdm(data.queries.items(), desc="  LLM Reranking"):
        if query_id not in ce_results:
            continue

        original = sorted_candidates(ce_results[query_id], top_k_output)
        ce_ids = [doc_id for doc_id, _ in original]
        head = ce_ids[:top_k_rerank]
        documents = [(doc_id, data.corpus.get(doc_id, "")) for doc_id in head]

        try:
            ranking = _parse_ranking(
                generate(_create_listwise_prompt(query_text, documents)), len(head)
            )
        except Exception as e:  # network or backend error: count it as a fallback
            console.print(f"    [yellow]LLM error for query {query_id}: {e}[/yellow]")
            ranking = None

        if ranking is None:
            fallbacks += 1
            ordered = ce_ids  # keep the entire cross-encoder order
        else:
            ordered = [head[i] for i in ranking] + ce_ids[top_k_rerank:]

        reranked_results[query_id] = rank_scores(ordered)

    total = len(reranked_results)
    console.print(
        f"  {total:,} queries processed: {total - fallbacks:,} parsed, "
        f"{fallbacks:,} fell back to cross-encoder order "
        f"(fallback rate {fallbacks / max(total, 1):.1%})"
    )
    return reranked_results, fallbacks


def _local_generate() -> Callable[[str], str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print(f"  Loading local model: {LLM_MODEL_LOCAL}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_LOCAL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_LOCAL, torch_dtype="auto", device_map="auto"
    )

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    return generate


def _api_generate() -> Callable[[str], str]:
    import anthropic

    console.print(f"  Using Claude API: {LLM_MODEL_API}...")
    client = anthropic.Anthropic()

    def generate(prompt: str) -> str:
        response = client.messages.create(
            model=LLM_MODEL_API, max_tokens=100, messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    return generate


def _ollama_generate() -> Callable[[str], str]:
    import httpx

    console.print(f"  Using Ollama model: {OLLAMA_MODEL}...")

    def generate(prompt: str) -> str:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 100},
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    return generate


BACKENDS = {
    "local": (LLM_MODEL_LOCAL, _local_generate),
    "ollama": (OLLAMA_MODEL, _ollama_generate),
    "api": (LLM_MODEL_API, _api_generate),
}


def run_llm_rerank(
    data: ESCIData,
    ce_results: dict[str, dict[str, float]],
    mode: str,
    top_k_rerank: int = TOP_K_RERANK_LLM,
    top_k_output: int = TOP_K_RETRIEVAL,
) -> tuple[dict[str, dict[str, float]], int]:
    """Rerank the cross-encoder's top results with an LLM.

    Returns:
        ({query_id: {doc_id: rank_score}}, fallback count)
    """
    if mode not in BACKENDS:
        raise ValueError(f"Unknown LLM mode: {mode}. Valid modes: {', '.join(BACKENDS)}")

    model_name, make_generate = BACKENDS[mode]
    console.print(f"\n[bold cyan]Stage 3: LLM Listwise Reranking ({model_name})[/bold cyan]")
    return _rerank(data, ce_results, make_generate(), top_k_rerank, top_k_output)
