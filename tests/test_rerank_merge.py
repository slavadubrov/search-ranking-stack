from search_ranking_stack.stages.s04_cross_encoder import merge_head_and_tail
from search_ranking_stack.stages.s05_llm_rerank import _parse_ranking


def test_negative_head_scores_stay_above_unscored_tail():
    original = [("a", 0.03), ("b", 0.02), ("c", 0.016), ("d", 0.01)]
    scores = merge_head_and_tail(original, [("a", -3.2), ("b", 4.1)])
    assert sorted(scores, key=scores.get, reverse=True) == ["b", "a", "c", "d"]


def test_parser_rejects_missing_duplicate_and_out_of_range_ids():
    assert _parse_ranking("[2], [1], [3]", 3) == [1, 0, 2]
    assert _parse_ranking("[2], [1]", 3) is None
    assert _parse_ranking("[2], [2], [1]", 3) is None
    assert _parse_ranking("[2], [1], [4]", 3) is None
    assert _parse_ranking("no ranking", 3) is None
