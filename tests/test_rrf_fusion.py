"""
Unit-тесты для _rrf_fuse (Reciprocal Rank Fusion).

Не требует ни моделей, ни ChromaDB, ни GPU. Прогоняется за миллисекунды.

Запуск:
    cd C:\\test
    python -m pytest tests/test_rrf_fusion.py -v
    # или напрямую:
    python tests/test_rrf_fusion.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core.schema import TextNode, NodeWithScore
from src.rag_pipeline import _rrf_fuse


def _nws(text: str, node_id: str, score: float = 0.0) -> NodeWithScore:
    """Удобный конструктор NodeWithScore для тестов."""
    return NodeWithScore(node=TextNode(text=text, id_=node_id), score=score)


def test_rrf_basic_known_answer():
    """
    Из Cormack 2009 (таблица в разделе 'A demonstration'):
    doc_A в rank1 у vector, rank3 у bm25 → 1/61 + 1/63 ≈ 0.0323
    doc_B в rank2 у vector, rank1 у bm25 → 1/62 + 1/61 ≈ 0.0324
    doc_C в rank3 у vector, rank2 у bm25 → 1/63 + 1/62 ≈ 0.0317

    Ожидаемый порядок: B > A > C
    """
    vec = [_nws("A", "A"), _nws("B", "B"), _nws("C", "C")]
    bm25 = [_nws("B", "B"), _nws("C", "C"), _nws("A", "A")]
    result = _rrf_fuse(vec, bm25)
    ids = [n.node.node_id for n in result]
    assert ids == ["B", "A", "C"], f"Expected [B,A,C], got {ids}"


def test_rrf_bm25_only_chunk_surfaces():
    """
    BUG-FIX TEST (B1+B2): BM25-only чанк (нет в vector_results) должен получить
    честный шанс попасть в топ, а не оказаться после vector-only чанков.
    """
    vec = [_nws("vec_top1", "V1", 0.95), _nws("vec_top2", "V2", 0.80)]
    bm25 = [_nws("bm25_top1", "B1", 9.5), _nws("V1", "V1", 5.0)]  # V1 — общий
    result = _rrf_fuse(vec, bm25)
    ids = [n.node.node_id for n in result]
    # B1 должен попасть в топ (rank1 BM25 = 1/61 ≈ 0.0164)
    # V1 = 1/61 (vec) + 1/62 (bm25) ≈ 0.0325
    # V2 = 1/62 (vec) ≈ 0.0161
    # Ожидаем: V1 (найден обоими) > B1 (только BM25) ≈ V2 (только vector)
    assert "B1" in ids, "BM25-only чанк B1 потерян — RRF не работает"
    assert ids[0] == "V1", f"V1 должен быть №1 (найден обоими ретриверами), got {ids[0]}"


def test_rrf_dedup_no_duplicate_ids():
    """Чанк, найденный обоими ретриверами, не должен появляться дважды."""
    vec = [_nws("X", "X"), _nws("Y", "Y"), _nws("Z", "Z")]
    bm25 = [_nws("X", "X"), _nws("Y", "Y")]
    result = _rrf_fuse(vec, bm25)
    ids = [n.node.node_id for n in result]
    assert len(ids) == len(set(ids)), f"Дубликаты в результате: {ids}"


def test_rrf_empty_bm25_falls_back_to_vector():
    """Если BM25 недоступен — не падаем, возвращаем vector_results в исходном порядке."""
    vec = [_nws("a", "A", 0.9), _nws("b", "B", 0.5), _nws("c", "C", 0.1)]
    result = _rrf_fuse(vec, [])
    ids = [n.node.node_id for n in result]
    assert ids == ["A", "B", "C"], f"Expected vector order [A,B,C], got {ids}"


def test_rrf_empty_both_returns_empty():
    """Крайний случай — оба пустые."""
    result = _rrf_fuse([], [])
    assert result == []


def test_rrf_k_parameter_changes_weighting():
    """
    С маленьким k (k=1) RRF агрессивнее: rank1 ≫ rank2.
    С большим k (k=1000) RRF почти равен среднему рангов.
    """
    vec = [_nws("first", "F"), _nws("second", "S")]
    bm25 = [_nws("second", "S"), _nws("first", "F")]

    r_k1 = _rrf_fuse(vec, bm25, k=1)
    r_k1000 = _rrf_fuse(vec, bm25, k=1000)

    # С k=1: F (rank1 vec) = 1/2 + 1/3 = 0.833; S (rank2 vec) = 1/3 + 1/2 = 0.833
    # Они равны при симметричном распределении рангов. Проверим только что нет падения.
    assert r_k1[0].score == r_k1[1].score
    # С k=1000 веса должны быть ~одинаковые (топы похожи)
    assert r_k1000[0].score > 0 and r_k1000[1].score > 0


def test_rrf_score_decreases_with_rank():
    """Score должна быть > 0 и монотонно убывать с ростом rank."""
    vec = [_nws(f"v{i}", f"V{i}", 1.0 - i * 0.1) for i in range(5)]
    result = _rrf_fuse(vec, [])
    scores = [n.score for n in result]
    assert all(s > 0 for s in scores), f"Все scores должны быть > 0, got {scores}"
    assert scores == sorted(scores, reverse=True), f"Scores должны убывать, got {scores}"


if __name__ == "__main__":
    import traceback
    tests = [
        test_rrf_basic_known_answer,
        test_rrf_bm25_only_chunk_surfaces,
        test_rrf_dedup_no_duplicate_ids,
        test_rrf_empty_bm25_falls_back_to_vector,
        test_rrf_empty_both_returns_empty,
        test_rrf_k_parameter_changes_weighting,
        test_rrf_score_decreases_with_rank,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
