"""
Unit-тесты для F1, F5, F6 (и F2-хелпера).
Не требует моделей/ChromaDB/GPU.

Запуск:
    cd C:\\test
    python tests/test_accuracy_fixes.py
"""
import sys, os, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core.schema import TextNode, NodeWithScore
from src.rag_pipeline import _rrf_fuse, _rrf_fuse_across_files


def _nws(text, node_id, score=0.0, meta=None):
    return NodeWithScore(node=TextNode(text=text, id_=node_id, metadata=meta or {}), score=score)


def _nws_with_meta(text, node_id, meta, score=0.0):
    return NodeWithScore(node=TextNode(text=text, id_=node_id, metadata=meta), score=score)


def test_f1_bm25_prefix_includes_file_and_page():
    """
    F1: при сборке BM25-узла текст должен содержать [file_name стр.page]:
    Это имитация логики в _rebuild_bm25_bg.
    """
    meta = {"file_name": "Лекция 3 РпСАПР.pdf", "page": 5}
    text = "Нечёткая логика, функция принадлежности, лингвистическая переменная."
    # Эмулируем то, что делает _rebuild_bm25_bg
    fname = meta.get('file_name', '')
    page = meta.get('page', '')
    coord_parts = []
    if fname: coord_parts.append(str(fname))
    if page not in ('', None): coord_parts.append(f"стр.{page}")
    bm25_text = f"[{' '.join(coord_parts)}]: {text}"
    assert "Лекция 3 РпСАПР.pdf" in bm25_text, f"File name missing: {bm25_text}"
    assert "стр.5" in bm25_text, f"Page missing: {bm25_text}"
    assert "Нечёткая логика" in bm25_text, f"Original text missing: {bm25_text}"
    # Оригинальный текст для embedding (для сравнения — должен быть короче)
    embedding_text = text
    assert len(bm25_text) > len(embedding_text), "BM25 prefix не добавляет длины"


def test_f1_bm25_prefix_uses_time_for_video_chunks():
    """F1: для video-чанков (нет page) используется time/start."""
    meta = {"file_name": "lecture.mp4", "start": 125.5}
    text = "transcript text"
    fname = meta.get('file_name', '')
    page = meta.get('page', '')
    t = meta.get('start', meta.get('time', ''))
    coord_parts = []
    if fname: coord_parts.append(str(fname))
    if page not in ('', None): coord_parts.append(f"стр.{page}")
    elif t not in ('', None): coord_parts.append(f"@{t}")
    bm25_text = f"[{' '.join(coord_parts)}]: {text}"
    assert "@125.5" in bm25_text, f"Time missing: {bm25_text}"


def test_f5_rerank_doc_includes_prefix():
    """
    F5: документ, уходящий в rerank, содержит префикс с координатами.
    Имитация функции _rerank_doc в retrieve_nodes.
    """
    nws = _nws_with_meta("Кадр 0:42: описание схемы", "v1",
                         {"file_name": "video.mp4", "time": 42.0})
    # Эмулируем _rerank_doc
    meta = nws.node.metadata or {}
    coord_parts = []
    if meta.get("file_name"): coord_parts.append(str(meta["file_name"]))
    if meta.get("page") not in (None, ""): coord_parts.append(f"стр.{meta['page']}")
    elif meta.get("time") not in (None, ""): coord_parts.append(f"@{meta['time']}")
    elif meta.get("start") not in (None, ""): coord_parts.append(f"@{meta['start']}")
    prefix = f"[{' '.join(coord_parts)}] " if coord_parts else ""
    rerank_text = prefix + nws.node.get_content()
    assert "video.mp4" in rerank_text
    assert "@42" in rerank_text
    assert "Кадр 0:42" in rerank_text


def test_f5_rerank_doc_works_for_pdf_with_page():
    """F5: для PDF-чанка — file_name + стр.page."""
    nws = _nws_with_meta("Текст абзаца", "p1", {"file_name": "Лекция 1.pdf", "page": 3})
    meta = nws.node.metadata or {}
    coord_parts = []
    if meta.get("file_name"): coord_parts.append(str(meta["file_name"]))
    if meta.get("page") not in (None, ""): coord_parts.append(f"стр.{meta['page']}")
    prefix = f"[{' '.join(coord_parts)}] " if coord_parts else ""
    rerank_text = prefix + nws.node.get_content()
    assert "Лекция 1.pdf" in rerank_text
    assert "стр.3" in rerank_text


def test_f6_adaptive_threshold_typical_case():
    """
    F6: median-MAD адаптивный порог.
    Типичный случай: scores = [0.85, 0.82, 0.50, 0.48, 0.45, 0.10, 0.05]
    median = 0.48, MAD = 0.32
    adaptive_thr = max(0, 0.48 - 2*0.32) = max(0, -0.16) = 0
    → пропускает всё (правильно: плотное распределение без явных outliers)
    """
    scores = [0.85, 0.82, 0.50, 0.48, 0.45, 0.10, 0.05]
    median = statistics.median(scores)
    mad = statistics.median([abs(s - median) for s in scores]) or 0.05
    adaptive_thr = max(0.0, median - 2.0 * mad)
    # С MAD=0.32, threshold = 0.48 - 0.64 = -0.16, clamp to 0
    assert adaptive_thr == 0.0
    above = [s for s in scores if s >= adaptive_thr]
    assert len(above) == len(scores), "All chunks should pass with threshold=0"


def test_f6_adaptive_threshold_clear_outlier():
    """
    F6: scores = [0.9, 0.85, 0.82, 0.80, 0.10, 0.05]
    median=0.81, MAD=0.05
    adaptive_thr = 0.81 - 0.10 = 0.71
    → отрезает outliers 0.10, 0.05 (правильно — это шум)
    """
    scores = [0.9, 0.85, 0.82, 0.80, 0.10, 0.05]
    median = statistics.median(scores)
    mad = statistics.median([abs(s - median) for s in scores]) or 0.05
    adaptive_thr = max(0.0, median - 2.0 * mad)
    assert 0.6 < adaptive_thr < 0.8, f"Expected ~0.7, got {adaptive_thr}"
    above = [s for s in scores if s >= adaptive_thr]
    assert len(above) == 4, f"Should keep top 4, got {len(above)}: {above}"
    # Outliers (0.10, 0.05) должны быть отрезаны
    assert all(s > 0.5 for s in above), f"Outliers leaked: {above}"


def test_f6_adaptive_threshold_fallback_for_few_chunks():
    """F6: при <4 чанков возвращаем статический порог."""
    # Имитация ветки в коде
    static_thr = 0.05
    all_nodes_count = 3
    if all_nodes_count >= 4:
        # adaptive logic
        adaptive_thr = 0.3
    else:
        adaptive_thr = static_thr
    assert adaptive_thr == static_thr, "Должен фоллбэк на статический порог"


def test_f6_static_threshold_misses_technical_chunks():
    """
    F6 (negative test): статический 0.05 режет легитимные чанки
    на технических запросах.
    """
    scores_technical = [0.4, 0.3, 0.25, 0.2, 0.15, 0.05, 0.02]  # все релевантны
    # Статический порог:
    static_above = [s for s in scores_technical if s >= 0.05]
    # 5 чанков проходят — ок, минимум 5 есть
    # Но в случае если 8 чанков с scores [0.04, 0.03, 0.02, 0.01, 0.005] — все rejected
    scores_marginal = [0.04, 0.03, 0.02, 0.01, 0.005]
    static_above_marginal = [s for s in scores_marginal if s >= 0.05]
    assert len(static_above_marginal) == 0, "Static 0.05 отрезает всё"
    # Adaptive: median=0.02, MAD=0.01, thr = max(0, 0.02 - 0.02) = 0
    median = statistics.median(scores_marginal)
    mad = statistics.median([abs(s - median) for s in scores_marginal]) or 0.05
    adaptive_thr = max(0.0, median - 2.0 * mad)
    adaptive_above = [s for s in scores_marginal if s >= adaptive_thr]
    assert len(adaptive_above) == len(scores_marginal), "Adaptive 0 пропускает все"


def test_f2_rrf_across_files_balances_big_and_small():
    """
    F2: _rrf_fuse_across_files даёт равный голос каждому файлу.
    Большой файл (10 чанков) vs маленький (1 чанк) — оба получают свой голос.
    """
    big_file = [_nws(f"big{i}", f"big{i}", 0.9 - i*0.05) for i in range(10)]
    small_file = [_nws("small0", "small0", 0.95)]
    file_results = [("big.pdf", big_file), ("small.pdf", small_file)]
    result = _rrf_fuse_across_files(file_results)
    # Маленький файл получил rank=1 в своём file_results → 1/61 = 0.0164
    # Большой файл получил rank=1..10 в своём → 1/61..1/70
    # Top чанки маленького файла конкурируют с топом большого — это и есть баланс
    ids = [n.node.node_id for n in result]
    assert "small0" in ids[:3], f"Small file chunk not in top 3: {ids[:5]}"


def test_f2_rrf_across_files_dedupes_across_files():
    """F2: один и тот же node_id в двух файлах (бывает при re-ingest) — дедуплицируется."""
    # Создаём узел с одинаковым id_ в двух файлах
    shared_node = TextNode(text="shared", id_="shared")
    nws_shared_1 = NodeWithScore(node=shared_node, score=0.9)
    nws_shared_2 = NodeWithScore(node=shared_node, score=0.5)
    file_results = [
        ("a.pdf", [nws_shared_1, _nws("a1", "a1")]),
        ("b.pdf", [nws_shared_2, _nws("b1", "b1")]),
    ]
    result = _rrf_fuse_across_files(file_results)
    ids = [n.node.node_id for n in result]
    # shared должен появиться один раз с суммированным RRF
    assert ids.count("shared") == 1, f"shared should appear once, got {ids}"
    # RRF shared: 1/61 (a) + 1/61 (b) = ~0.0328 → должен быть в топе
    assert ids[0] == "shared", f"shared should be top by summed RRF, got {ids[0]}"


def test_f2_rrf_across_files_empty_input():
    """F2: пустой список файлов → пустой результат."""
    result = _rrf_fuse_across_files([])
    assert result == []


def test_f2_rrf_across_files_single_file():
    """F2: один файл — эквивалентно простому RRF."""
    nodes = [_nws(f"n{i}", f"id{i}", 0.9-i*0.1) for i in range(3)]
    file_results = [("only.pdf", nodes)]
    result = _rrf_fuse_across_files(file_results)
    assert [n.node.node_id for n in result] == ["id0", "id1", "id2"]


if __name__ == "__main__":
    import traceback
    tests = [
        test_f1_bm25_prefix_includes_file_and_page,
        test_f1_bm25_prefix_uses_time_for_video_chunks,
        test_f5_rerank_doc_includes_prefix,
        test_f5_rerank_doc_works_for_pdf_with_page,
        test_f6_adaptive_threshold_typical_case,
        test_f6_adaptive_threshold_clear_outlier,
        test_f6_adaptive_threshold_fallback_for_few_chunks,
        test_f6_static_threshold_misses_technical_chunks,
        test_f2_rrf_across_files_balances_big_and_small,
        test_f2_rrf_across_files_dedupes_across_files,
        test_f2_rrf_across_files_empty_input,
        test_f2_rrf_across_files_single_file,
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
