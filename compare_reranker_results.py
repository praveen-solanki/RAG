"""
compare_reranker_results.py
===========================
Compare two retrieval result JSON files — typically one produced with
the reranker enabled and one without — and print a comprehensive analysis.

Usage
-----
    python compare_reranker_results.py \\
        --file-a  with_reranker_result.json \\
        --file-b  without_reranker_result.json \\
        --label-a "With reranker" \\
        --label-b "Without reranker"

The script:
  1. Detects whether reranking was actually applied in each file.
  2. Compares final / child / rerank scores across all 10 retrieved chunks
     per question.
  3. Reports per-question rank-order changes.
  4. Summarises retrieval latency.
  5. Lists zero-result questions and breaks them down by difficulty /
     source document.
  6. Flags any anomalies (e.g. files that claim rerank=True but have
     identical / zero rerank_scores).
"""

import argparse
import json
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _reranker_active(results: List[Dict]) -> bool:
    """Return True when at least one chunk has a non-zero rerank_score."""
    for q in results:
        for c in q.get("retrieved_chunks", []):
            if c.get("rerank_score", 0.0) != 0.0:
                return True
    return False


def _score_stats(values: List[float]) -> str:
    if not values:
        return "n/a"
    mean  = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return (f"mean={mean:.5f}  stdev={stdev:.5f}  "
            f"min={min(values):.5f}  max={max(values):.5f}  "
            f"n={len(values)}")


def _top1(results: List[Dict], field: str) -> List[float]:
    return [
        q["retrieved_chunks"][0][field]
        for q in results
        if q.get("retrieved_chunks")
    ]


def _all_chunks(results: List[Dict], field: str) -> List[float]:
    out = []
    for q in results:
        for c in q.get("retrieved_chunks", []):
            out.append(c[field])
    return out


def _ordering(chunks: List[Dict]) -> List[str]:
    return [c["parent_id"] for c in chunks]


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse(
    data_a: Dict[str, Any],
    data_b: Dict[str, Any],
    label_a: str,
    label_b: str,
) -> None:
    results_a: List[Dict] = data_a.get("results", [])
    results_b: List[Dict] = data_b.get("results", [])

    cfg_a = data_a.get("retrieval_config", {})
    cfg_b = data_b.get("retrieval_config", {})

    W = 70

    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * W)
    print("RERANKER COMPARISON ANALYSIS")
    print("=" * W)
    print(f"  File A : {label_a}")
    print(f"  File B : {label_b}")
    print()

    # ── Retrieval config ─────────────────────────────────────────────────────
    print("-" * W)
    print("RETRIEVAL CONFIGURATION")
    print("-" * W)
    all_keys = sorted(set(cfg_a) | set(cfg_b))
    for k in all_keys:
        va = cfg_a.get(k, "—")
        vb = cfg_b.get(k, "—")
        flag = "  =" if va == vb else "  ≠"
        print(f"  {flag}  {k:<22} A={str(va):<25}  B={str(vb)}")
    print()

    # ── Reranker detection ───────────────────────────────────────────────────
    print("-" * W)
    print("RERANKER DETECTION")
    print("-" * W)
    active_a = _reranker_active(results_a)
    active_b = _reranker_active(results_b)

    cfg_says_rerank_a = cfg_a.get("rerank", None)
    cfg_says_rerank_b = cfg_b.get("rerank", None)

    def _yn(v: bool) -> str:
        return "YES  ✓" if v else "NO   ✗"

    print(f"  {'':3}  {'':30} {'A':^16}  {'B':^16}")
    print(f"  {'':3}  {'Config claims rerank':30} "
          f"{str(cfg_says_rerank_a):^16}  {str(cfg_says_rerank_b):^16}")
    print(f"  {'':3}  {'Rerank scores actually non-zero':30} "
          f"{_yn(active_a):^16}  {_yn(active_b):^16}")
    print()

    if cfg_says_rerank_a and not active_a:
        print(f"  ⚠  {label_a}: config says rerank=True but ALL rerank_scores "
              "are zero — reranker may have failed silently.")
    if cfg_says_rerank_b and not active_b:
        print(f"  ⚠  {label_b}: config says rerank=True but ALL rerank_scores "
              "are zero — reranker may have failed silently.")
    if active_a and active_b:
        print(f"  ⚠  BOTH files have non-zero rerank_scores.  "
              "The files may not represent a true with/without reranker comparison.")
    print()

    # ── Question coverage ────────────────────────────────────────────────────
    print("-" * W)
    print("QUESTION COVERAGE")
    print("-" * W)

    ids_a = {q["id"] for q in results_a}
    ids_b = {q["id"] for q in results_b}
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    print(f"  Questions in A          : {len(results_a)}")
    print(f"  Questions in B          : {len(results_b)}")
    if only_a:
        print(f"  Only in A               : {sorted(only_a)}")
    if only_b:
        print(f"  Only in B               : {sorted(only_b)}")
    print()

    # ── Zero-result questions ────────────────────────────────────────────────
    print("-" * W)
    print("ZERO-RESULT QUESTIONS (retrieval returned 0 chunks)")
    print("-" * W)

    zero_a = {q["id"]: q for q in results_a if q.get("num_results", 0) == 0}
    zero_b = {q["id"]: q for q in results_b if q.get("num_results", 0) == 0}
    zero_both = set(zero_a) & set(zero_b)
    zero_only_a = set(zero_a) - set(zero_b)
    zero_only_b = set(zero_b) - set(zero_a)

    print(f"  Zero-result in A        : {len(zero_a)}")
    print(f"  Zero-result in B        : {len(zero_b)}")
    print(f"  Zero-result in both     : {len(zero_both)}")
    if zero_only_a:
        print(f"  Zero ONLY in A          : {sorted(zero_only_a)}")
    if zero_only_b:
        print(f"  Zero ONLY in B          : {sorted(zero_only_b)}")
    if zero_both:
        print(f"\n  Affected question IDs:")
        for qid in sorted(zero_both):
            q = zero_a[qid]
            print(f"    {qid:<20}  diff={q.get('difficulty','?'):<6}  "
                  f"src={q.get('source_document','?')[:55]}")
    print()

    # ── Difficulty breakdown ─────────────────────────────────────────────────
    print("-" * W)
    print("DIFFICULTY BREAKDOWN")
    print("-" * W)

    diff_a: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "zero": 0})
    diff_b: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "zero": 0})
    for q in results_a:
        d = q.get("difficulty", "unknown")
        diff_a[d]["total"] += 1
        if q.get("num_results", 0) == 0:
            diff_a[d]["zero"] += 1
    for q in results_b:
        d = q.get("difficulty", "unknown")
        diff_b[d]["total"] += 1
        if q.get("num_results", 0) == 0:
            diff_b[d]["zero"] += 1

    all_diffs = sorted(set(diff_a) | set(diff_b))
    print(f"  {'Difficulty':<10}  {'A total':>8}  {'A zeros':>8}  "
          f"{'B total':>8}  {'B zeros':>8}")
    for d in all_diffs:
        a = diff_a[d]
        b = diff_b[d]
        print(f"  {d:<10}  {a['total']:>8}  {a['zero']:>8}  "
              f"{b['total']:>8}  {b['zero']:>8}")
    print()

    # ── Source-document breakdown ────────────────────────────────────────────
    print("-" * W)
    print("SOURCE-DOCUMENT BREAKDOWN (zero-result questions)")
    print("-" * W)

    doc_zero_a: Dict[str, int] = defaultdict(int)
    doc_total_a: Dict[str, int] = defaultdict(int)
    for q in results_a:
        doc = q.get("source_document", "unknown")
        doc_total_a[doc] += 1
        if q.get("num_results", 0) == 0:
            doc_zero_a[doc] += 1

    affected_docs = [d for d in doc_zero_a if doc_zero_a[d] > 0]
    if affected_docs:
        for doc in sorted(affected_docs, key=lambda d: -doc_zero_a[d]):
            short = doc[:55]
            print(f"  {short:<55}  "
                  f"{doc_zero_a[doc]}/{doc_total_a[doc]} questions → 0 results")
    else:
        print("  No documents had zero-result questions.")
    print()

    # ── Score comparison (successful queries only) ───────────────────────────
    print("-" * W)
    print("SCORE STATISTICS  (successful queries — top-1 chunk per question)")
    print("-" * W)

    for field in ("final_score", "child_score", "rerank_score"):
        vals_a = _top1(results_a, field)
        vals_b = _top1(results_b, field)
        identical = all(abs(a - b) < 1e-9 for a, b in zip(vals_a, vals_b))
        print(f"  {field}")
        print(f"    A: {_score_stats(vals_a)}")
        print(f"    B: {_score_stats(vals_b)}")
        print(f"    Scores bit-identical: {identical}")
        print()

    # ── All-chunk score comparison ────────────────────────────────────────────
    print("-" * W)
    print("SCORE STATISTICS  (all retrieved chunks)")
    print("-" * W)

    for field in ("final_score", "child_score", "rerank_score"):
        vals_a = _all_chunks(results_a, field)
        vals_b = _all_chunks(results_b, field)
        identical = (len(vals_a) == len(vals_b) and
                     all(abs(a - b) < 1e-9 for a, b in zip(vals_a, vals_b)))
        print(f"  {field}")
        print(f"    A: {_score_stats(vals_a)}")
        print(f"    B: {_score_stats(vals_b)}")
        print(f"    Scores bit-identical: {identical}")
        print()

    # ── Rank-order changes ───────────────────────────────────────────────────
    print("-" * W)
    print("CHUNK RANK-ORDER CHANGES (per question)")
    print("-" * W)

    common_ids = ids_a & ids_b
    lookup_a = {q["id"]: q for q in results_a}
    lookup_b = {q["id"]: q for q in results_b}

    same_order_count    = 0
    changed_order_count = 0
    changed_examples: List[Tuple[str, List[str], List[str]]] = []

    for qid in sorted(common_ids):
        qa = lookup_a[qid]
        qb = lookup_b[qid]
        order_a = _ordering(qa.get("retrieved_chunks", []))
        order_b = _ordering(qb.get("retrieved_chunks", []))
        if order_a == order_b:
            same_order_count += 1
        else:
            changed_order_count += 1
            changed_examples.append((qid, order_a, order_b))

    print(f"  Same rank order         : {same_order_count}")
    print(f"  Different rank order    : {changed_order_count}")
    if changed_examples:
        print(f"\n  Examples (first 5):")
        for qid, oa, ob in changed_examples[:5]:
            print(f"    {qid}")
            print(f"      A order: {oa}")
            print(f"      B order: {ob}")
    print()

    # ── Latency comparison ───────────────────────────────────────────────────
    print("-" * W)
    print("RETRIEVAL LATENCY  (seconds per query, successful queries only)")
    print("-" * W)

    times_a = [q["retrieval_time_s"] for q in results_a if q.get("num_results", 0) > 0]
    times_b = [q["retrieval_time_s"] for q in results_b if q.get("num_results", 0) > 0]

    def _lat(vals: List[float]) -> str:
        if not vals:
            return "n/a"
        p95 = sorted(vals)[int(0.95 * (len(vals) - 1))]
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return (f"mean={statistics.mean(vals):.3f}s  "
                f"stdev={stdev:.3f}s  "
                f"min={min(vals):.3f}s  max={max(vals):.3f}s  "
                f"p95={p95:.3f}s  total={sum(vals):.1f}s")

    print(f"  A: {_lat(times_a)}")
    print(f"  B: {_lat(times_b)}")
    if times_a and times_b:
        delta = statistics.mean(times_a) - statistics.mean(times_b)
        sign  = "+" if delta >= 0 else ""
        print(f"  Δ mean (A − B): {sign}{delta:.3f}s  "
              f"({'A slower' if delta > 0.05 else 'B slower' if delta < -0.05 else 'negligible'})")
    print()

    # ── Overall observations ─────────────────────────────────────────────────
    print("=" * W)
    print("KEY OBSERVATIONS")
    print("=" * W)

    obs: List[str] = []

    if active_a and active_b:
        obs.append(
            "IMPORTANT — Both files contain non-zero rerank_scores. "
            "The two files were produced with the reranker ACTIVE in both runs. "
            "A true 'without reranker' run should have all-zero rerank_scores "
            "and final_score == child_score."
        )
    elif active_a and not active_b:
        obs.append(
            "File A has non-zero rerank_scores (reranker was active). "
            "File B has all-zero rerank_scores (reranker was NOT active or failed)."
        )
    elif not active_a and active_b:
        obs.append(
            "File A has all-zero rerank_scores (reranker was NOT active or failed). "
            "File B has non-zero rerank_scores (reranker was active). "
            "Labels A/B may be swapped."
        )
    else:
        obs.append("Neither file has non-zero rerank_scores — reranker was not active in either run.")

    all_scores_identical = all(
        abs(a - b) < 1e-9
        for qa, qb in zip(results_a, results_b)
        for ca, cb in zip(qa.get("retrieved_chunks", []),
                          qb.get("retrieved_chunks", []))
        for a, b in [(ca["final_score"], cb["final_score"])]
    )
    if all_scores_identical:
        obs.append(
            "All final_scores, child_scores and rerank_scores are bit-for-bit "
            "identical between the two files — the retrieval results are the same run."
        )

    if changed_order_count == 0:
        obs.append(
            "Chunk ordering is identical for every question in both files."
        )
    else:
        obs.append(
            f"Chunk ordering differs for {changed_order_count} question(s) — "
            "the reranker changed the ranking for those queries."
        )

    if zero_both:
        obs.append(
            f"{len(zero_both)} question(s) returned 0 chunks in both runs.  "
            "These questions likely reference source documents that are not "
            "present in the Qdrant collection (documents not indexed)."
        )

    if times_a and times_b:
        delta_s = statistics.mean(times_a) - statistics.mean(times_b)
        if abs(delta_s) > 0.1:
            slower_label = label_a if delta_s > 0 else label_b
            obs.append(
                f"'{slower_label}' is ~{abs(delta_s):.2f}s/query slower on average, "
                "consistent with the cross-encoder reranking pass adding latency."
            )
        else:
            obs.append(
                "Latency is nearly identical between both runs "
                f"(Δ mean ≈ {abs(delta_s)*1000:.0f} ms/query)."
            )

    for i, o in enumerate(obs, 1):
        print(f"\n  {i}. {o}")

    print("\n" + "=" * W)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two retrieval result JSON files (reranker vs no-reranker)"
    )
    parser.add_argument(
        "--file-a",
        required=True,
        metavar="PATH",
        help="Path to first results JSON (e.g. with_reranker_result.json)",
    )
    parser.add_argument(
        "--file-b",
        required=True,
        metavar="PATH",
        help="Path to second results JSON (e.g. without_reranker_result.json)",
    )
    parser.add_argument(
        "--label-a",
        default="File A",
        help="Human-readable label for file A (default: 'File A')",
    )
    parser.add_argument(
        "--label-b",
        default="File B",
        help="Human-readable label for file B (default: 'File B')",
    )
    args = parser.parse_args()

    data_a = load(args.file_a)
    data_b = load(args.file_b)

    analyse(data_a, data_b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
