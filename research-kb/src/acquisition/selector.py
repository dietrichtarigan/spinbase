"""Curate the candidate pool down to the ~500-paper corpus (and the ~30-paper MVP batch)."""
from __future__ import annotations

from collections import Counter, defaultdict

from config.settings import PRIORITIES, settings

TIER_SHARE = {"core": 0.60, "supporting": 0.30, "background": 0.10}


def select_corpus(kb, target: int | None = None, min_per_node: int = 5, min_score: float = 0.6) -> dict:
    target = target or settings.corpus_target_size
    pool = kb.papers("is_private=0 AND status='candidate' AND relevance_score >= ?", (min_score,))
    already = kb.papers("is_private=0 AND status NOT IN ('candidate','rejected')")
    chosen: dict[str, object] = {p.paper_id: p for p in already}
    rank = lambda p: (PRIORITIES.index(p.priority), -(p.relevance_score + (1.5 if p.is_review else 0)))
    pool.sort(key=rank)

    # 1) node coverage floor
    by_node = defaultdict(list)
    for p in pool:
        by_node[p.node].append(p)
    have = Counter(p.node for p in chosen.values())
    for node, ps in by_node.items():
        for p in ps:
            if have[node] >= min_per_node or len(chosen) >= target:
                break
            if p.paper_id not in chosen:
                chosen[p.paper_id] = p
                have[node] += 1
    # 2) tier shares
    for tier in PRIORITIES:
        quota = int(target * TIER_SHARE[tier])
        cur = sum(1 for p in chosen.values() if p.priority == tier)
        for p in (x for x in pool if x.priority == tier):
            if cur >= quota or len(chosen) >= target:
                break
            if p.paper_id not in chosen:
                chosen[p.paper_id] = p
                cur += 1
    # 3) fill remainder by rank
    for p in pool:
        if len(chosen) >= target:
            break
        chosen.setdefault(p.paper_id, p)
    newly = [pid for pid in chosen if pid in {p.paper_id for p in pool}]
    with kb.tx() as c:
        c.executemany("UPDATE papers SET status='selected' WHERE paper_id=? AND status='candidate'",
                      [(pid,) for pid in newly])
    return {"selected_now": len(newly), "corpus_size": len(chosen),
            "by_tier": dict(Counter(p.priority for p in chosen.values())),
            "nodes_covered": len({p.node for p in chosen.values()})}


def select_mvp(kb, n: int | None = None) -> list[str]:
    """Pick the MVP batch: core papers, preferring reviews and node diversity, open full text first."""
    n = n or settings.mvp_batch_size
    cands = kb.papers("is_private=0 AND priority='core' AND status != 'rejected'")
    cands.sort(key=lambda p: (not p.arxiv_id and not p.pdf_path, not p.is_review, -p.relevance_score))
    picked, per_node = [], Counter()
    cap = max(3, n // max(1, len({p.node for p in cands})) + 2)
    for p in cands:
        if len(picked) >= n:
            break
        if per_node[p.node] < cap:
            picked.append(p.paper_id)
            per_node[p.node] += 1
    for p in cands:
        if len(picked) >= n:
            break
        if p.paper_id not in picked:
            picked.append(p.paper_id)
    with kb.tx() as c:
        c.executemany("UPDATE papers SET mvp=1, status=CASE WHEN status='candidate' THEN 'selected' ELSE status END "
                      "WHERE paper_id=?", [(pid,) for pid in picked])
    return picked
