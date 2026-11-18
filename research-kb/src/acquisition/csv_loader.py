"""Load the keyword taxonomy CSV (node_id, node, sub_node, keyword, priority) and
score arbitrary text against it."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from config.settings import PRIORITIES, settings

REQUIRED = ("node_id", "node", "sub_node", "keyword", "priority")
_PRIO_ALIASES = {"core": "core", "high": "core", "primary": "core", "1": "core",
                 "supporting": "supporting", "support": "supporting", "medium": "supporting", "2": "supporting",
                 "background": "background", "low": "background", "context": "background", "3": "background"}


@dataclass(frozen=True)
class Keyword:
    node_id: str
    node: str
    sub_node: str
    keyword: str
    priority: str

    @cached_property
    def pattern(self) -> re.Pattern:
        # tolerant match: case-insensitive, flexible whitespace/hyphen between tokens
        toks = [re.escape(t) for t in re.split(r"[\s\-–]+", self.keyword.strip()) if t]
        k = r"[\s\-–]*".join(toks)
        return re.compile(rf"(?<![A-Za-z0-9]){k}(?![a-z0-9])", re.I)


@dataclass
class Taxonomy:
    keywords: list[Keyword]
    source: Path | None = None
    _by_node: dict[str, list[Keyword]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        d = defaultdict(list)
        for k in self.keywords:
            d[k.node].append(k)
        self._by_node = dict(d)

    @property
    def nodes(self) -> list[str]:
        return list(self._by_node)

    def by_node(self, node: str) -> list[Keyword]:
        return self._by_node.get(node, [])

    def by_priority(self, prio: str) -> list[Keyword]:
        return [k for k in self.keywords if k.priority == prio]

    def node_priority(self, node: str) -> str:
        """Highest tier present in a node."""
        tiers = {k.priority for k in self.by_node(node)}
        return next((p for p in PRIORITIES if p in tiers), "background")

    def find_node(self, label: str) -> Keyword | None:
        """Map a free label (e.g. a Zotero tag) to a node/sub_node/keyword."""
        l = label.strip().lower()
        for k in self.keywords:
            if l in (k.node_id.lower(), k.node.lower(), k.sub_node.lower(), k.keyword.lower()):
                return k
        return None

    def score_text(self, text: str) -> "Match":
        """Weighted keyword hits. Core hits count most; title hits counted by caller."""
        w = settings.priority_weights
        hits: list[Keyword] = [k for k in self.keywords if k.pattern.search(text or "")]
        if not hits:
            return Match(0.0, [], None)
        score = sum(w[k.priority] for k in hits)
        node_score: dict[str, float] = defaultdict(float)
        for k in hits:
            node_score[k.node] += w[k.priority]
        best_node = max(node_score, key=node_score.get)
        best_kw = max((k for k in hits if k.node == best_node),
                      key=lambda k: (w[k.priority], len(k.keyword)))
        return Match(score, hits, best_kw)


@dataclass
class Match:
    score: float
    hits: list[Keyword]
    best: Keyword | None

    @property
    def priority(self) -> str:
        """Paper tier = best tier among hits (core if any core keyword matched)."""
        tiers = {k.priority for k in self.hits}
        return next((p for p in PRIORITIES if p in tiers), "background")


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    p = Path(path) if path else settings.resolve_taxonomy_csv()
    with open(p, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rdr = csv.DictReader(f, dialect=dialect)
        cols = {c.strip().lower(): c for c in (rdr.fieldnames or [])}
        missing = [c for c in REQUIRED if c not in cols]
        if missing:
            raise ValueError(f"{p}: missing columns {missing}; found {list(cols)}")
        kws, seen = [], set()
        for row in rdr:
            g = lambda c: (row.get(cols[c]) or "").strip()
            kw = g("keyword")
            if not kw:
                continue
            prio = _PRIO_ALIASES.get(g("priority").lower())
            if prio is None:
                raise ValueError(f"Unknown priority '{g('priority')}' for keyword '{kw}'")
            key = (g("node"), kw.lower())
            if key in seen:
                continue
            seen.add(key)
            kws.append(Keyword(g("node_id"), g("node"), g("sub_node"), kw, prio))
    return Taxonomy(kws, source=p)


def summarize(tax: Taxonomy) -> list[dict]:
    rows = []
    for n in tax.nodes:
        ks = tax.by_node(n)
        rows.append({"node_id": ks[0].node_id, "node": n, "keywords": len(ks),
                     **{p: sum(k.priority == p for k in ks) for p in PRIORITIES}})
    return rows
