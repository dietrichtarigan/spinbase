"""Hard privacy air-gap for unpublished data.

Three independent layers, all enforced in code (not by convention):

1. **Taint flags** – every Paper/Chunk/measurement carries ``is_private``. Anything
   tagged private is refused by :func:`assert_public` before an outbound call.
2. **Content fingerprints** – at private-ingest time, hashed word 8-gram shingles of
   every private document are stored locally. Every outbound payload (DeepSeek
   prompts, search-API queries) is shingled and checked; a single hit blocks the call.
   This catches private text that leaked into a prompt by copy/paste or bug.
3. **Path & egress guard** – files under ``data/private_local`` are tainted by path, and
   :mod:`src.core.http` only talks to hosts in ``settings.egress_allowlist``.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from config.settings import settings


class PrivacyViolation(RuntimeError):
    """Raised when private material is about to leave the machine."""


_WORD = re.compile(r"[A-Za-z0-9_.\-+=]+")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def shingle_hashes(text: str, n: int | None = None) -> set[int]:
    n = n or settings.privacy_shingle_size
    w = _words(text)

    def h(s: str) -> int:  # 63-bit positive int -> fits SQLite INTEGER
        return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") >> 1

    if len(w) < n:
        return {h(" ".join(w))} if len(w) >= 4 else set()
    return {h(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


def is_private_path(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(settings.private_dir.resolve())
        return True
    except ValueError:
        return False


def register_private_text(kb, paper_id: str, text: str) -> int:
    hs = shingle_hashes(text)
    with kb.tx() as c:
        c.executemany("INSERT OR IGNORE INTO private_fingerprints (h, paper_id) VALUES (?,?)",
                      [(h, paper_id) for h in hs])
    return len(hs)


def _leaked(kb, text: str) -> list[str]:
    hs = list(shingle_hashes(text))
    if not hs:
        return []
    hits: set[str] = set()
    for i in range(0, len(hs), 900):
        part = hs[i:i + 900]
        q = ",".join("?" for _ in part)
        for (pid,) in kb.conn.execute(f"SELECT DISTINCT paper_id FROM private_fingerprints WHERE h IN ({q})", part):
            hits.add(pid)
    return sorted(hits)


def assert_public(items: Iterable[Any] = (), text: str = "", kb=None) -> None:
    """Raise PrivacyViolation if any item is private or ``text`` contains private fingerprints."""
    if not settings.privacy_hard_block:
        return
    for it in items:
        flag = getattr(it, "is_private", None)
        if flag is None and isinstance(it, dict):
            flag = it.get("is_private")
        if flag:
            ident = getattr(it, "chunk_id", None) or getattr(it, "paper_id", None) or "<item>"
            raise PrivacyViolation(f"Refusing to transmit private item {ident}")
        p = getattr(it, "pdf_path", None)
        if p and is_private_path(p):
            raise PrivacyViolation(f"Refusing to transmit content from private path {p}")
    if text:
        if kb is None:
            from src.core.db import get_kb
            kb = get_kb()
        leaked = _leaked(kb, text)
        if leaked:
            raise PrivacyViolation(
                f"Outbound payload contains text fingerprinted as private (docs: {', '.join(leaked[:5])}). Blocked.")
