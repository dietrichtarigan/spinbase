"""The only outbound-HTTP helper. Enforces the egress allowlist and per-host rate limits."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from config.settings import settings
from src.core.privacy import PrivacyViolation

USER_AGENT = "research-kb/0.1 (local-first scientific KB; +https://github.com)"


class RateLimiter:
    """Min-interval limiter. Persisted to disk so separate CLI runs also respect it."""

    def __init__(self, name: str, min_interval_s: float):
        self.min_interval = min_interval_s
        self._lock = threading.Lock()
        self._stamp = settings.db_dir / f".ratelimit_{name}.json"
        self._last = self._load()

    def _load(self) -> float:
        try:
            return float(json.loads(Path(self._stamp).read_text())["t"])
        except Exception:
            return 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.time()
            try:
                self._stamp.write_text(json.dumps({"t": self._last}))
            except OSError:
                pass


def check_egress(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return  # local services (GROBID) are fine
    if not any(host == h or host.endswith("." + h) for h in settings.egress_allowlist):
        raise PrivacyViolation(f"Egress to '{host}' is not in EGRESS_ALLOWLIST")


class GuardedClient:
    """httpx.Client wrapper: allowlist + rate limit + retries. One connection per instance."""

    def __init__(self, limiter: RateLimiter | None = None, timeout: float = 60.0,
                 headers: dict | None = None):
        self.limiter = limiter
        self.client = httpx.Client(
            timeout=timeout, follow_redirects=True,
            headers={"User-Agent": USER_AGENT, **(headers or {})},
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        )

    def request(self, method: str, url: str, retries: int = 4, **kw) -> httpx.Response:
        check_egress(url)
        backoff = 3.0
        for attempt in range(retries + 1):
            if self.limiter:
                self.limiter.wait()
            try:
                r = self.client.request(method, url, **kw)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    ra = r.headers.get("retry-after")
                    # 429 = server-side throttling: back off hard (arXiv often needs 20 s+)
                    wait = float(ra) if ra and ra.isdigit() else (20.0 * 2 ** attempt if r.status_code == 429 else backoff)
                    time.sleep(wait)
                    backoff *= 2
                    continue
                # follow_redirects may land elsewhere -> re-check final host
                check_egress(str(r.url))
                return r
            except (httpx.TransportError,) as e:
                if attempt >= retries:
                    raise
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("unreachable")

    def get(self, url: str, **kw) -> httpx.Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw) -> httpx.Response:
        return self.request("POST", url, **kw)
