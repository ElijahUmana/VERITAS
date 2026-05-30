"""CourtListener citation-existence oracle — VERITAS cold open (FLOOR §1/§2).

The cold open's mechanical, EXTERNAL truth test: does a cited case actually
exist? An agent can *say* anything; a database either has the citation or it
does not.

  real citation   347 U.S. 483 (Brown v. Board)  -> HTTP 200 -> CONFIRMED (GREEN)
  fake citation   999 U.S. 9999                   -> HTTP 404 -> REFUTED   (RED)

DETERMINISM CONTRACT (FLOOR §1 COLD OPEN row, "Hard-cache ... live call is
best-effort overlay"):

  * The on-disk cache under ``cold_open/cache/`` is AUTHORITATIVE by default.
    The demo NEVER depends on the network or a token to land its verdict.
  * The CourtListener v4 citation-lookup API
    (POST https://www.courtlistener.com/api/rest/v4/citation-lookup/) is a
    BEST-EFFORT OVERLAY. It now REQUIRES ``Authorization: Token <token>``
    (401 unauthenticated, as of 2026-05-07). We read ``COURTLISTENER_TOKEN``
    from the environment; if it is absent or the call fails we fall back to the
    cache and say *why*, LOUDLY — never a silent or faked verdict.
  * On a SUCCESSFUL authenticated live call we self-refresh the on-disk fixture
    with the real captured bytes (write-through), upgrading the fixture to a
    real capture the first time the token works.

This oracle emits a :class:`crucible.schemas.Verdict` (``oracle_type="citation"``)
via :meth:`CitationCheck.to_verdict`, so a citation claim is gated by the SAME
truth floor (:func:`crucible.schemas.evaluate_truth_floor`) as a kernel claim —
no special path, no trust shortcut.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
from typing import Any, Optional

CITATION_LOOKUP_URL = "https://www.courtlistener.com/api/rest/v4/citation-lookup/"

# cold_open/cache/ lives two levels up from this file (crucible/oracle/ -> repo root).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _REPO_ROOT / "cold_open" / "cache"

# Optional integration with crucible-core's canonical schema (FLOOR §2). It is
# published, but we guard the import so this oracle is also runnable standalone
# (e.g. for the cold open) without the rest of the package importing cleanly.
try:
    from crucible.schemas import Verdict as _Verdict  # type: ignore
    _HAVE_SCHEMA = True
except Exception:  # pragma: no cover - standalone fallback
    _Verdict = None
    _HAVE_SCHEMA = False


def _norm(cite: str) -> str:
    """Collapse internal whitespace; keep case (citations are case-sensitive: 'U.S.')."""
    return " ".join((cite or "").split()).strip()


@dataclasses.dataclass
class CitationCheck:
    """The display-oriented result of one citation existence check."""

    citation: str
    found: bool
    status: int                  # per-citation HTTP status from the API (200/404/...), 0 if undetermined
    verdict: str                 # confirmed | refuted | unverified
    color: str                   # GREEN | RED | GREY
    case_name: Optional[str]
    source: str                  # cache | live | live->cache-fallback | none
    note: str                    # provenance / why this verdict (always populated)
    raw: dict                    # the matched citation object from the API/fixture

    def to_verdict(self, *, claim_id: str, candidate_id: str, mission_id: str):
        """Adapt to ``crucible.schemas.Verdict`` (``oracle_type='citation'``).

        Maps onto the gate-critical flat fields:
          confirmed -> correctness_passed=True   (the cited case exists)
          refuted   -> correctness_passed=False  (the agent's existence claim is false)
          unverified-> blocked_reason set; verifier_status stays OK (we simply
                       could not reach a decision — never faked).

        Returns a ``Verdict`` instance if the schema is importable, else a plain
        dict with the identical field names so callers can swap in seamlessly.
        """
        evidence = {
            "citation": self.citation,
            "http_status": self.status,
            "case_name": self.case_name,
            "source": self.source,
            "note": self.note,
            "courtlistener": self.raw,
        }
        payload: dict[str, Any] = dict(
            claim_id=claim_id,
            candidate_id=candidate_id,
            mission_id=mission_id,
            verdict=self.verdict,
            oracle_type="citation",
            verifier_status="OK",
            correctness_passed=bool(self.found) if self.verdict != "unverified" else False,
            tamper_detected=False,
            blocked_reason=(self.note if self.verdict == "unverified" else None),
            evidence=evidence,
        )
        if _HAVE_SCHEMA:
            return _Verdict(**payload)
        return payload


class CitationOracle:
    """Cache-first CourtListener citation-existence oracle with a best-effort
    live overlay. See module docstring for the determinism contract."""

    def __init__(
        self,
        cache_dir: os.PathLike | str = DEFAULT_CACHE_DIR,
        token: Optional[str] = None,
        prefer_live: bool = False,
        timeout: float = 6.0,
        write_through: bool = True,
        verbose: bool = True,
    ) -> None:
        self.cache_dir = pathlib.Path(cache_dir)
        # explicit token arg wins; else env; else None (=> cache-only, by design today)
        self.token = token if token is not None else (os.environ.get("COURTLISTENER_TOKEN") or None)
        self.prefer_live = prefer_live
        self.timeout = timeout
        self.write_through = write_through
        self.verbose = verbose
        self._index = self._build_index()

    # -- logging (loud, to stderr; never swallow) ---------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[citation_oracle] {msg}", file=sys.stderr)

    # -- cache indexing ------------------------------------------------------
    @staticmethod
    def _response_of(doc: Any) -> list:
        """Unwrap a cache file to the citation-lookup response array.

        Accepts either the raw API array (what a live call returns) or our
        ``{"_cache_meta": ..., "response": [...]}`` fixture wrapper.
        """
        if isinstance(doc, list):
            return doc
        if isinstance(doc, dict):
            r = doc.get("response")
            return r if isinstance(r, list) else []
        return []

    def _build_index(self) -> dict[str, pathlib.Path]:
        idx: dict[str, pathlib.Path] = {}
        if not self.cache_dir.is_dir():
            self._log(f"WARN: cache dir not found: {self.cache_dir}")
            return idx
        for p in sorted(self.cache_dir.glob("*.json")):
            try:
                doc = json.loads(p.read_text())
            except Exception as e:  # corrupt fixture: surface, do not silently skip-as-ok
                self._log(f"WARN: cache file {p.name} is unreadable JSON: {e}")
                continue
            meta = doc.get("_cache_meta", {}) if isinstance(doc, dict) else {}
            if isinstance(meta, dict) and meta.get("citation"):
                idx[_norm(meta["citation"])] = p
            for obj in self._response_of(doc):
                c = obj.get("citation") if isinstance(obj, dict) else None
                if c:
                    idx.setdefault(_norm(c), p)
        return idx

    # -- live overlay --------------------------------------------------------
    def _post_form(self, url: str, data: dict, headers: dict) -> Any:
        """POST application/x-www-form-urlencoded; return parsed JSON.

        Uses ``requests`` if present, else stdlib ``urllib`` (zero added deps).
        Raises on auth/rate-limit/transport errors (caller turns this into a
        loud cache-fallback, never a faked verdict).
        """
        try:
            import requests  # type: ignore

            resp = requests.post(url, data=data, headers=headers, timeout=self.timeout)
            code = resp.status_code
            body = resp.text
            if code == 401:
                raise RuntimeError("401 Unauthorized — CourtListener now requires 'Authorization: Token <token>'")
            if code == 403:
                raise RuntimeError("403 Forbidden — token lacks permission")
            if code == 429:
                raise RuntimeError("429 Too Many Requests — rate limited (60 valid citations/min)")
            if code >= 400:
                raise RuntimeError(f"HTTP {code}: {body[:200]}")
            return resp.json()
        except ImportError:
            import urllib.error
            import urllib.parse
            import urllib.request

            encoded = urllib.parse.urlencode(data).encode()
            req = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                code = e.code
                detail = e.read().decode(errors="replace")[:200]
                if code == 401:
                    raise RuntimeError("401 Unauthorized — CourtListener now requires 'Authorization: Token <token>'")
                if code == 429:
                    raise RuntimeError("429 Too Many Requests — rate limited")
                raise RuntimeError(f"HTTP {code}: {detail}")
            except urllib.error.URLError as e:
                raise RuntimeError(f"transport error reaching CourtListener: {e}")

    def _live_lookup(self, citation: str) -> tuple[Optional[list], str]:
        headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "VERITAS-cold-open/1.0",
        }
        try:
            arr = self._post_form(CITATION_LOOKUP_URL, {"text": citation}, headers)
        except Exception as e:
            return None, f"live overlay failed ({e}) -> cache fallback"
        if not isinstance(arr, list):
            return None, f"live overlay returned non-array payload ({type(arr).__name__}) -> cache fallback"
        return arr, "live overlay OK (authenticated CourtListener v4)"

    def _write_through(self, citation: str, arr: list) -> None:
        """Self-refresh the fixture for ``citation`` with real captured bytes."""
        if not self.write_through:
            return
        path = self._index.get(_norm(citation))
        if path is None:
            return  # only refresh known fixtures during a demo; never create stray files
        try:
            old = json.loads(path.read_text())
            meta = old.get("_cache_meta", {}) if isinstance(old, dict) else {}
        except Exception:
            meta = {}
        meta = dict(meta)
        meta["is_fixture"] = False
        meta["source"] = "live"
        meta["last_live_refresh"] = "captured from authenticated CourtListener v4 citation-lookup"
        try:
            path.write_text(json.dumps({"_cache_meta": meta, "response": arr}, indent=2) + "\n")
            self._log(f"self-refreshed fixture {path.name} with real captured bytes")
        except Exception as e:
            self._log(f"WARN: could not self-refresh {path.name}: {e}")

    def _load_cache(self, citation: str) -> tuple[Optional[list], str]:
        path = self._index.get(_norm(citation))
        if path is None:
            return None, f"no cache fixture for '{citation}'"
        try:
            doc = json.loads(path.read_text())
        except Exception as e:
            return None, f"cache fixture {path.name} unreadable: {e}"
        return self._response_of(doc), f"loaded fixture {path.name}"

    # -- matching + interpretation ------------------------------------------
    @staticmethod
    def _match(arr: list, citation: str) -> dict:
        norm = _norm(citation)
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            if _norm(obj.get("citation", "")) == norm:
                return obj
            for nc in obj.get("normalized_citations") or []:
                if _norm(nc) == norm:
                    return obj
        # single-element responses (our fixtures) -> use it
        return arr[0] if len(arr) == 1 and isinstance(arr[0], dict) else {}

    def _interpret(self, citation: str, obj: dict, source: str, note: str) -> CitationCheck:
        status = 0
        try:
            status = int(obj.get("status", 0)) if obj else 0
        except (TypeError, ValueError):
            status = 0
        clusters = obj.get("clusters", []) if obj else []
        case_name = None
        if clusters and isinstance(clusters[0], dict):
            case_name = clusters[0].get("case_name") or clusters[0].get("case_name_full")

        if status == 200 and clusters:
            return CitationCheck(
                citation=citation, found=True, status=200, verdict="confirmed",
                color="GREEN", case_name=case_name, source=source,
                note=(note + "; " if note else "") + f"CourtListener returned 200 — real case: {case_name}",
                raw=obj,
            )
        if status == 404 or (obj and not clusters and status in (0, 404)):
            return CitationCheck(
                citation=citation, found=False, status=status or 404, verdict="refuted",
                color="RED", case_name=None, source=source,
                note=(note + "; " if note else "") + "CourtListener returned 404 — this citation does not exist",
                raw=obj,
            )
        # 300 multiple / 400 invalid reporter / unexpected -> do NOT fake GREEN or RED
        return CitationCheck(
            citation=citation, found=False, status=status, verdict="unverified",
            color="GREY", case_name=case_name, source=source,
            note=(note + "; " if note else "") + f"non-decisive status {status} — UNVERIFIED (not a verdict)",
            raw=obj,
        )

    # -- public entry point --------------------------------------------------
    def check(self, citation: str) -> CitationCheck:
        """Resolve a single citation to a GREEN/RED/GREY existence verdict."""
        norm = _norm(citation)
        if not norm:
            return CitationCheck(norm, False, 0, "unverified", "GREY", None, "none",
                                 "empty citation string — UNVERIFIED", {})
        notes: list[str] = []
        arr: Optional[list] = None
        source: Optional[str] = None

        if self.prefer_live:
            if self.token:
                arr, n = self._live_lookup(norm)
                notes.append(n)
                if arr is not None:
                    source = "live"
                    self._write_through(norm, arr)
            else:
                notes.append("live overlay skipped: COURTLISTENER_TOKEN absent (cache is authoritative)")

        if arr is None:
            arr, n = self._load_cache(norm)
            notes.append(n)
            if arr is not None:
                source = "live->cache-fallback" if (self.prefer_live and self.token) else "cache"

        if arr is None:
            return CitationCheck(
                norm, False, 0, "unverified", "GREY", None, "none",
                "; ".join(notes) + "; NO cache fixture and no live result — UNVERIFIED (never faked)",
                {},
            )

        obj = self._match(arr, norm)
        return self._interpret(norm, obj, source or "cache", "; ".join(notes))


def check_citation(citation: str, **kwargs: Any) -> CitationCheck:
    """Convenience: one-shot check with a fresh oracle."""
    return CitationOracle(**kwargs).check(citation)


if __name__ == "__main__":
    # Quick manual probe:  python -m crucible.oracle.citation_oracle "347 U.S. 483"
    cites = sys.argv[1:] or ["347 U.S. 483", "999 U.S. 9999"]
    oracle = CitationOracle(prefer_live=("--live" in cites))
    cites = [c for c in cites if c != "--live"]
    for c in cites:
        res = oracle.check(c)
        print(f"{res.color:5} {res.verdict:10} status={res.status:<4} {res.citation!r} "
              f"case={res.case_name!r} source={res.source}")
        print(f"      note: {res.note}")
