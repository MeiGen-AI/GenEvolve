"""Open-source friendly text and image search tools for GenEvolve.

Both tools speak the public Serper.dev REST protocol. To use a different
backend, override the ``base_url`` constructor argument or set
``SERPER_BASE_URL`` --- the request schema (``q``/``num``) and the response
parsing (``organic`` / ``images``) follow the standard Serper format.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


DEFAULT_SERPER_BASE_URL = "https://google.serper.dev"
DEFAULT_IMAGE_DIR = "/tmp/genevolve_images"


def _require_requests() -> None:
    if requests is None:
        raise RuntimeError(
            "The `requests` package is required for GenEvolve search tools. "
            "Install it with `pip install requests`."
        )


def _read_serper_api_key() -> str:
    key = (os.environ.get("SERPER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "SERPER_API_KEY is not set. Get a key at https://serper.dev "
            "and `export SERPER_API_KEY=...` before running GenEvolve."
        )
    return key


def _serper_base_url(base_url: Optional[str]) -> str:
    if base_url:
        return base_url.rstrip("/")
    return (os.environ.get("SERPER_BASE_URL") or DEFAULT_SERPER_BASE_URL).rstrip("/")


# ---------------------------------------------------------------------------
# Text search
# ---------------------------------------------------------------------------


class WebTextSearchTool:
    """Synchronous text search compatible with the Serper /search endpoint."""

    TOOL_DEFINITION = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Web text search. Supply an array of queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of query strings.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results (default: 5).",
                    },
                },
                "required": ["queries"],
            },
        },
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        _require_requests()
        self.base_url = _serper_base_url(base_url)
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/search"
        headers = {
            "X-API-KEY": _read_serper_api_key(),
            "Content-Type": "application/json",
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(8, 1 + 2 * attempt))
        raise RuntimeError(f"Serper /search failed after {self.max_retries} retries: {last_err}")

    @staticmethod
    def _format_text_results(query: str, items: Sequence[Dict[str, Any]], top_k: int) -> str:
        lines: List[str] = [f"--- search result for [{query}] ---"]
        for idx, item in enumerate(items[:top_k], 1):
            title = (item.get("title") or "").strip()
            link = (item.get("link") or item.get("url") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            lines.append(f"{idx}. title: {title}")
            if link:
                lines.append(f"  url: {link}")
            if snippet:
                lines.append(f"  snippet: {snippet}")
        lines.append("--- end of search result ---")
        return "\n".join(lines)

    def call(self, queries: Sequence[str], top_k: int = 5, **_: object) -> str:
        if isinstance(queries, str):
            queries = [queries]
        if not queries:
            return "[search] empty queries"
        out_blocks: List[str] = []
        for q in queries:
            q = (q or "").strip()
            if not q:
                continue
            try:
                data = self._post({"q": q, "num": int(top_k)})
                organic = data.get("organic") or []
            except Exception as exc:  # noqa: BLE001
                out_blocks.append(f"--- search error for [{q}]: {exc} ---")
                continue
            out_blocks.append(self._format_text_results(q, organic, int(top_k)))
        return "\n".join(out_blocks) if out_blocks else "[search] no results"


# ---------------------------------------------------------------------------
# Image search
# ---------------------------------------------------------------------------


class ImageSearchTool:
    """Image search + local caching, compatible with Serper /images.

    The agent receives a unique ``IMG_###`` identifier per result; the
    downloader writes images under ``IMAGE_DOWNLOAD_DIR`` and returns the
    local path. Trajectories should pass these local paths to downstream
    image generators.
    """

    TOOL_DEFINITION = {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": "Text-to-image search. Returns image results with titles and IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Descriptive text query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        download_dir: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        download_workers: int = 4,
    ) -> None:
        _require_requests()
        self.base_url = _serper_base_url(base_url)
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.download_dir = Path(download_dir or os.environ.get("IMAGE_DOWNLOAD_DIR") or DEFAULT_IMAGE_DIR)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(download_workers)))
        self._lock = threading.Lock()

    # ---- HTTP ----
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/images"
        headers = {
            "X-API-KEY": _read_serper_api_key(),
            "Content-Type": "application/json",
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(8, 1 + 2 * attempt))
        raise RuntimeError(f"Serper /images failed after {self.max_retries} retries: {last_err}")

    # ---- Download ----
    def _local_path_for(self, image_url: str) -> Path:
        digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:24]
        suffix = ".jpg"
        try:
            parsed = urlparse(image_url)
            ext = Path(parsed.path).suffix.lower()
            if ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
                suffix = ext
        except Exception:  # noqa: BLE001
            pass
        return self.download_dir / f"{digest}{suffix}"

    def _download(self, image_url: str) -> Optional[str]:
        if not image_url:
            return None
        target = self._local_path_for(image_url)
        if target.exists() and target.stat().st_size > 0:
            return str(target)
        try:
            resp = requests.get(image_url, timeout=self.timeout, stream=True)
            resp.raise_for_status()
            tmp = target.with_suffix(target.suffix + ".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            tmp.replace(target)
            return str(target)
        except Exception:  # noqa: BLE001
            return None

    # ---- Public API ----
    def call(self, query: str = "", top_k: int = 5, **_: object) -> List[Dict[str, str]]:
        """Return a list of {title, url, local_path, page_url}.

        The agent layer is responsible for assigning IMG_### identifiers.
        """
        query = (query or "").strip()
        if not query:
            return []
        try:
            data = self._post({"q": query, "num": int(top_k)})
            raw = data.get("images") or []
        except Exception:  # noqa: BLE001
            return []

        results: List[Dict[str, str]] = []
        candidate_urls: List[str] = []
        for item in raw[: int(top_k)]:
            image_url = (
                item.get("imageUrl")
                or item.get("image_url")
                or item.get("thumbnailUrl")
                or item.get("url")
                or ""
            )
            page_url = item.get("link") or item.get("source") or ""
            title = item.get("title") or "image"
            if not image_url:
                continue
            results.append({
                "title": title,
                "url": image_url,
                "local_path": "",
                "page_url": page_url,
            })
            candidate_urls.append(image_url)

        # Parallel download.
        futures = [self._executor.submit(self._download, u) for u in candidate_urls]
        for rec, fut in zip(results, futures):
            try:
                rec["local_path"] = fut.result() or ""
            except Exception:  # noqa: BLE001
                rec["local_path"] = ""

        # Drop entries without a usable local path.
        return [r for r in results if r["local_path"]]
