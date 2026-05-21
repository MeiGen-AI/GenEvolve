"""Evaluate generated images against ground-truth images with Gemini.

Input is the ``results.json`` written by ``scripts/generate_images.py``.
Each record should contain:

  - ``image_path``: generated image path, written by ``generate_images.py``.
  - ``gt_image``: ground-truth image path copied through from the input JSONL.
  - ``prompt``: the original user prompt.

The script writes per-sample judge results plus aggregate metrics:
faithfulness, visual_correctness, text_accuracy, aesthetics, and overall.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


SCORE_KEYS = ("faithfulness", "visual_correctness", "text_accuracy", "aesthetics")
OVERALL_WEIGHTS = {
    "faithfulness": 0.1,
    "visual_correctness": 0.4,
    "text_accuracy": 0.4,
    "aesthetics": 0.1,
}
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


SYSTEM_PROMPT = """You are a strict image-generation evaluation judge.
Compare the generated image with the ground-truth image and the user prompt.
Score each criterion as exactly 0, 0.5, or 1.

Criteria:
- faithfulness: Does the generated image satisfy the user prompt constraints?
- visual_correctness: Does it match stable visual entities, layout, attributes, and reference details in the ground-truth image?
- text_accuracy: If readable text is required, is it correct and legible? If no readable text is required, use 0.5 and set text_accuracy_na=true.
- aesthetics: Is the generated image visually polished and free of severe artifacts?

Return only one JSON object with:
{
  "rationale": "brief reason",
  "faithfulness": 0|0.5|1,
  "visual_correctness": 0|0.5|1,
  "text_accuracy": 0|0.5|1,
  "aesthetics": 0|0.5|1,
  "text_accuracy_na": true|false
}
"""


class RateLimiter:
    def __init__(self, rpm: float) -> None:
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_time - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_time = max(now, self._next_time) + self.min_interval


def _require_runtime() -> None:
    if requests is None:
        raise RuntimeError("requests is required. Install it with `pip install requests`.")
    if Image is None:
        raise RuntimeError("Pillow is required. Install it with `pip install pillow`.")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    rows.append({"split": "all", **summary["all"]})
    for split, metrics in sorted((summary.get("by_eval_type") or {}).items()):
        rows.append({"split": split, **metrics})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "split",
                "faithfulness",
                "visual_correctness",
                "text_accuracy",
                "aesthetics",
                "overall",
                "count_eval_success",
                "denominator",
                "missing_or_failed",
                "overall_missing_zero",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _candidate_paths(value: str, roots: Iterable[Path]) -> Iterable[Path]:
    p = Path(value).expanduser()
    if p.is_absolute():
        yield p
    else:
        for root in roots:
            yield (root / p).resolve()


def _resolve_path(value: Any, roots: Iterable[Path]) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for path in _candidate_paths(text, roots):
        if path.exists():
            return path
    return None


def _record_id(record: Dict[str, Any], index: int) -> str:
    return str(record.get("id", index))


def _find_gt_path(record: Dict[str, Any], roots: Iterable[Path]) -> Optional[Path]:
    for key in ("gt_image", "ground_truth", "target_image", "reference_image", "answer_image"):
        path = _resolve_path(record.get(key), roots)
        if path:
            return path
    return None


def _find_generated_path(record: Dict[str, Any], roots: Iterable[Path]) -> Optional[Path]:
    for key in ("image_path", "output_path", "generated_image", "gen_image"):
        path = _resolve_path(record.get(key), roots)
        if path:
            return path
    return None


def _image_to_b64(path: Path, max_side: int = 1536) -> str:
    assert Image is not None
    with Image.open(path) as im:
        im.load()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        elif im.mode == "L":
            im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_text(data: Dict[str, Any]) -> str:
    texts: List[str] = []
    for cand in data.get("candidates") or []:
        content = (cand or {}).get("content") or {}
        for part in content.get("parts") or []:
            if isinstance(part, dict) and part.get("text"):
                texts.append(str(part["text"]))
    return "\n".join(texts).strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    recovered: Dict[str, Any] = {}
    for key in (*SCORE_KEYS, "text_accuracy_na", "rationale"):
        if key == "rationale":
            match = re.search(r'["\']?rationale["\']?\s*:\s*"([^"]{0,1000})"', text, flags=re.S)
            if match:
                recovered[key] = re.sub(r"\s+", " ", match.group(1)).strip()
            continue
        match = re.search(rf'["\']?{re.escape(key)}["\']?\s*:\s*([^,\n}}\]]+)', text)
        if not match:
            continue
        raw = match.group(1).strip().strip('"\'')
        if key == "text_accuracy_na":
            recovered[key] = raw.lower() in {"true", "1", "yes"}
        else:
            try:
                recovered[key] = float(raw)
            except ValueError:
                pass
    if recovered:
        return recovered
    raise ValueError(f"Could not parse judge JSON: {text[:300]}")


def _normalize_score(value: Any) -> float:
    try:
        num = float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid score: {value!r}") from exc
    allowed = (0.0, 0.5, 1.0)
    return min(allowed, key=lambda x: abs(x - num))


def _normalize_scores(obj: Dict[str, Any]) -> Dict[str, Any]:
    scores: Dict[str, Any] = {}
    for key in SCORE_KEYS:
        if key not in obj and key == "text_accuracy" and obj.get("text_accuracy_na") is True:
            scores[key] = 0.5
        else:
            scores[key] = _normalize_score(obj.get(key))
    scores["text_accuracy_na"] = bool(obj.get("text_accuracy_na", False))
    scores["rationale"] = str(obj.get("rationale") or "").strip()
    scores["overall"] = round(sum(scores[k] * OVERALL_WEIGHTS[k] for k in SCORE_KEYS), 4)
    return scores


def _judge_payload(prompt: str, generated_b64: str, gt_b64: str) -> Dict[str, Any]:
    user_text = (
        "User prompt:\n"
        f"{prompt}\n\n"
        "Image 1 is the generated image. Image 2 is the ground-truth image. "
        "Evaluate Image 1 against the prompt and Image 2."
    )
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": SYSTEM_PROMPT + "\n\n" + user_text},
                    {"inlineData": {"mimeType": "image/jpeg", "data": generated_b64}},
                    {"inlineData": {"mimeType": "image/jpeg", "data": gt_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }


def _call_gemini(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    generated_path: Path,
    gt_path: Path,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    assert requests is not None
    url = endpoint.format(model=model)
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = _judge_payload(prompt, _image_to_b64(generated_path), _image_to_b64(gt_path))
    last_err: Optional[Exception] = None
    for attempt in range(max(1, max_retries)):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"Gemini API error: {data.get('error')}")
            text = _extract_text(data)
            obj = _parse_json_object(text)
            return _normalize_scores(obj)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < max_retries:
                time.sleep(min(30, 2 + 2 * attempt))
    raise RuntimeError(f"Gemini judge failed after {max_retries} attempts: {last_err}")


def _eval_type(record: Dict[str, Any]) -> Optional[str]:
    for key in ("eval_type", "tier", "track", "split"):
        value = record.get(key)
        if value:
            return str(value)
    meta = record.get("meta")
    if isinstance(meta, dict):
        for key in ("eval_type", "tier", "track", "split"):
            value = meta.get(key)
            if value:
                return str(value)
    return None


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def summarize_subset(subset: List[Dict[str, Any]], denominator: int) -> Dict[str, Any]:
        ok_scores = [
            row["scores"]
            for row in subset
            if row.get("eval_success") and isinstance(row.get("scores"), dict)
        ]
        out: Dict[str, Any] = {}
        for key in (*SCORE_KEYS, "overall"):
            out[key] = round(sum(float(s.get(key, 0.0)) for s in ok_scores) / len(ok_scores), 4) if ok_scores else 0.0
        out["count_eval_success"] = len(ok_scores)
        out["denominator"] = denominator
        out["missing_or_failed"] = max(0, denominator - len(ok_scores))
        total_overall = sum(float(s.get("overall", 0.0)) for s in ok_scores)
        out["overall_missing_zero"] = round(total_overall / denominator, 4) if denominator else 0.0
        return out

    summary: Dict[str, Any] = {
        "total_cases": len(rows),
        "all": summarize_subset(rows, len(rows)),
        "by_eval_type": {},
        "missing_or_failed_ids": [
            row.get("id")
            for row in rows
            if not row.get("eval_success")
        ],
    }
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        split = _eval_type(row)
        if split:
            groups.setdefault(split, []).append(row)
    summary["by_eval_type"] = {
        split: summarize_subset(group_rows, len(group_rows))
        for split, group_rows in sorted(groups.items())
    }
    return summary


def _evaluate_one(
    index: int,
    record: Dict[str, Any],
    *,
    roots: List[Path],
    endpoint: str,
    api_key: str,
    model: str,
    timeout: int,
    max_retries: int,
    limiter: Optional[RateLimiter],
) -> Dict[str, Any]:
    out = dict(record)
    out["id"] = _record_id(record, index)
    gen_path = _find_generated_path(record, roots)
    gt_path = _find_gt_path(record, roots)
    if not gen_path:
        out.update({"eval_success": False, "eval_error": "generated image path not found"})
        return out
    if not gt_path:
        out.update({"eval_success": False, "eval_error": "ground-truth image path not found"})
        return out
    if limiter:
        limiter.acquire()
    prompt = str(record.get("prompt") or record.get("question") or record.get("gen_prompt") or "")
    try:
        scores = _call_gemini(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            prompt=prompt,
            generated_path=gen_path,
            gt_path=gt_path,
            timeout=timeout,
            max_retries=max_retries,
        )
        out.update(
            {
                "eval_success": True,
                "eval_error": "",
                "gt_image_resolved": str(gt_path),
                "generated_image_resolved": str(gen_path),
                "scores": scores,
            }
        )
    except Exception as exc:  # noqa: BLE001
        out.update(
            {
                "eval_success": False,
                "eval_error": str(exc),
                "gt_image_resolved": str(gt_path),
                "generated_image_resolved": str(gen_path),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated images with Gemini.")
    parser.add_argument("--results", required=True, help="results.json from scripts/generate_images.py")
    parser.add_argument("--output-json", default=None, help="Per-sample output JSON")
    parser.add_argument("--summary-json", default=None, help="Aggregate summary JSON")
    parser.add_argument("--summary-csv", default=None, help="Aggregate summary CSV")
    parser.add_argument("--gt-root", action="append", default=[], help="Root used to resolve relative gt_image paths")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("GEMINI_JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--endpoint", default=os.environ.get("GEMINI_API_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--rpm", type=float, default=0, help="Global request-per-minute limit; 0 disables limiting")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Reuse successful rows from output-json")
    args = parser.parse_args()

    _require_runtime()
    if not args.api_key:
        raise SystemExit("Set GOOGLE_API_KEY or pass --api-key for Gemini evaluation.")

    results_path = Path(args.results).resolve()
    records = _load_json(results_path)
    if not isinstance(records, list):
        raise SystemExit("--results must be a JSON array.")

    output_json = Path(args.output_json).resolve() if args.output_json else results_path.with_name("results_eval.json")
    summary_json = Path(args.summary_json).resolve() if args.summary_json else results_path.with_name("summary.json")
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else results_path.with_name("summary.csv")

    roots = [results_path.parent, Path.cwd()]
    roots.extend(Path(p).resolve() for p in args.gt_root)

    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume and output_json.exists():
        old = _load_json(output_json)
        if isinstance(old, list):
            existing = {str(row.get("id")): row for row in old if isinstance(row, dict) and row.get("eval_success")}

    limiter = RateLimiter(args.rpm) if args.rpm and args.rpm > 0 else None
    rows: List[Optional[Dict[str, Any]]] = [None] * len(records)

    def run_or_reuse(i: int, record: Dict[str, Any]) -> Dict[str, Any]:
        rid = _record_id(record, i)
        if rid in existing:
            return existing[rid]
        return _evaluate_one(
            i,
            record,
            roots=roots,
            endpoint=args.endpoint,
            api_key=args.api_key,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            limiter=limiter,
        )

    with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as executor:
        futures = {
            executor.submit(run_or_reuse, i, record): i
            for i, record in enumerate(records)
            if isinstance(record, dict)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            rows[idx] = fut.result()
            done += 1
            if done == len(futures) or done % 10 == 0:
                compact = [row for row in rows if row is not None]
                _write_json(output_json, compact)
                print(f"[GenEvolveEval] {done}/{len(futures)} evaluated", flush=True)

    final_rows = [row for row in rows if row is not None]
    _write_json(output_json, final_rows)
    summary = _summarize(final_rows)
    summary["results_json"] = str(results_path)
    summary["eval_json"] = str(output_json)
    summary["judge_model"] = args.model
    _write_json(summary_json, summary)
    _write_csv(summary_csv, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
