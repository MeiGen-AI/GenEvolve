#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-compatible GenEvolve image evaluation with Gemini.

This is the public version of the evaluator used for the GenEvolve benchmark
numbers. It keeps the same judge prompt, image order, score normalization, and
overall formula as the paper evaluation. The only repository-specific change is
path handling: it accepts ``results.json`` produced by ``scripts/generate_images.py``.

Expected per-sample fields:

  - ``prompt`` or ``question``: original image-generation request.
  - ``output_path`` or ``image_path``: generated image path.
  - ``gt_image``: ground-truth image path.
  - optional ``meta.eval_type`` or top-level ``eval_type`` for split summaries.

The judge call uses an OpenAI-compatible multimodal chat-completions endpoint.
For strict reproducibility, use the same model family as the paper, e.g.
``gemini-3.1-pro-preview``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI, RateLimitError
from tqdm import tqdm


LLM_TIMEOUT_SEC = float(os.environ.get("GENEVOLVE_EVAL_TIMEOUT_SEC", "300"))
LLM_MAX_TRY = int(os.environ.get("GENEVOLVE_EVAL_MAX_TRY", "20"))
LLM_MAX_TOKENS = 8192
LLM_TEMPERATURE = 0.0
MAX_SIDE = 4096
JPEG_QUALITY = 100
SCORE_KEYS = ("faithfulness", "visual_correctness", "text_accuracy", "aesthetics")
SUMMARY_SCORE_KEYS = (*SCORE_KEYS, "overall")


class RateLimiter:
    """Thread-safe global RPM limiter shared by all evaluation workers."""

    def __init__(self, rpm: float):
        self.min_interval = 60.0 / float(rpm) if rpm and rpm > 0 else 0.0
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


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: str, obj: Any) -> None:
    dirname = os.path.dirname(path)
    _ensure_dir(dirname)
    tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _write_csv(path: str, summary: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    fields = [
        "split",
        "count_eval_success",
        "denominator",
        "missing_or_failed",
        "faithfulness",
        "visual_correctness",
        "text_accuracy",
        "aesthetics",
        "overall",
        "overall_missing_zero",
    ]
    rows: List[Dict[str, Any]] = []
    rows.append({"split": "all", **summary["all"]})
    for split, metrics in sorted(summary.get("by_eval_type", {}).items()):
        rows.append({"split": f"eval_type:{split}", **metrics})
    for split, metrics in sorted(summary.get("by_category", {}).items()):
        rows.append({"split": f"category:{split}", **metrics})
    for split, metrics in sorted(summary.get("by_difficulty", {}).items()):
        rows.append({"split": f"difficulty:{split}", **metrics})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _round_01(x: float) -> float:
    return round(_clip01(x), 2)


def _encode_image_to_data_url(path: str, max_side: int = MAX_SIDE, quality: int = JPEG_QUALITY) -> str:
    import base64
    import io

    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(quality), optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


def _build_user_message(sample_id: str, prompt: str, gen_path: str, gt_path: str) -> dict:
    text = (
        f"Sample id: {sample_id}\n\n"
        f"Task prompt (image requirement):\n{prompt}\n\n"
        "Image 1: the **generated image** (to be evaluated).\n"
        "Image 2: the **ground-truth reference image**.\n\n"
        "Output a single JSON object with the three scores and a short rationale."
    )
    content = [{"type": "text", "text": text}]
    for path in [gen_path, gt_path]:
        try:
            url = _encode_image_to_data_url(path)
            content.append({"type": "image_url", "image_url": {"url": url}})
        except Exception as exc:  # noqa: BLE001
            content.append({"type": "text", "text": f"\n[WARN] failed to load image: {path}, err={exc}\n"})
    return {"role": "user", "content": content}


SYSTEM_PROMPT = r"""You are a strict and professional expert evaluator for AI-generated image grounded with world knowledge (MODEL EVALUATION).

You will receive:
1) A task prompt (what the image must show).
2) Image 1: the generated image (model output to be evaluated).
3) Image 2: the ground-truth reference image (a strong reference implementation).

All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

Critical clarification (VERY IMPORTANT):
- This is NOT a pixel-level similarity task.
- Image 2 (GT) is a REFERENCE for intended identity, key grounded details, and stable visual attributes.
  Image 1 may use a different camera angle/layout as long as it still satisfies the prompt.
- Focus on whether prompt-required, externally-checkable (search-grounded) details are correctly AND verifiably realized in Image 1.
- Do NOT assume correctness if a key detail is not clearly visible/readable. If unverifiable, score lower.

Output format (MUST follow exactly):
Output ONLY one valid JSON object with EXACTLY these keys:
{
  "rationale": string,
  "faithfulness": number,
  "visual_correctness": number,
  "text_accuracy": number,
  "aesthetics": number,
  "text_accuracy_na": boolean
}
SCORING SCALE (VERY IMPORTANT):
- Each score MUST be exactly one of: 0, 0.5, 1
- 1 (Exemplary) is rare and requires perfect success for that dimension.
- 0.5 (Conditional) means mostly correct but not perfect.
- 0 (Rejected) means failed on important requirements.

- "rationale" must be 5–10 short sentences, evidence-based, referring only to what is visible.
- "text_accuracy_na" should be true if the prompt does not require any readable text, otherwise it should be false.

Implicit required step (ENFORCED via rationale):
- In the rationale, you MUST explicitly list the extracted prompt hard constraints (2–5, or more if needed) BEFORE scoring.
  If you cannot identify the constraints, you must still list what you believe are the hard constraints.

Evaluation procedure (follow silently, but the rationale MUST reflect it):
1) Extract the prompt’s TOP hard constraints (2–5, or more if needed): required subjects/identities, setting/props,
   relations/counts, required style, and any externally-checkable requirements (readable text/landmark/logo/badge/version/year/etc.).
2) Score Image 1 against the constraints. Use Image 2 only as a reference for stable identity/visual attributes and grounded evidence.
3) If a key requirement is not verifiable (too small/blurred/occluded/warped), do NOT assume it is correct; score lower.
4) Assessment of the primary subjects' visual identity correctness and consistency is mandatory in every case.

Boundary between visual_correctness vs text_accuracy:
- Visual-only grounded cues (subject visual features, logo SHAPE, badge EMBLEM geometry, landmark facade/massing, outfit/weapon silhouette, object geometry)
  belong to visual_correctness.
- Any grounded cue that must be READ as text (spelling, year numbers, titles, institution names, badge text) belongs to text_accuracy.

==========================
STRICT 3-LEVEL RUBRICS
(Each dimension uses ONLY {0, 0.5, 1})
==========================

1) faithfulness (overall prompt adherence: presence & structure only; not GT-identity correctness):
- This score does NOT require matching GT’s exact identity or fine-grained visual features; it focuses on whether Image 1 includes the prompt-requested elements and scene structure (who/what is present, what is happening, where it happens, and the required style/format).

(Exemplary) Score = 1 ONLY IF:
- Image 1 clearly includes everything the prompt asks for in terms of visible content and structure:
  all required subjects/entities are present, the required setting and key props appear,
  required actions/relations/counts are shown, and the required style/format is followed.
- Any required in-scene evidence elements requested by the prompt (e.g., a plaque/sign, a map, a report paper, a badge) are present as elements.

(Conditional) Score = 0.5 ONLY IF:
- Image 1 includes almost all prompt-requested content and structure, with only minor omissions or minor staging differences
  that do not change what the scene is supposed to depict (e.g., small placement differences, slight simplification of a secondary prop).

(Rejected) Score = 0 IF:
- One or more prompt-requested essential elements are not shown at all, or the scene structure clearly does not match the prompt’s request
  (e.g., missing a required subject/entity, missing the required setting, missing the required key prop/evidence element,
  missing the requested action/relationship/count, or not following the requested style/format).

2) visual_correctness (GT visual-feature agreement is the core; extremely strict):
(Exemplary) Score = 1 ONLY IF:
- The prompt-required primary subjects/objects in Image 1 match the GT reference (Image 2) in visual characteristics
  with NO substantive changes.
- This means: the same face/hairstyle silhouette, the same armor/clothing design and key colors/patterns,
  the same distinctive props/object geometry, the same emblem/logo/landmark facade/massing cues when applicable, etc.
- Any meaningful difference in these stable visual features disqualifies a score of 1.

(Conditional) Score = 0.5 ONLY IF:
- Image 1 can still be considered the same overall visual instance as the GT, and the differences are limited to relatively minor variations, allowing some changes to the visual features (face, hairstyle, armor design, key colors/patterns, key prop shapes), while the overall identity and major visual features remain recognizable and broadly consistent.
- IMPORTANT: "same role archetype" (generic knight/princess/warrior) alone does NOT qualify for 0.5.

(Rejected) Score = 0 IF:
- Any substantive mismatch vs GT in stable visual features of the required subjects/objects
  (different face/hair/armor design/color scheme/emblem/prop geometry/landmark cues),
  even if the overall scene still looks plausible or stylistically similar.

3) text_accuracy (required readable text; ALL relevant text must be correct AND very clearly readable; NO partial credit for wrong text):
Rule:
- If the prompt does NOT require any readable text: you MUST output "text_accuracy_na": true and "text_accuracy": 0.5 in the JSON. In your rationale state that the prompt did not require readable text.
- If the prompt DOES require readable text: output "text_accuracy_na": false and score "text_accuracy" (0, 0.5, or 1) per the criteria below.

(Exemplary) Score = 1 ONLY IF:
- ALL required text AND any prompt-involved text elements are:
  (a) present,
  (b) very clearly readable (crisp, unambiguous),
  (c) correct and consistent with the prompt’s requirements.
(Conditional) Score = 0.5 ONLY IF:
- Much of the required/prompt-involved text is readable and generally correct, and although parts may contain inaccuracies, omissions, or deviations, the overall meaning remains clear and is not seriously inconsistent with the prompt requirements.
(Rejected) Score = 0 IF:
- Any required/prompt-involved text is missing, unclear, not very readable, gibberish, placeholder, OR incorrect.
- Even if perfectly readable, if content is not correct, text_accuracy MUST be 0.

4) aesthetics:
(Exemplary) Score = 1 ONLY IF:
- Masterpiece-level composition and polish, AND Image 1 is NOT worse than GT in overall aesthetic quality.
(Conditional) Score = 0.5 ONLY IF:
- Very beautiful and polished, but slightly worse than GT (ONLY slightly) OR slightly less refined than top-tier.
(Rejected) Score = 0 IF:
- Anything clearly worse than GT in a noticeable way, OR merely average/OK-looking, OR cluttered/awkward framing,
  OR visible artifacts/noise that harm the overall appeal.

Rationale requirements (MANDATORY):
- Start with: "Constraints:" and list the extracted constraints (2–5, or more if needed).
- State whether the prompt required readable text; if not required, output "text_accuracy_na": true and "text_accuracy": 0.5 in the JSON and say so in the rationale.
- Mention 2–5 key comparisons (or more if needed) to GT focused on stable identity/visual traits (NOT demanding identical layout).
- Keep within 10 sentences.

Output JSON only. No markdown. No extra text."""


def _parse_flat_score_json(content: str) -> Dict[str, Any]:
    """Recover score fields when the judge emits malformed JSON strings."""
    text = (content or "").strip()
    if not text:
        return {}
    text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    obj: Dict[str, Any] = {}
    for key in (*SCORE_KEYS, "score"):
        match = re.search(rf'["\']?{re.escape(key)}["\']?\s*:\s*([^,\n}}\]]+)', text)
        if not match:
            continue
        raw = match.group(1).strip().strip('"\'')
        try:
            obj[key] = float(raw)
        except ValueError:
            obj[key] = raw
    match = re.search(r'["\']?text_accuracy_na["\']?\s*:\s*([^,\n}}\]]+)', text)
    if match:
        raw = match.group(1).strip().strip('"\'').lower()
        obj["text_accuracy_na"] = raw in {"true", "1", "yes"}
    match = re.search(r'["\']?rationale["\']?\s*:\s*"(.{0,1000})', text, flags=re.DOTALL)
    if match:
        obj["rationale"] = re.sub(r"\s+", " ", match.group(1)).strip()
    return obj


def _parse_llm_json(content: str) -> dict:
    """Parse LLM-returned JSON, tolerating extra commas and surrounding text."""
    content = (content or "").strip()
    if not content:
        raise ValueError("empty_content")
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9]*\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        content = content[start : end + 1]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    content = re.sub(r",\s*}", "}", content)
    content = re.sub(r",\s*]", "]", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fallback = _parse_flat_score_json(content)
        if fallback:
            return fallback
        raise ValueError("LLM output is not valid JSON.")


def _call_llm_json(client: OpenAI, model: str, messages: List[dict], max_try: int = LLM_MAX_TRY) -> dict:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_try + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT_SEC,
            )
            msg = getattr(resp, "choices", [None])[0]
            if msg is not None:
                msg = getattr(msg, "message", None)
            content = getattr(msg, "content", None) if msg else None
            if not isinstance(content, str):
                raise RuntimeError("empty_content")
            obj = _parse_llm_json(content)
            if not isinstance(obj, dict):
                raise ValueError("LLM output is not a JSON object.")
            return obj
        except RateLimitError as exc:
            last_exc = exc
            if attempt % 5 == 0:
                _log(f"[429 RETRY] attempt={attempt}/{max_try}")
            time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random()))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            _log(f"[LLM ERROR] attempt={attempt}/{max_try} err={str(exc)[:150]}")
            if attempt >= max_try:
                break
            time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random()))
    raise last_exc if last_exc else RuntimeError("LLM call failed")


def _normalize_scores(obj: Any) -> Tuple[float, float, Optional[float], float, str]:
    """Return (faithfulness, visual_correctness, text_accuracy|None, aesthetics, rationale)."""
    fallback: Tuple[float, float, Optional[float], float, str] = (0.0, 0.0, 0.0, 0.0, "")
    if not isinstance(obj, dict):
        return fallback
    f = _round_01(float(obj.get("faithfulness", 0)))
    v = _round_01(float(obj.get("visual_correctness", 0)))
    text_na = obj.get("text_accuracy_na")
    if text_na in (True, "true", "True", 1):
        t: Optional[float] = None
    else:
        try:
            t = _round_01(float(obj.get("text_accuracy", 0)))
        except (TypeError, ValueError):
            t = None
    a = _round_01(float(obj.get("aesthetics", 0)))
    rationale = str(obj.get("rationale", ""))[:1000]
    return (f, v, t, a, rationale)


def _candidate_paths(value: Any, roots: List[Path]) -> List[Path]:
    if value is None:
        return []
    raw = str(value).strip()
    if not raw:
        return []
    p = Path(raw).expanduser()
    if p.is_absolute():
        return [p]
    return [(root / p).resolve() for root in roots]


def _resolve_path(value: Any, roots: List[Path]) -> Optional[str]:
    for path in _candidate_paths(value, roots):
        if path.exists():
            return str(path)
    return None


def _generated_path(entry: dict, roots: List[Path]) -> Optional[str]:
    for key in ("output_path", "image_path", "generated_image", "gen_image"):
        found = _resolve_path(entry.get(key), roots)
        if found:
            return found
    return None


def _gt_path(entry: dict, roots: List[Path]) -> Optional[str]:
    for key in ("gt_image", "ground_truth", "target_image", "reference_image"):
        found = _resolve_path(entry.get(key), roots)
        if found:
            return found
    meta = entry.get("meta")
    if isinstance(meta, dict):
        for key in ("gt_image", "ground_truth", "target_image", "gt_copied_path"):
            found = _resolve_path(meta.get(key), roots)
            if found:
                return found
    return None


def _sample_success(entry: dict) -> bool:
    if "success" in entry:
        return bool(entry.get("success"))
    if entry.get("image_status") == "ok":
        return True
    return bool(_generated_path(entry, [Path.cwd()]))


def load_results(results_path: str) -> Tuple[List[dict], Path]:
    data = _read_json(results_path)
    if not isinstance(data, list):
        raise ValueError("results.json must be a JSON array.")
    base_dir = Path(results_path).resolve().parent
    return data, base_dir


def run_one_eval(
    entry: dict,
    roots: List[Path],
    client: OpenAI,
    model: str,
    rate_limiter: Optional[RateLimiter] = None,
) -> Tuple[str, bool, dict]:
    """Returns (sample_id, ok, result_dict)."""
    sample_id = str(entry.get("id", ""))
    prompt = (entry.get("prompt") or entry.get("question") or "").strip()
    out_path = _generated_path(entry, roots)
    gt_path = _gt_path(entry, roots)

    if not prompt:
        return sample_id, False, {"error": "empty_prompt"}
    if not out_path or not os.path.isfile(out_path):
        return sample_id, False, {"error": "missing_generated_image"}
    if not gt_path or not os.path.isfile(gt_path):
        return sample_id, False, {"error": "missing_gt_image"}

    try:
        user_msg = _build_user_message(sample_id, prompt, out_path, gt_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_msg,
        ]
        if rate_limiter is not None:
            rate_limiter.acquire()
        obj = _call_llm_json(client, model, messages)
        f, v, t, a, rationale = _normalize_scores(obj)
        t_val = 0.5 if t is None else t
        text_accuracy_na = t is None
        overall = round(0.1 * f + 0.4 * v + 0.4 * t_val + 0.1 * a, 2)
        payload = {
            "rationale": rationale,
            "faithfulness": f,
            "visual_correctness": v,
            "text_accuracy": t_val,
            "aesthetics": a,
            "overall": overall,
        }
        if text_accuracy_na:
            payload["text_accuracy_na"] = True
        return sample_id, True, payload
    except Exception as exc:  # noqa: BLE001
        return sample_id, False, {"error": str(exc)[:300]}


def avg_score_dict(score_rows: List[dict]) -> dict:
    """Average score dictionaries using the same text-NA handling as the evaluator."""
    if not score_rows:
        return {key: 0.0 for key in SUMMARY_SCORE_KEYS}
    out: Dict[str, float] = {}
    for key in ("faithfulness", "visual_correctness", "aesthetics"):
        vals = [float(s.get(key, 0)) for s in score_rows if isinstance(s.get(key), (int, float))]
        out[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    text_vals = [
        float(s.get("text_accuracy", 0))
        for s in score_rows
        if not s.get("text_accuracy_na") and isinstance(s.get("text_accuracy"), (int, float))
    ]
    out["text_accuracy"] = round(sum(text_vals) / len(text_vals), 4) if text_vals else 0.5
    out["overall"] = round(
        0.1 * out["faithfulness"]
        + 0.4 * out["visual_correctness"]
        + 0.4 * out["text_accuracy"]
        + 0.1 * out["aesthetics"],
        4,
    )
    return out


def build_summary_for_output(rows: List[dict]) -> dict:
    """Build summary rows matching the released benchmark splits."""
    summary: Dict[str, Any] = {"by_eval_type": {}, "overall_avg": {}}
    for eval_type in sorted({_row_eval_type(row) for row in rows}):
        score_rows = [
            row["scores"]
            for row in rows
            if _row_eval_type(row) == eval_type
            and row.get("eval_success") is True
            and isinstance(row.get("scores"), dict)
        ]
        summary["by_eval_type"][eval_type] = {**avg_score_dict(score_rows), "count": len(score_rows)}
    ok_rows = [row for row in rows if row.get("eval_success") is True and isinstance(row.get("scores"), dict)]
    all_scores = [row["scores"] for row in ok_rows]
    summary["overall_avg"] = {**avg_score_dict(all_scores), "count": len(all_scores)}
    return summary


def build_output_with_summary(data: List[dict], results_by_id: Dict[str, dict]) -> Tuple[List[dict], dict]:
    """Build output list in original order and append benchmark split summaries."""
    sorted_items: List[dict] = []
    for index, entry in enumerate(data):
        sample_id = str(entry.get("id", index))
        record = results_by_id.get(sample_id)
        if record is None:
            record = dict(entry)
            record["eval_success"] = False
            record["scores"] = None
            record["error"] = "not_evaluated"
        sorted_items.append(record)

    summary = build_summary_for_output(sorted_items)

    output_items = list(sorted_items)
    for eval_type, metrics in sorted(summary["by_eval_type"].items()):
        output_items.append(
            {
                "summary_type": f"eval_type:{eval_type}",
                "avg_scores": {key: metrics[key] for key in SUMMARY_SCORE_KEYS},
                "count": metrics["count"],
            }
        )
    output_items.append(
        {
            "summary_type": "overall_avg",
            "avg_scores": {key: summary["overall_avg"][key] for key in SUMMARY_SCORE_KEYS},
            "count": summary["overall_avg"]["count"],
        }
    )
    return output_items, summary


def eval_rows(data: Any) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict) and "summary_type" not in row]


def _row_eval_type(row: Dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(meta.get("eval_type") or row.get("eval_type") or row.get("tier") or "unknown")


def _row_category(row: Dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(row.get("category") or meta.get("category") or "unknown")


def _row_difficulty(row: Dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(row.get("difficulty") or meta.get("difficulty") or "unknown")


def _row_case_id(row: Dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(meta.get("case_id") or meta.get("source_id") or row.get("id"))


def avg_metric(rows: List[Dict[str, Any]], key: str) -> float:
    vals: List[float] = []
    for row in rows:
        scores = row.get("scores")
        if not isinstance(scores, dict):
            continue
        value = scores.get(key)
        if isinstance(value, (int, float)):
            vals.append(float(value))
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def summarize_subset(rows: List[Dict[str, Any]], denominator: int) -> Dict[str, Any]:
    ok = [row for row in rows if row.get("eval_success") is True and isinstance(row.get("scores"), dict)]
    out = {key: avg_metric(ok, key) for key in SUMMARY_SCORE_KEYS}
    out["count_eval_success"] = len(ok)
    out["denominator"] = denominator
    out["missing_or_failed"] = max(0, denominator - len(ok))
    out["overall_missing_zero"] = round(out["overall"] * len(ok) / denominator, 4) if denominator else 0.0
    return out


def build_benchmark_summary(eval_path: Path, total_cases: int) -> Dict[str, Any]:
    rows = eval_rows(_read_json(str(eval_path)))
    denominator = total_cases or len(rows)
    summary: Dict[str, Any] = {
        "eval_json": str(eval_path),
        "total_cases": denominator,
        "all": summarize_subset(rows, denominator),
        "by_eval_type": {},
        "by_category": {},
        "by_difficulty": {},
        "missing_case_ids": [],
        "failed_case_ids": [],
    }
    for row in rows:
        case_id = _row_case_id(row)
        if not _sample_success(row):
            summary["missing_case_ids"].append(case_id)
        elif row.get("eval_success") is not True:
            summary["failed_case_ids"].append(case_id)
    eval_types = sorted({_row_eval_type(row) for row in rows})
    for eval_type in eval_types:
        subset = [row for row in rows if _row_eval_type(row) == eval_type]
        summary["by_eval_type"][eval_type] = summarize_subset(subset, len(subset))
    categories = sorted({_row_category(row) for row in rows})
    for category in categories:
        subset = [row for row in rows if _row_category(row) == category]
        summary["by_category"][category] = summarize_subset(subset, len(subset))
    difficulties = sorted({_row_difficulty(row) for row in rows})
    for difficulty in difficulties:
        subset = [row for row in rows if _row_difficulty(row) == difficulty]
        summary["by_difficulty"][difficulty] = summarize_subset(subset, len(subset))
    return summary


def main() -> None:
    default_model = os.getenv("GENEVOLVE_EVAL_MODEL", "gemini-3.1-pro-preview")
    parser = argparse.ArgumentParser(
        description="Paper-compatible GenEvolve evaluation: prompt + generated image + GT image -> Gemini scoring"
    )
    parser.add_argument("--results", "-r", required=True, help="results.json produced by scripts/generate_images.py")
    parser.add_argument("--output-json", "-o", default=None, help="Output JSON path (default: <results_dir>/results_eval.json)")
    parser.add_argument("--summary-json", default=None, help="Benchmark summary JSON path (default: <results_dir>/summary.json)")
    parser.add_argument("--summary-csv", default=None, help="Benchmark summary CSV path (default: <results_dir>/summary.csv)")
    parser.add_argument("--gt-root", action="append", default=[], help="Root used to resolve relative gt_image paths")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key (or set OPENAI_API_KEY)")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible base_url (or set OPENAI_API_BASE)")
    parser.add_argument("--model", default=default_model, help="Judge model name")
    parser.add_argument("--max-workers", "--parallel", dest="max_workers", type=int, default=8, help="Parallelism")
    parser.add_argument("--rpm", type=float, default=0, help="Global request-per-minute limit across all workers; 0 disables limiting")
    parser.add_argument("--total-cases", type=int, default=0, help="Denominator for overall_missing_zero; defaults to result count")
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in output-json")
    args = parser.parse_args()

    api_key = (args.api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    api_base = (args.api_base or os.getenv("OPENAI_API_BASE", "")).strip()
    if not api_key:
        raise SystemExit("Please provide --api-key or set OPENAI_API_KEY")
    if api_base and not api_base.endswith("/"):
        api_base += "/"

    results_path = Path(args.results).resolve()
    if not results_path.exists():
        raise SystemExit(f"results file not found: {results_path}")
    data, base_dir = load_results(str(results_path))
    roots = [base_dir, Path.cwd()]
    roots.extend(Path(root).resolve() for root in args.gt_root)

    eval_path = Path(args.output_json).resolve() if args.output_json else (base_dir / "results_eval.json")
    summary_json = Path(args.summary_json).resolve() if args.summary_json else (base_dir / "summary.json")
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else (base_dir / "summary.csv")

    existing: Dict[str, dict] = {}
    if args.resume and eval_path.exists():
        raw = _read_json(str(eval_path))
        if isinstance(raw, dict):
            existing = raw
        elif isinstance(raw, list):
            for row in raw:
                if not isinstance(row, dict) or "summary_type" in row:
                    continue
                sample_id = row.get("id")
                if sample_id is not None:
                    existing[str(sample_id)] = row

    to_run = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        sample_id = str(entry.get("id", index))
        if args.resume and sample_id in existing:
            old = existing[sample_id] or {}
            if isinstance(old.get("scores"), dict) or "eval_success" in old:
                continue
        if not _sample_success(entry):
            continue
        if not _gt_path(entry, roots) or not _generated_path(entry, roots):
            continue
        to_run.append(entry)

    _log(f"results total={len(data)}, pending_eval={len(to_run)}")
    client = OpenAI(api_key=api_key, base_url=api_base or None, timeout=LLM_TIMEOUT_SEC)
    model = (args.model or default_model).strip()
    results_by_id = dict(existing)
    rate_limiter = RateLimiter(args.rpm) if args.rpm and args.rpm > 0 else None
    _log(f"model={model} workers={args.max_workers} rpm={args.rpm}")

    if to_run:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_one_eval, entry, roots, client, model, rate_limiter): entry
                for entry in to_run
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Eval"):
                entry = futures[fut]
                try:
                    sample_id, ok, payload = fut.result()
                    record = dict(entry)
                    record["eval_success"] = ok
                    if ok:
                        record["scores"] = payload
                        record["error"] = None
                    else:
                        record["scores"] = None
                        record["error"] = payload.get("error")
                    results_by_id[sample_id] = record
                except Exception as exc:  # noqa: BLE001
                    sample_id = str(entry.get("id", ""))
                    record = dict(entry)
                    record["eval_success"] = False
                    record["scores"] = None
                    record["error"] = str(exc)[:200]
                    results_by_id[sample_id] = record
                output_items, _ = build_output_with_summary(data, results_by_id)
                _write_json(str(eval_path), output_items)

    output_items, output_summary = build_output_with_summary(data, results_by_id)
    _write_json(str(eval_path), output_items)
    benchmark_summary = build_benchmark_summary(eval_path, args.total_cases or len(data))
    _write_json(str(summary_json), benchmark_summary)
    _write_csv(str(summary_csv), benchmark_summary)

    _log(f"Eval output: {eval_path}")
    _log(f"Eval-type summaries: {output_summary['by_eval_type']}")
    _log(f"Overall Avg: {output_summary['overall_avg']}")
    print(json.dumps(benchmark_summary, ensure_ascii=False, indent=2))
    print(f"summary_json={summary_json}")
    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
