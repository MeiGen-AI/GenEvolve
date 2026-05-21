"""Run the GenEvolve agent on a single prompt or a JSON/JSONL file of prompts.

This script does not generate images --- it only runs the agent and saves the
resulting prompt-reference programs ``z = (gen_prompt, reference_images)`` to
``results.json``. To turn the saved programs into images, run
``scripts/generate_images.py`` next.

Examples
--------
Run a single prompt:

    python scripts/run_agent.py \
        --prompt "Draw a cyberpunk Eiffel Tower at sunset." \
        --output-dir runs/single

Run a batch of prompts from a JSONL file. Each row may use either ``prompt`` or
the Hugging Face benchmark field ``question``:

    python scripts/run_agent.py \
        --input prompts.jsonl \
        --output-dir runs/batch

Environment variables
---------------------
``OPENAI_BASE_URL``  : base URL of the OpenAI-compatible inference server.
``OPENAI_API_KEY``   : API key (use ``EMPTY`` for a local vLLM server).
``SERPER_API_KEY``   : API key for https://serper.dev (text + image search).
``IMAGE_DOWNLOAD_DIR``: where to cache image_search downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# Allow running from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genevolve import GenEvolveAgent  # noqa: E402


def _load_prompts(prompt: str | None, input_path: str | None, max_samples: int | None) -> List[Dict[str, Any]]:
    if prompt:
        return [{"id": "single", "prompt": prompt}]
    if not input_path:
        raise ValueError("Provide either --prompt or --input.")

    path = Path(input_path).resolve()
    raw = path.read_text(encoding="utf-8")
    items: List[Dict[str, Any]] = []
    if raw.strip().startswith("["):
        items = json.loads(raw)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if max_samples is not None:
        items = items[:max_samples]
    cleaned: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"Sample {i} is not a JSON object.")
        prompt_text = (it.get("prompt") or it.get("question") or "").strip()
        if not prompt_text:
            raise ValueError(f"Sample {i} missing non-empty 'prompt' or 'question' field.")
        row = dict(it)
        row["prompt"] = prompt_text
        if "id" not in row:
            row["id"] = i
        cleaned.append(row)
    return cleaned


def _build_agent(args: argparse.Namespace) -> GenEvolveAgent:
    return GenEvolveAgent(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_rounds=args.max_rounds,
        max_tokens_per_round=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        n_max_reference_images=args.n_max_reference_images,
    )


def _save(results: List[Dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GenEvolve image-generation agent.")
    parser.add_argument("--prompt", default=None, help="Single user prompt.")
    parser.add_argument("--input", default=None, help="JSON/JSONL file of samples with prompt or question fields.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--model", default=os.environ.get("GENEVOLVE_MODEL", "GenEvolve"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    parser.add_argument("--max-rounds", type=int, default=11)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-prompt-length", type=int, default=6144)
    parser.add_argument("--max-response-length", type=int, default=30000)
    parser.add_argument("--n-max-reference-images", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    prompts = _load_prompts(args.prompt, args.input, args.max_samples)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    agent = _build_agent(args)
    results: List[Dict[str, Any]] = []

    def _run_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            res = agent.run(sample["prompt"])
            d = res.to_dict()
        except Exception as exc:  # noqa: BLE001
            d = {
                "prompt": sample["prompt"],
                "gen_prompt": "",
                "reference_images": [],
                "messages": [],
                "termination": "error",
                "rounds": 0,
                "error": f"{exc}",
            }
        for key, value in sample.items():
            if key not in d:
                d[key] = value
        d["id"] = sample.get("id")
        return d

    if args.parallel <= 1:
        for s in prompts:
            results.append(_run_one(s))
            _save(results, out_path)
    else:
        partial: List[Dict[str, Any] | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(_run_one, s): i for i, s in enumerate(prompts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                partial[idx] = fut.result()
                results = [item for item in partial if item is not None]
                _save(results, out_path)

    print(f"[GenEvolve] Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
