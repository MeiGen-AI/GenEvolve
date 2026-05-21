"""Render images from saved GenEvolve prompt-reference programs.

This is the second step of the pipeline. The first step (``run_agent.py``)
produces a ``results.json`` containing one record per prompt with
``gen_prompt`` and a list of ``reference_images`` (each with a ``local_path``).
This script feeds those programs into the chosen reference-conditioned
generator and saves the rendered images.

Backends
--------
- ``qwen-image-edit``  : local diffusers debug run of ``Qwen/Qwen-Image-Edit-2511``.
- ``qwen-image-edit-service`` : POSTs to a self-hosted Qwen-Image-Edit FastAPI
                                service (``--service-url`` may be repeated).
- ``nano-banana-pro``  : Google Generative Language API
                         (``gemini-3-pro-image-preview``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genevolve.generator import (  # noqa: E402
    NanoBananaProGenerator,
    QwenImageEditGenerator,
    QwenImageEditServiceGenerator,
)


def _build_backend(args: argparse.Namespace):
    if args.backend == "qwen-image-edit":
        return QwenImageEditGenerator(
            model_id=args.qwen_model_id,
            device=args.device,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            seed=args.seed,
        )
    if args.backend == "qwen-image-edit-service":
        if not args.service_url:
            raise ValueError("--service-url is required for the qwen-image-edit-service backend.")
        return QwenImageEditServiceGenerator(
            urls=list(args.service_url),
            seed=args.seed,
        )
    if args.backend == "nano-banana-pro":
        return NanoBananaProGenerator(
            model=args.nano_model,
            api_key=args.google_api_key,
        )
    raise ValueError(f"Unknown backend: {args.backend}")


def _ref_paths(record: Dict[str, Any]) -> List[str]:
    refs = record.get("reference_images") or []
    paths: List[str] = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        p = (r.get("local_path") or "").strip()
        if p and Path(p).exists():
            paths.append(p)
    return paths


def _save(summary: List[Dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _render_one(record: Dict[str, Any], backend: Any, images_dir: Path) -> Dict[str, Any]:
    rec_id = str(record.get("id", "unknown"))
    gen_prompt = (record.get("gen_prompt") or "").strip()
    prompt = (record.get("prompt") or "").strip()
    ref_paths = _ref_paths(record)
    if gen_prompt and ref_paths:
        run_prompt, run_refs = gen_prompt, ref_paths
    else:
        run_prompt, run_refs = prompt, []
    out_image = images_dir / f"{rec_id}.png"
    info: Dict[str, Any] = dict(record)
    try:
        image = backend.generate(run_prompt, run_refs)
        image.save(out_image)
        info["image_path"] = str(out_image)
        info["output_path"] = str(out_image)
        info["success"] = True
        info["image_status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        info["image_path"] = ""
        info["output_path"] = ""
        info["success"] = False
        info["image_status"] = f"error: {exc}"
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Render images from GenEvolve agent outputs.")
    parser.add_argument("--input", required=True, help="results.json produced by run_agent.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backend",
        choices=["qwen-image-edit", "qwen-image-edit-service", "nano-banana-pro"],
        required=True,
    )
    # Qwen local
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    # Qwen service
    parser.add_argument("--service-url", action="append", default=None)
    # Nano
    parser.add_argument("--nano-model", default="gemini-3-pro-image-preview")
    parser.add_argument("--google-api-key", default=os.environ.get("GOOGLE_API_KEY"))
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Concurrent render requests. Supported for qwen-image-edit-service "
            "and nano-banana-pro; keep 1 for local qwen-image-edit."
        ),
    )
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if args.backend == "qwen-image-edit" and args.parallel != 1:
        raise ValueError(
            "--parallel is only supported for service/API backends. "
            "Use --backend qwen-image-edit-service with one or more --service-url values "
            "for multi-worker Qwen rendering."
        )
    backend = _build_backend(args)

    out_dir = Path(args.output_dir).resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "results.json"

    summary: List[Dict[str, Any]] = []
    parallel = max(1, int(args.parallel))
    if parallel == 1:
        for rec in records:
            info = _render_one(rec, backend, images_dir)
            summary.append(info)
            _save(summary, summary_path)
            print(f"[GenEvolve] {info.get('id', 'unknown')} -> {info.get('image_status')}")
    else:
        partial: List[Dict[str, Any] | None] = [None] * len(records)
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {ex.submit(_render_one, rec, backend, images_dir): i for i, rec in enumerate(records)}
            for fut in as_completed(futures):
                idx = futures[fut]
                info = fut.result()
                partial[idx] = info
                summary = [item for item in partial if item is not None]
                _save(summary, summary_path)
                print(f"[GenEvolve] {info.get('id', 'unknown')} -> {info.get('image_status')}")

    print(f"[GenEvolve] Saved {len(summary)} images to {images_dir}")


if __name__ == "__main__":
    main()
