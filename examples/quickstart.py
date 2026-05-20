"""Minimal end-to-end example for GenEvolve inference.

1. Start an OpenAI-compatible inference server with the GenEvolve checkpoint
   (see ``scripts/serve_vllm.sh`` or ``scripts/serve_sglang.sh``).
2. ``export SERPER_API_KEY=...`` so the agent can call text/image search.
3. Run this script. By default it uses Nano Banana Pro for the final
   reference-conditioned image; pass ``--backend qwen-image-edit`` to use the
   open-source Qwen-Image-Edit generator instead.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from genevolve import GenEvolveAgent
from genevolve.generator import NanoBananaProGenerator, QwenImageEditGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="A 1990s travel-magazine cover photo of two backpackers in front of "
        "the Eiffel Tower at golden hour, the title \"PARIS\" rendered in bold "
        "serif type at the top.",
    )
    parser.add_argument("--backend", choices=["nano-banana-pro", "qwen-image-edit"], default="nano-banana-pro")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="GenEvolve-8B")
    parser.add_argument("--output", default="quickstart_output.png")
    args = parser.parse_args()

    agent = GenEvolveAgent(model=args.model, base_url=args.base_url, api_key=args.api_key)
    result = agent.run(args.prompt)

    print("== Trajectory finished ==")
    print(f"termination: {result.termination}")
    print(f"rounds: {result.rounds}")
    print("\n[gen_prompt]")
    print(result.gen_prompt or "(empty)")
    print("\n[reference_images]")
    for r in result.reference_images:
        print(f"  {r.get('img_id')}: {r.get('local_path')}  -- {r.get('note')}")

    if not result.gen_prompt or not result.reference_images:
        print("\n[!] Agent did not produce a usable program; skipping image generation.")
        return

    if args.backend == "qwen-image-edit":
        backend = QwenImageEditGenerator()
    else:
        backend = NanoBananaProGenerator()

    refs = [r["local_path"] for r in result.reference_images if r.get("local_path")]
    image = backend.generate(result.gen_prompt, refs)
    image.save(args.output)
    print(f"\nImage saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
