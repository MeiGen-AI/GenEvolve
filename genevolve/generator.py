"""Reference-conditioned image generators used at the end of a GenEvolve rollout.

Two backends are provided out of the box:

  - ``QwenImageEditGenerator``  : open-source generator running
                                  ``Qwen/Qwen-Image-Edit-2511`` (the version
                                  used in the paper) via ``diffusers``. Used
                                  for the open-generator setting.
  - ``NanoBananaProGenerator``  : strong proprietary generator served by
                                  Google Generative Language API
                                  (``gemini-3-pro-image-preview``). Used for
                                  the strong-generator setting in the paper.

Both backends implement the same interface:

    image: PIL.Image = backend.generate(prompt, image_paths)

so they can be swapped freely. ``image_paths`` are the local paths returned
by the GenEvolve agent's ``reference_images`` field.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Any, List, Optional

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError(
            "Pillow is required for GenEvolve generators. Install it with `pip install pillow`."
        )


def _load_image(path: str) -> "Image.Image":
    _require_pillow()
    img = Image.open(path)
    if getattr(img, "mode", "") == "RGBA":
        img = img.convert("RGB")
    return img


def _img_to_base64(image: "Image.Image", fmt: str = "JPEG") -> str:
    _require_pillow()
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _bytes_to_image(raw: bytes) -> "Image.Image":
    _require_pillow()
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


# ===========================================================================
# Qwen-Image-Edit (open-source generator)
# ===========================================================================


class QwenImageEditGenerator:
    """Local Qwen-Image-Edit generator, powered by ``diffusers``.

    The default model id is ``Qwen/Qwen-Image-Edit-2511`` (the version used
    to produce the GenEvolve paper numbers). The model is loaded lazily on
    the first ``.generate(...)`` call.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-Edit-2511",
        device: str = "cuda",
        dtype: str = "bfloat16",
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        long_side: int = 1024,
        size_multiple: int = 16,
        seed: Optional[int] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = int(num_inference_steps)
        self.true_cfg_scale = float(true_cfg_scale)
        self.guidance_scale = float(guidance_scale)
        self.long_side = int(long_side)
        self.size_multiple = int(size_multiple)
        self.seed = seed
        self._pipe = None

    def _ensure_pipe(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import QwenImageEditPlusPipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "QwenImageEditGenerator requires `torch` and `diffusers>=0.38`. "
                "Install the Qwen renderer extra with `pip install -e \".[qwen]\"`."
            ) from exc
        torch_dtype = getattr(torch, self.dtype)
        self._pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.model_id, torch_dtype=torch_dtype
        ).to(self.device)

    def _output_size(self, image_paths: List[str]) -> tuple[int, int]:
        if not image_paths:
            return self.long_side, self.long_side
        try:
            with _load_image(image_paths[0]) as img:
                rw, rh = img.size
        except Exception:  # noqa: BLE001
            return self.long_side, self.long_side
        scale = self.long_side / max(rw, rh)
        return (
            _round_to_multiple(rw * scale, self.size_multiple),
            _round_to_multiple(rh * scale, self.size_multiple),
        )

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None) -> "Image.Image":
        _require_pillow()
        self._ensure_pipe()
        image_paths = list(image_paths or [])
        if not image_paths:
            raise ValueError("QwenImageEditGenerator requires at least one reference image path.")
        width, height = self._output_size(image_paths)
        ref_images = [_load_image(p) for p in image_paths]
        kwargs: dict = {
            "image": ref_images,
            "prompt": prompt,
            "num_inference_steps": self.num_inference_steps,
            "true_cfg_scale": self.true_cfg_scale,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": " ",
            "width": int(width),
            "height": int(height),
        }
        if self.seed is not None:
            try:
                import torch
                generator = torch.Generator(device=self.device).manual_seed(int(self.seed))
                kwargs["generator"] = generator
            except Exception:  # pragma: no cover
                pass
        out = self._pipe(**kwargs)  # type: ignore[union-attr]
        return out.images[0]


class QwenImageEditServiceGenerator:
    """HTTP client for a self-hosted Qwen-Image-Edit FastAPI service.

    Useful when you want to keep the heavy diffusion model on a dedicated GPU
    box and run the GenEvolve agent on a separate machine. The service must
    accept ``POST {base_url}/generate`` with ``image_urls`` (a list of base64
    data URIs) and return ``{"success": true, "image": "<base64>"}``.
    """

    def __init__(
        self,
        urls: List[str],
        path: str = "/generate",
        timeout: int = 1800,
        max_retries: int = 3,
        long_side: int = 1024,
        size_multiple: int = 16,
        max_refs: int = 4,
        seed: int = 0,
    ) -> None:
        if requests is None:
            raise RuntimeError("The `requests` package is required. `pip install requests`.")
        if not urls:
            raise ValueError("QwenImageEditServiceGenerator: `urls` must not be empty.")
        self.urls = [u.rstrip("/") for u in urls]
        self.path = path if path.startswith("/") else "/" + path
        self.timeout = int(timeout)
        self.max_retries = max(1, int(max_retries))
        self.long_side = int(long_side)
        self.size_multiple = int(size_multiple)
        self.max_refs = int(max_refs)
        self.seed = int(seed)
        self._cursor = 0

    def _next_url(self) -> str:
        url = self.urls[self._cursor % len(self.urls)] + self.path
        self._cursor += 1
        return url

    def _refs(self, image_paths: List[str]) -> List[str]:
        out: List[str] = []
        for p in image_paths[: self.max_refs]:
            if not p or not Path(p).exists():
                continue
            with open(p, "rb") as fh:
                out.append(f"data:image/jpeg;base64,{base64.b64encode(fh.read()).decode('utf-8')}")
        return out

    def _output_size(self, image_paths: List[str]) -> tuple[int, int]:
        if not image_paths or Image is None:
            return self.long_side, self.long_side
        try:
            with _load_image(image_paths[0]) as img:
                rw, rh = img.size
        except Exception:  # noqa: BLE001
            return self.long_side, self.long_side
        scale = self.long_side / max(rw, rh)
        return (
            _round_to_multiple(rw * scale, self.size_multiple),
            _round_to_multiple(rh * scale, self.size_multiple),
        )

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None) -> "Image.Image":
        image_paths = list(image_paths or [])
        ref_urls = self._refs(image_paths)
        if not ref_urls:
            raise ValueError("QwenImageEditServiceGenerator requires at least one valid reference image.")
        width, height = self._output_size(image_paths)
        payload = {
            "image_urls": ref_urls,
            "prompt": prompt,
            "seed": self.seed,
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
            "width": int(width),
            "height": int(height),
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            url = self._next_url()
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                if data.get("success") is not True:
                    raise RuntimeError(f"service error: {str(data)[:300]}")
                b64 = data.get("image") or data.get("image_base64")
                if not b64:
                    raise RuntimeError(f"empty image payload: {str(data)[:200]}")
                return _bytes_to_image(base64.b64decode(b64))
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(10, 1 + 2 * attempt))
        raise RuntimeError(f"Qwen-Image-Edit service failed after {self.max_retries} attempts: {last_err}")


# ===========================================================================
# Nano Banana Pro (Gemini 3 Pro Image)
# ===========================================================================


class NanoBananaProGenerator:
    """Reference-conditioned generation via the Google Generative Language API.

    Set ``GOOGLE_API_KEY`` (or pass ``api_key=...``). The default endpoint
    matches the public Google REST API:

        https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent

    The default model name is ``gemini-3-pro-image-preview`` (a.k.a. Nano
    Banana Pro).
    """

    DEFAULT_ENDPOINT_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3-pro-image-preview",
        endpoint_template: Optional[str] = None,
        timeout: int = 600,
        max_retries: int = 5,
    ) -> None:
        if requests is None:
            raise RuntimeError("The `requests` package is required. `pip install requests`.")
        self.api_key = (api_key or os.environ.get("GOOGLE_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Set it in the environment or pass api_key=... to "
                "NanoBananaProGenerator."
            )
        self.model = model
        self.endpoint = (endpoint_template or self.DEFAULT_ENDPOINT_TEMPLATE).format(model=model)
        self.timeout = int(timeout)
        self.max_retries = max(1, int(max_retries))

    @staticmethod
    def _extract_inline_image(data: Any) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        for cand in data.get("candidates") or []:
            content = (cand or {}).get("content") or {}
            for part in content.get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline, dict):
                    b64 = inline.get("data")
                    if b64:
                        return str(b64)
        return None

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None) -> "Image.Image":
        _require_pillow()
        image_paths = list(image_paths or [])
        parts: List[dict] = [{"text": prompt}]
        for p in image_paths:
            if not p or not Path(p).exists():
                continue
            img = _load_image(p)
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": _img_to_base64(img),
                }
            })
        payload = json.dumps({"contents": [{"parts": parts}]}, ensure_ascii=False)
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.endpoint,
                    data=payload.encode("utf-8"),
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(f"API error: {data.get('error')}")
                b64 = self._extract_inline_image(data)
                if not b64:
                    raise RuntimeError(f"empty inline image: {str(data)[:300]}")
                return _bytes_to_image(base64.b64decode(b64))
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(20, 2 + 2 * attempt))
        raise RuntimeError(f"Nano Banana Pro failed after {self.max_retries} attempts: {last_err}")


__all__ = [
    "QwenImageEditGenerator",
    "QwenImageEditServiceGenerator",
    "NanoBananaProGenerator",
]
