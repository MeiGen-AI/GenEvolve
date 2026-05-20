"""GenEvolve image-generation agent (OpenAI-compatible).

This is a self-contained ReAct-style agent that talks to any OpenAI-compatible
chat completion endpoint (vLLM, SGLang, OpenAI proxy, etc.) and orchestrates
three tools:

  - ``search``           : text search.
  - ``image_search``     : visual reference retrieval (returns IMG_### identifiers).
  - ``query_knowledge``  : activates one of eight callable generation skills.

The final answer is a prompt-reference program ``z = (gen_prompt, reference_images)``
that the user can feed to any reference-conditioned image generator (we ship
backends for Qwen-Image-Edit and Nano Banana Pro --- see ``genevolve.generator``).

At rollout and evaluation time, the released policy uses the same
``SYSTEM_PROMPT`` it was trained against. Privileged teacher context and
dynamic visual-experience memory belong to the RL training path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from .knowledge_tool import KnowledgeTool
from .system_prompt import FINAL_STEP_MESSAGE, SYSTEM_PROMPT
from .tools import ImageSearchTool, WebTextSearchTool


# ---------------------------------------------------------------------------
# IMG_### identifier manager
# ---------------------------------------------------------------------------


class ImageIdManager:
    """Per-trajectory IMG_### id manager.

    Image IDs are unique within a single rollout. The agent only ever refers
    to images via these IDs in <think>; in the final <answer>, ordinal
    phrases like ``the first reference image`` are used.
    """

    def __init__(self) -> None:
        self.counter = 0
        self.by_id: Dict[str, Dict[str, str]] = {}
        self._key_to_id: Dict[str, str] = {}

    def register(self, refs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for r in refs:
            key = (r.get("local_path") or r.get("url") or "").strip()
            if not key:
                continue
            if key in self._key_to_id:
                img_id = self._key_to_id[key]
            else:
                self.counter += 1
                img_id = f"IMG_{self.counter:03d}"
                self._key_to_id[key] = img_id
                self.by_id[img_id] = {
                    "img_id": img_id,
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "local_path": r.get("local_path", ""),
                    "page_url": r.get("page_url", ""),
                }
            rr = dict(r)
            rr["img_id"] = img_id
            out.append(rr)
        return out

    def lookup(self, img_id: str) -> Optional[Dict[str, str]]:
        return self.by_id.get(img_id)

    def all_image_paths(self) -> Dict[str, str]:
        return {iid: rec.get("local_path", "") for iid, rec in self.by_id.items()}


# ---------------------------------------------------------------------------
# Helpers: tool-call / answer parsing
# ---------------------------------------------------------------------------


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL)


def _parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except Exception:  # noqa: BLE001
        return None
    name = (payload.get("name") or "").strip()
    args = payload.get("arguments") or {}
    if not isinstance(args, dict):
        return None
    return name, args


def _parse_answer(text: str) -> Optional[Dict[str, Any]]:
    m = _ANSWER_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:  # noqa: BLE001
        return None


def _format_image_search_block(query: str, refs: List[Dict[str, str]]) -> str:
    lines = [f"--- image search result for [{query}] ---"]
    for r in refs:
        img_id = r.get("img_id", "")
        title = r.get("title", "image")
        url = r.get("url", "")
        local_path = r.get("local_path", "")
        page_url = r.get("page_url", "")
        lines.append(f"{img_id}: title: {title}")
        if url:
            lines.append(f"  url: {url}")
        if local_path:
            lines.append(f"  local_path: {local_path}")
        if page_url:
            lines.append(f"  page_url: {page_url}")
    lines.append("--- end of image search result ---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    role: str
    content: str


@dataclass
class GenEvolveResult:
    """Outcome of running the agent on a single user prompt."""

    prompt: str
    gen_prompt: str
    reference_images: List[Dict[str, str]]
    messages: List[Dict[str, str]] = field(default_factory=list)
    termination: str = "answer"
    rounds: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "gen_prompt": self.gen_prompt,
            "reference_images": self.reference_images,
            "messages": self.messages,
            "termination": self.termination,
            "rounds": self.rounds,
            "error": self.error,
        }


class GenEvolveAgent:
    """OpenAI-compatible inference agent for GenEvolve.

    Example
    -------
    >>> from genevolve import GenEvolveAgent
    >>> agent = GenEvolveAgent(
    ...     model="GenEvolve-8B",
    ...     base_url="http://localhost:8000/v1",
    ...     api_key="EMPTY",
    ... )
    >>> result = agent.run("Draw a cyberpunk version of the Eiffel Tower at sunset.")
    >>> print(result.gen_prompt)
    >>> print(result.reference_images)
    """

    def __init__(
        self,
        model: str = "GenEvolve-8B",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_rounds: int = 11,
        max_tokens_per_round: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_prompt_length: int = 6144,
        max_response_length: int = 30000,
        n_max_reference_images: int = 2,
        text_search_tool: Optional[WebTextSearchTool] = None,
        image_search_tool: Optional[ImageSearchTool] = None,
        knowledge_tool: Optional[KnowledgeTool] = None,
        system_prompt: Optional[str] = None,
        request_timeout: int = 600,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError(
                "The `openai` python package is required. Install with `pip install openai>=1.30`."
            )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_rounds = int(max_rounds)
        self.max_tokens_per_round = int(max_tokens_per_round)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_prompt_length = int(max_prompt_length)
        self.max_response_length = int(max_response_length)
        self.n_max_reference_images = max(1, int(n_max_reference_images))
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.request_timeout = int(request_timeout)

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.request_timeout)
        self.text_search = text_search_tool or WebTextSearchTool()
        self.image_search = image_search_tool or ImageSearchTool()
        self.knowledge = knowledge_tool or KnowledgeTool()

    # -----------------------------------------------------------------
    # Tool execution
    # -----------------------------------------------------------------
    def _exec_tool(
        self,
        name: str,
        args: Dict[str, Any],
        img_manager: ImageIdManager,
    ) -> str:
        if name == "search":
            queries = args.get("queries") or []
            if isinstance(queries, str):
                queries = [queries]
            top_k = int(args.get("top_k", 5) or 5)
            return self.text_search.call(queries=queries, top_k=top_k)

        if name == "image_search":
            query = (args.get("query") or "").strip()
            top_k = int(args.get("top_k", 5) or 5)
            results = self.image_search.call(query=query, top_k=top_k)
            with_ids = img_manager.register(results)
            return _format_image_search_block(query, with_ids)

        if name == "query_knowledge":
            skill_name = (args.get("skill_name") or "").strip()
            return self.knowledge.call(skill_name=skill_name)

        return f"Unknown tool: {name}"

    # -----------------------------------------------------------------
    # LLM call
    # -----------------------------------------------------------------
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens_per_round,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        choice = completion.choices[0]
        return (choice.message.content or "").strip()

    # -----------------------------------------------------------------
    # Public entrypoint
    # -----------------------------------------------------------------
    def run(self, user_prompt: str) -> GenEvolveResult:
        """Run a single rollout and return the final prompt-reference program."""

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        img_manager = ImageIdManager()
        rounds = 0
        error: Optional[str] = None
        termination = "max_rounds_reached"
        gen_prompt = ""
        reference_images: List[Dict[str, str]] = []

        while rounds < self.max_rounds:
            rounds += 1
            is_last = rounds == self.max_rounds
            if is_last:
                # Inject the final-step override BEFORE the assistant turn.
                messages.append({"role": "user", "content": FINAL_STEP_MESSAGE})

            try:
                response = self._call_llm(messages)
            except Exception as exc:  # noqa: BLE001
                error = f"llm_call_failed: {exc}"
                termination = "error"
                break

            messages.append({"role": "assistant", "content": response})

            answer_payload = _parse_answer(response)
            if answer_payload is not None:
                gen_prompt, reference_images = self._finalize_answer(
                    answer_payload, img_manager, self.n_max_reference_images
                )
                termination = "answer"
                break

            tool = _parse_tool_call(response)
            if tool is None:
                # No valid action this round.
                if is_last:
                    error = "no_answer_in_final_round"
                    termination = "no_answer"
                    break
                # Otherwise let the model retry with a gentle nudge.
                messages.append({
                    "role": "user",
                    "content": (
                        "<tool_response>\n"
                        "Format error: must output one <tool_call>{...}</tool_call> "
                        "or <answer>{...}</answer>.\n"
                        "</tool_response>"
                    ),
                })
                continue

            tool_name, tool_args = tool
            try:
                tool_obs = self._exec_tool(tool_name, tool_args, img_manager)
            except Exception as exc:  # noqa: BLE001
                tool_obs = f"[tool error] {tool_name}: {exc}"
            messages.append({
                "role": "user",
                "content": f"<tool_response>\n{tool_obs}\n</tool_response>",
            })

        return GenEvolveResult(
            prompt=user_prompt,
            gen_prompt=gen_prompt,
            reference_images=reference_images,
            messages=messages,
            termination=termination,
            rounds=rounds,
            error=error,
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _finalize_answer(
        answer_payload: Dict[str, Any],
        img_manager: ImageIdManager,
        n_max_reference_images: int = 2,
    ) -> Tuple[str, List[Dict[str, str]]]:
        gen_prompt = (answer_payload.get("gen_prompt") or "").strip()
        raw_refs = answer_payload.get("reference_images") or []
        if not isinstance(raw_refs, list):
            raw_refs = []

        # Resolve IMG_### -> records.
        resolved: List[Dict[str, str]] = []
        for r in raw_refs:
            if not isinstance(r, dict):
                continue
            img_id = (r.get("img_id") or "").strip()
            note = (r.get("note") or "").strip()
            rec = img_manager.lookup(img_id)
            if not rec:
                continue
            resolved.append({
                "img_id": img_id,
                "note": note,
                "local_path": rec.get("local_path", ""),
                "url": rec.get("url", ""),
                "title": rec.get("title", ""),
                "page_url": rec.get("page_url", ""),
            })

        # Sort by IMG_### so that ordinal phrases line up with the list order.
        resolved.sort(key=lambda x: x.get("img_id", ""))
        # Cap at the configured reference budget (default 2 — matches the
        # training-time convention used by the released checkpoint).
        resolved = resolved[: max(1, int(n_max_reference_images))]
        return gen_prompt, resolved
