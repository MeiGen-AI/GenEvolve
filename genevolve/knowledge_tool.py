"""GenEvolve KnowledgeTool: exposes the eight callable generation skills.

The agent calls ``query_knowledge(skill_name=<one of 8>)`` and gets back the
corresponding skill instructions to weave into the final prompt-reference
program. Skill text files are stored as plain Markdown under ``knowledge/skills/``.

During RL training, the full training tree augments these static skills with
dynamic visual experience memory. This standalone package exposes the same
skill interface needed for rollout, evaluation, and model release usage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SKILL_NAMES: List[str] = [
    "spatial_layout",
    "aesthetic_drawing",
    "text_rendering",
    "creative_drawing",
    "anatomy_body_coherence",
    "attribute_binding",
    "physical_material_consistency",
    "quantity_counting",
]


def _default_skills_dir() -> Path:
    """Return the directory holding the 8 skill markdown files."""
    return Path(__file__).resolve().parent / "knowledge" / "skills"


class SkillBank:
    """Static skill bank: loads markdown instructions for the 8 skills."""

    def __init__(self, skills_dir: Optional[str] = None) -> None:
        self.skills_dir = Path(skills_dir).resolve() if skills_dir else _default_skills_dir()
        self.skills: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.skills_dir.exists():
            raise FileNotFoundError(
                f"Skills directory not found: {self.skills_dir}. "
                "Make sure the package was installed correctly."
            )
        for name in SKILL_NAMES:
            md_path = self.skills_dir / f"{name}.md"
            if md_path.exists():
                self.skills[name] = {
                    "name": name,
                    "instructions": md_path.read_text(encoding="utf-8"),
                }

    def get(self, name: str) -> Optional[str]:
        entry = self.skills.get(name)
        if not entry:
            return None
        return entry.get("instructions")

    def available(self) -> List[str]:
        return sorted(self.skills.keys())


class KnowledgeTool:
    """``query_knowledge`` tool implementation, callable by the agent.

    Usage:
        tool = KnowledgeTool()
        out = tool.call(skill_name="spatial_layout")
    """

    TOOL_DEFINITION = {
        "type": "function",
        "function": {
            "name": "query_knowledge",
            "description": (
                "Get expert prompt-writing guidance for a specific generation "
                "skill. Specify which skill via skill_name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "enum": SKILL_NAMES,
                        "description": "Which skill to query.",
                    }
                },
                "required": ["skill_name"],
            },
        },
    }

    def __init__(self, skills_dir: Optional[str] = None) -> None:
        self.skill_bank = SkillBank(skills_dir=skills_dir)

    def call(self, skill_name: str = "", **_: object) -> str:
        if not skill_name or skill_name not in SKILL_NAMES:
            return (
                f"Unknown skill '{skill_name}'. Available skills: "
                f"{', '.join(SKILL_NAMES)}"
            )
        instructions = self.skill_bank.get(skill_name)
        if not instructions:
            return (
                f"No skill content available for '{skill_name}'. "
                f"Expected file: {self.skill_bank.skills_dir / (skill_name + '.md')}"
            )
        return f"## Skill Guidance\n\n=== Skill: {skill_name} ===\n{instructions}"
