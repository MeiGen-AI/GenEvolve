"""GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation.

Inference-only release. Provides the trained image-generation agent
(prompt-reference program synthesis) plus reference-conditioned generation
backends (Qwen-Image-Edit, Nano Banana Pro).
"""

from .agent import GenEvolveAgent
from .knowledge_tool import KnowledgeTool, SkillBank

__version__ = "0.1.0"
__all__ = ["GenEvolveAgent", "KnowledgeTool", "SkillBank"]
