from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="genevolve",
    version="0.1.0",
    description=(
        "GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated "
        "Visual Experience Distillation (inference release)."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://ephemeral182.github.io/GenEvolve/",
    license="Apache-2.0",
    packages=find_packages(exclude=("scripts", "examples", "assets")),
    include_package_data=True,
    package_data={"genevolve": ["knowledge/skills/*.md"]},
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.30",
        "requests>=2.28",
        "pillow>=10.0",
    ],
    extras_require={
        "qwen": [
            "torch>=2.4",
            "diffusers>=0.32",
            "transformers>=4.45",
            "accelerate>=0.30",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
