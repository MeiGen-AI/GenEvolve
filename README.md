<div align="center">

<img src="assets/logo_genevolve.png" alt="GenEvolve" width="160">

<h1>GenEvolve</h1>

<p><strong><em>Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation</em></strong></p>

<p>
  <a href="https://ephemeral182.github.io/GenEvolve/">
    <img alt="Project Page" src="https://img.shields.io/badge/🌐_Project-Page-1f6feb"></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img alt="arXiv" src="https://img.shields.io/badge/📄_arXiv-XXXX.XXXXX-b31b1b"></a>
  <a href="https://huggingface.co/Ephemeral182/GenEvolve-8B">
    <img alt="Weights" src="https://img.shields.io/badge/🤗_HuggingFace-GenEvolve--8B-FFD21E"></a>
  <a href="https://github.com/Ephemeral182/GenEvolve">
    <img alt="GitHub" src="https://img.shields.io/badge/💾_GitHub-Code-181717"></a>
</p>

<p>
  <img alt="python" src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="pytorch" src="https://img.shields.io/badge/pytorch-2.8-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="vllm" src="https://img.shields.io/badge/vLLM-0.11-30A14E">
  <img alt="cuda" src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white">
  <img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-green">
  <img alt="status" src="https://img.shields.io/badge/status-active-brightgreen">
</p>

</div>

## 👥 Authors

> [**Sixiang Chen**](https://ephemeral182.github.io/)<sup>1</sup>, [**Zhaohu Xing**](https://ge-xing.github.io/)<sup>1</sup>, [**Tian Ye**](https://owen718.github.io/)<sup>1</sup>, [**Xinyu Geng**](https://scholar.google.com/citations?user=rYB-IBwAAAAJ&hl=zh-CN)<sup>2</sup>, [**Yunlong Lin**](https://lyl1015.github.io/)<sup>3</sup>, [**Jianyu Lai**](https://alexlai2860.github.io/)<sup>1</sup>, [**Xuanhua He**](https://xuanhuahe.github.io/)<sup>2</sup>, [**Fuxiang Zhai**](https://fuxiangzhai.github.io/)<sup>1</sup>, [**Jialin Gao**](https://scholar.google.com/citations?user=sj4FqEgAAAAJ&hl=zh-CN)<sup>4</sup>, [**Lei Zhu**](https://sites.google.com/site/indexlzhu/home)<sup>1,2</sup>†
>
> <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)
>
> <sup>2</sup>The Hong Kong University of Science and Technology
>
> <sup>3</sup>The Chinese University of Hong Kong
>
> <sup>4</sup>National University of Singapore
>
> †Corresponding Author

---

<div align="center">
<img src="assets/teaser.jpg" alt="GenEvolve teaser" width="100%">

<p><em>The same trained agent policy paired with two reference-conditioned generators ⟶<br>
<strong>Qwen-Image-Edit (open)</strong> &nbsp;·&nbsp; <strong>Nano Banana Pro (strong)</strong></em></p>
</div>

## 🌟 What is GenEvolve?

GenEvolve formulates open-ended image generation as a **tool-orchestrated visual trajectory**. The agent gathers external textual evidence, retrieves visual references, performs **internal knowledge activation** through callable generation skills, and synthesizes a **prompt-reference program** $z = (g, R)$ that any reference-conditioned generator can render.

The released `GenEvolve-8B` policy is based on Qwen3-VL-8B and is designed to be **generator-transferable**: the same agent output can be rendered by the open Qwen-Image-Edit backend or by a stronger proprietary renderer such as Nano Banana Pro.

## 🎁 What's released

| Component | Where |
|---|---|
| 🧠 Trained agent policy `GenEvolve-8B` (Qwen3-VL-8B-based) | 🤗 [`Ephemeral182/GenEvolve-8B`](https://huggingface.co/Ephemeral182/GenEvolve-8B) |
| ⚡ Standalone inference runtime (`GenEvolveAgent`, OpenAI-compatible) | this repo |
| 🛠️ Three tools (`search`, `image_search`, `query_knowledge`) | this repo |
| 📚 The eight skill markdown files used at training time | this repo |
| 🎨 Reference-conditioned generator wrappers (Qwen-Image-Edit + Nano Banana Pro) | this repo |
| 📦 SFT trajectories (9,000 records) | 🤗 [`Ephemeral182/GenEvolve-Data-SFT`](https://huggingface.co/datasets/Ephemeral182/GenEvolve-Data-SFT) |
| 🎯 Self-evolution prompts + GT images (3,175 records) | 🤗 [`Ephemeral182/GenEvolve-Data-RL`](https://huggingface.co/datasets/Ephemeral182/GenEvolve-Data-RL) |
| 📊 Held-out evaluation benchmark (594 prompts + GT images) | 🤗 [`Ephemeral182/GenEvolve-Bench`](https://huggingface.co/datasets/Ephemeral182/GenEvolve-Bench) |

## 📋 Requirements

The most important setup detail is that GenEvolve intentionally separates the **agent / LLM-server stack** from the optional **diffusion renderer stack**. The `requirements.txt` file is lightweight on purpose: it installs the Python client runtime and tools, while heavyweight GPU packages such as `vllm`, `sglang`, `flash-attn`, and `diffusers` are installed explicitly for the environment that needs them.

### Environment A - `genevolve`: agent runtime + LLM server

Use this environment to run `GenEvolveAgent` and to serve the released `GenEvolve-8B` checkpoint through an OpenAI-compatible API. This is the only required environment for agent rollouts.

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 tested | package metadata supports Python >= 3.10 |
| `openai`, `requests`, `pillow` | see `requirements.txt` | lightweight agent runtime |
| `torch` | 2.8.0 + CUDA 12.x | Qwen3-VL inference |
| `transformers` | >= 4.57 | Qwen3-VL support |
| `flash-attn` | 2.8.3 | fused attention used by Qwen3-VL |
| `vllm` | >= 0.11 | recommended OpenAI-compatible server |
| `sglang` | >= 0.5.4 | alternative OpenAI-compatible server |

```bash
conda create -n genevolve python=3.11 -y
conda activate genevolve
pip install -e .

# Pick one server stack for GenEvolve-8B.
pip install "vllm>=0.11"
# or
pip install "sglang[all]>=0.5.4"

# Recommended for Qwen3-VL.
pip install flash-attn==2.8.3 --no-build-isolation
```

GPU: a single CUDA GPU with at least 24 GB VRAM can serve `GenEvolve-8B` at `tp=1`; for large batch evaluation we recommend 40 GB+ VRAM or tensor parallelism.

### Environment B - `genevolve-qwen`: local Qwen-Image-Edit renderer

Use this environment only when rendering with `--backend qwen-image-edit`. Skip it if you render with Nano Banana Pro or a remote Qwen-Image-Edit service.

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 tested | |
| `torch` | >= 2.6, < 2.7 + CUDA 12.x | stable diffusers stack for Qwen Image Edit |
| `diffusers` | >= 0.38 | must include `QwenImageEditPlusPipeline` |
| `transformers` | >= 4.55 | |
| `accelerate` | >= 1.0 | |

```bash
conda create -n genevolve-qwen python=3.11 -y
conda activate genevolve-qwen
pip install "torch>=2.6,<2.7" --index-url https://download.pytorch.org/whl/cu124
pip install "diffusers>=0.38" "transformers>=4.55" "accelerate>=1.0"
pip install -e ".[qwen]"
```

GPU: at least 24 GB VRAM. In practice, it is cleaner to put the diffusion renderer on a separate GPU or a separate machine from the LLM server.

### External services

| Service | Variable | Used for |
|---|---|---|
| [serper.dev](https://serper.dev) | `SERPER_API_KEY` | required for `search` and `image_search` |
| [Google Generative Language API](https://ai.google.dev/api) | `GOOGLE_API_KEY` | only for `--backend nano-banana-pro` |

### Training environments used for our runs

These are **not required** to run this inference release, but they are useful if you want to reuse the released SFT/RL data for your own training.

| Stage | Stack used in our logs |
|---|---|
| SFT | Python 3.11 conda env `llamafactory`; LLaMA-Factory `0.9.4.dev0`; `torch==2.9.1`; `transformers==4.57.0`; `deepspeed==0.18.9`; `flash_attn==2.8.3`; `accelerate==1.11.0`; full-parameter SFT, bf16, FlashAttention-2, DeepSpeed ZeRO-3. |
| RL | Python 3.11 conda env `skillweaver_rl`; `rllm==0.2.1`; `verl==0.6.1`; `torch==2.8.0`; `transformers==4.57.1`; `sglang==0.5.4.post2`; `vllm==0.11.0`; `ray==2.54.1`; `flash_attn==2.8.3`; GRPO with FSDP and async SGLang rollout. |

## 🚀 Quickstart

### 1. Install the agent runtime

```bash
git clone https://github.com/Ephemeral182/GenEvolve.git
cd GenEvolve

conda create -n genevolve python=3.11 -y
conda activate genevolve
pip install -e .
```

Install either `vllm` or `sglang` in this same environment if you plan to host `GenEvolve-8B` locally. Install Environment B only when you need the local Qwen diffusion renderer.

### 2. Serve the released checkpoint

```bash
# vLLM (recommended)
MODEL_PATH=/path/to/GenEvolve-8B PORT=8000 TP=1 bash scripts/serve_vllm.sh

# or SGLang
MODEL_PATH=/path/to/GenEvolve-8B PORT=8000 TP=1 bash scripts/serve_sglang.sh
```

### 3. Run an end-to-end example

```bash
export SERPER_API_KEY=<your_key>             # required for search and image_search
export GOOGLE_API_KEY=<your_key>             # only for the Nano Banana Pro backend

python examples/quickstart.py \
    --backend nano-banana-pro \
    --base-url http://localhost:8000/v1 \
    --model GenEvolve-8B \
    --prompt "A 1990s travel-magazine cover of two backpackers in front of the Eiffel Tower at golden hour, the title \"PARIS\" rendered in bold serif type." \
    --output paris.png
```

For the open-generator path, use `--backend qwen-image-edit` after installing Environment B.

### 4. Batch pipeline

The agent rollout and the heavy image rendering are split into two stages so they can run on different machines.

```bash
# Stage 1: agent rollouts -> results.json.
python scripts/run_agent.py \
    --input examples/example_prompts.jsonl \
    --output-dir runs/example \
    --base-url http://localhost:8000/v1 \
    --model GenEvolve-8B \
    --parallel 4

# Stage 2a: render locally with Qwen-Image-Edit-2511.
python scripts/generate_images.py \
    --input runs/example/results.json \
    --output-dir runs/example_qwen \
    --backend qwen-image-edit

# Stage 2b: render through one or more remote Qwen-Image-Edit services.
python scripts/generate_images.py \
    --input runs/example/results.json \
    --output-dir runs/example_qwen_service \
    --backend qwen-image-edit-service \
    --service-url http://your-qwen-service:8001

# Stage 2c: render with Nano Banana Pro.
python scripts/generate_images.py \
    --input runs/example/results.json \
    --output-dir runs/example_nano \
    --backend nano-banana-pro
```

## 🧩 Programmatic API

```python
from genevolve import GenEvolveAgent
from genevolve.generator import QwenImageEditGenerator  # or NanoBananaProGenerator

agent = GenEvolveAgent(
    model="GenEvolve-8B",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)
result = agent.run("A cyberpunk version of the Sydney Opera House at sunset.")

# z = (gen_prompt, reference_images)
print(result.gen_prompt)
for r in result.reference_images:
    print(r["img_id"], r["local_path"], r["note"])

backend = QwenImageEditGenerator(model_id="Qwen/Qwen-Image-Edit-2511")
image = backend.generate(
    result.gen_prompt,
    [r["local_path"] for r in result.reference_images if r.get("local_path")],
)
image.save("opera.png")
```

## 🧠 Method overview

<p align="center"><img src="assets/overview.png" alt="GenEvolve method overview" width="92%"></p>

For a user request $x$, the agent samples a multi-turn trajectory

$$\tau = (a_1, o_1, \ldots, a_T, o_T, z), \qquad z = (g, R),$$

where each $a_t$ is one of the three actions below and $o_t$ is the corresponding observation. The downstream generator renders $\hat{y} = G(g, R)$.

<table>
  <thead>
    <tr><th>Tool</th><th>Role</th><th>Output</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><code>search(queries)</code></td>
      <td>External textual evidence - entities, dates, facts.</td>
      <td>Markdown digest.</td>
    </tr>
    <tr>
      <td><code>image_search(query)</code></td>
      <td>Visual references; each result gets a unique <code>IMG_###</code> id.</td>
      <td>Image list with local paths.</td>
    </tr>
    <tr>
      <td><code>query_knowledge(skill_name)</code></td>
      <td><strong>Internal knowledge activation</strong> - invokes one of the eight callable generation skills.</td>
      <td>Skill instructions in Markdown.</td>
    </tr>
  </tbody>
</table>

The final answer is a JSON object, the **prompt-reference program**:

```json
{
  "gen_prompt": "... a targeted instruction that refers to references by ordinal phrases ('the first reference image', 'the second reference image') ...",
  "reference_images": [
    {"img_id": "IMG_001", "note": "what to copy from this reference"}
  ]
}
```

## 📦 Data

We release three datasets on the Hugging Face Hub. The total trajectory data is too large for GitHub but installs in one line via 🤗 `datasets` / `huggingface-cli`.

| Dataset | Records | Size | Purpose |
|---|---|---|---|
| [`GenEvolve-Data-SFT`](https://huggingface.co/datasets/Ephemeral182/GenEvolve-Data-SFT) | 9,000 records | ~7.4 GB | Multi-turn tool-orchestrated trajectories used for the SFT cold start. Each record: `messages` (chat-format ReAct trajectory ending in `<answer>{gen_prompt, reference_images}`) + `images` (reference jpegs). |
| [`GenEvolve-Data-RL`](https://huggingface.co/datasets/Ephemeral182/GenEvolve-Data-RL) | 3,175 records | ~680 MB | Open-ended user requests paired with curated GT images. Used for GRPO + Visual Experience Distillation, where multiple agent rollouts per prompt are scored against the GT. |
| [`GenEvolve-Bench`](https://huggingface.co/datasets/Ephemeral182/GenEvolve-Bench) | 594 prompts | ~120 MB | Held-out evaluation benchmark. Contains both **Knowledge-Anchored** (T1, 335) and **Quality-Anchored** (T3, 259) tracks plus per-prompt category, difficulty, and skill metadata. |

### Quick load

```bash
pip install -U huggingface_hub datasets

huggingface-cli download Ephemeral182/GenEvolve-Bench \
    --repo-type dataset \
    --local-dir ./GenEvolve-Bench
```

```python
from datasets import load_dataset

bench = load_dataset("Ephemeral182/GenEvolve-Bench", split="test")
print(bench[0]["question"], bench[0]["gt_image"])

rl = load_dataset("Ephemeral182/GenEvolve-Data-RL", split="train")
sft = load_dataset("Ephemeral182/GenEvolve-Data-SFT", split="train")
print(sft[0]["messages"])
print(sft[0]["images"])
```

All paths inside the datasets are relative, for example `images/case_00512.jpg` or `images/traj_00213/IMG_001.jpg`; resolve them against the dataset directory you downloaded to. Per-dataset usage notes live on each dataset's Hub page.

Although GenEvolve's training pipeline is not part of this repository, the released SFT and RL datasets together with the inference runtime here let you reproduce the path from a user request to a rendered image.

## 🖼️ Visual results

<p align="center"><img src="assets/visual_comparison.png" alt="Qualitative comparison" width="100%"></p>

<p align="center"><sub>The same <code>GenEvolve-8B</code> policy paired with two different reference-conditioned generators. <span style="color:#D97706">Orange</span> marks external/uncommon knowledge, <span style="color:#2563EB">blue</span> marks internal generation-knowledge requirements.</sub></p>

### 🎨 Extended gallery - paired with Nano Banana Pro

<p align="center"><img src="assets/gallery_nano.jpg" alt="GenEvolve + Nano Banana Pro gallery" width="100%"></p>

<p align="center"><sub>Additional qualitative results of <code>GenEvolve-8B</code> with Nano Banana Pro as the downstream renderer. The agent autonomously orchestrates search, reference selection, and skill activation across diverse open-ended categories: spatial layout, text rendering, quantity counting, attribute binding, anatomy/pose, creative transfer, material physics, and aesthetic drawing.</sub></p>

### 🎨 Extended gallery - paired with Qwen-Image-Edit (open)

<p align="center"><img src="assets/gallery_qwen.jpg" alt="GenEvolve + Qwen-Image-Edit gallery" width="100%"></p>

<p align="center"><sub>The same trained agent policy paired with the open-source Qwen-Image-Edit-2511 renderer. Consistent quality across both generators demonstrates that <code>GenEvolve-8B</code> learns generator-transferable tool orchestration rather than overfitting to one specific renderer.</sub></p>

## ⚙️ Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_BASE_URL` | OpenAI-compatible chat-completions endpoint | `http://localhost:8000/v1` |
| `OPENAI_API_KEY` | API key for the inference server | `EMPTY` |
| `SERPER_API_KEY` | [serper.dev](https://serper.dev) key for text and image search | required |
| `SERPER_BASE_URL` | Override for Serper-compatible gateways | `https://google.serper.dev` |
| `IMAGE_DOWNLOAD_DIR` | Local cache for `image_search` downloads | `/tmp/genevolve_images` |
| `GOOGLE_API_KEY` | Google Generative Language API key | required for Nano backend |

`GenEvolveAgent` constructor knobs:

| Argument | Default | Note |
|---|---|---|
| `max_rounds` | `11` | Max ReAct turns; the last turn is forced to emit `<answer>`. |
| `max_tokens_per_round` | `4096` | Per-turn max new tokens. |
| `temperature` | `0.6` | Sampling temperature for the agent policy. |
| `top_p` | `0.9` | Nucleus sampling for the agent policy. |
| `max_prompt_length` | `6144` | Training/eval compatibility setting; the serving backend enforces the actual context limit. |
| `max_response_length` | `30000` | Training/eval compatibility setting for long multi-turn trajectories. |
| `n_max_reference_images` | `2` | Max reference images forwarded to the downstream generator. |

## 🧯 Troubleshooting

| Symptom | Check |
|---|---|
| `search` / `image_search` returns authentication errors | Set `SERPER_API_KEY` or configure `SERPER_BASE_URL` for your internal Serper-compatible gateway. |
| Agent cannot connect to the model | Confirm the vLLM/SGLang server is running and `OPENAI_BASE_URL` or `--base-url` ends with `/v1`. |
| Qwen local renderer fails at import time | Install Environment B and make sure `diffusers>=0.38` is active. |
| Qwen renderer says it needs a reference image | Qwen-Image-Edit is reference-conditioned; rerun the agent or use Nano Banana Pro for no-reference prompts. |
| `flash-attn` build fails | Install a PyTorch/CUDA wheel first, then run `pip install flash-attn==2.8.3 --no-build-isolation`. |
| Batch rendering resumes after interruption | `scripts/generate_images.py` writes `results.json` incrementally under the output directory. |

## 🗂️ Repository layout

```
genevolve/
├── genevolve/
│   ├── agent.py               # GenEvolveAgent: ReAct loop on top of an OpenAI-compatible server
│   ├── system_prompt.py       # system prompt used by the released agent
│   ├── knowledge_tool.py      # query_knowledge: eight callable generation skills
│   ├── tools/web_search.py    # search + image_search (Serper-compatible)
│   ├── generator.py           # Qwen-Image-Edit + Nano Banana Pro backends
│   └── knowledge/skills/      # skill markdown files
├── scripts/
│   ├── serve_vllm.sh          # serve the checkpoint with vLLM
│   ├── serve_sglang.sh        # serve the checkpoint with SGLang
│   ├── run_agent.py           # batch agent rollouts -> results.json
│   └── generate_images.py     # render images from results.json
├── examples/
│   ├── quickstart.py          # single-prompt end-to-end example
│   └── example_prompts.jsonl
├── assets/                    # README figures
├── requirements.txt
├── setup.py
└── README.md
```

## 🙏 Acknowledgements

GenEvolve builds directly on **[Gen-Searcher](https://github.com/RUCAIBox/Gen-Searcher)** and inherits its three-tool ReAct interface and dual image/text reward design. We thank the Gen-Searcher authors for making their work publicly available.

## 📝 Citation

```bibtex
@inproceedings{chen2026genevolve,
  title     = {GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation},
  author    = {Chen, Sixiang and Xing, Zhaohu and Ye, Tian and Geng, Xinyu and Lin, Yunlong and Lai, Jianyu and He, Xuanhua and Zhai, Fuxiang and Gao, Jialin and Zhu, Lei},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## 📜 License

Code is released under the [Apache 2.0](LICENSE) license. Released model weights inherit the upstream license of `Qwen3-VL-8B-Instruct`. Search results returned by Serper.dev and images rendered by Nano Banana Pro / Qwen-Image-Edit are governed by the respective upstream service terms.
