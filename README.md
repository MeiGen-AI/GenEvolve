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

GenEvolve follows the same environment split as Gen-Searcher: one LLaMA-Factory environment for SFT, and one RL/evaluation/inference environment for verl/rllm, SGLang/vLLM rollout, tool execution, and image generation. The public environment names below use `genevolve`; older internal log paths may contain historical names, but they are not part of the release.

### Full RL / evaluation / inference environment - `genevolve`

Use this environment for agent rollouts, held-out evaluation, OpenAI-compatible serving, SGLang/vLLM rollout, and Qwen/Nano final rendering wrappers.

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 |
| CUDA stack | CUDA 12.x; our logs used PyTorch CUDA 12.8 wheels |
| `torch` / `torchvision` | `2.8.0` / `0.23.0` |
| `transformers` | `4.57.1` |
| `vllm` | `0.11.0` |
| `sglang` | `0.5.4.post2` |
| `verl` / `rllm` | `0.6.1` / `0.2.1` |
| `ray` | `2.54.1` |
| `flash-attn` | `2.8.3` |
| `diffusers` | `>=0.38` for `QwenImageEditPlusPipeline` |

```bash
conda create -n genevolve python=3.11 -y
conda activate genevolve

# Install PyTorch first so CUDA extensions such as flash-attn build against
# the correct torch/CUDA pair. Adjust the CUDA wheel index if your cluster
# standardizes on another CUDA minor version.
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# Install the full GenEvolve experiment stack.
pip install --no-build-isolation -r requirements.txt
pip install -e .
```

This is the normal full environment for the released code path: agent rollout, tool calls, image rendering, benchmark inference, and the RL stack used in our full training tree.

### SFT environment - `genevolve-sft`

SFT follows [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), as in Gen-Searcher. Our SFT run used full-parameter training on Qwen3-VL-8B with bf16, FlashAttention-2, and DeepSpeed ZeRO-3.

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 |
| LLaMA-Factory | `0.9.4.dev0` in our run |
| `torch` / `torchvision` | `2.9.1` / `0.24.1` |
| `transformers` | `4.57.0` |
| `datasets` | `4.0.0` |
| `accelerate` | `1.11.0` |
| `deepspeed` | `0.18.9` |
| `flash-attn` | `2.8.3` |

```bash
conda create -n genevolve-sft python=3.11 -y
conda activate genevolve-sft

# In the full training tree, install LLaMA-Factory as the SFT trainer.
cd Gen-DeepResearch-SFT/LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install deepspeed==0.18.9 flash-attn==2.8.3 --no-build-isolation
```

The released repository provides the model, inference runtime, tools, skills, and datasets. The internal full training tree uses the same SFT/RL environment structure above; if you plug the released SFT/RL data into your own LLaMA-Factory + verl/rllm tree, use these versions as the reference.

### External services

| Service | Variable | Used for |
|---|---|---|
| [serper.dev](https://serper.dev) | `SERPER_API_KEY` | required for `search` and `image_search` |
| [Google Generative Language API](https://ai.google.dev/api) | `GOOGLE_API_KEY` | only for `--backend nano-banana-pro` |

## 🚀 Quickstart

### 1. Install the agent runtime

```bash
git clone https://github.com/Ephemeral182/GenEvolve.git
cd GenEvolve

conda create -n genevolve python=3.11 -y
conda activate genevolve
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation -r requirements.txt
pip install -e .
```

This installs the full GenEvolve runtime stack, including vLLM/SGLang serving, the agent tools, and Qwen/Nano rendering wrappers.

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

For the open-generator path, use `--backend qwen-image-edit`; the full `genevolve` environment above includes the Qwen-Image-Edit wrapper dependencies.

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

## 🧩 Optional Python Usage

If you only want to run the provided scripts, you can skip this section. This is for users who want to call the agent and renderer directly from their own Python pipeline instead of going through `scripts/run_agent.py` and `scripts/generate_images.py`.

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

The full training scripts are not included in this repository, but the released SFT/RL datasets, model weights, tools, and runtime let you reproduce the path from a user request to a rendered image.

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
| Qwen local renderer fails at import time | Install the full `genevolve` environment and make sure `diffusers>=0.38` is active. |
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
