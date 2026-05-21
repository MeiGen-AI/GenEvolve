<div align="center">

<img src="assets/logo_genevolve.png" alt="GenEvolve" width="160">

<h1>GenEvolve</h1>

<p><strong><em>Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation</em></strong></p>

<p>
  <a href="https://ephemeral182.github.io/GenEvolve/">
    <img alt="Project Page" src="https://img.shields.io/badge/🌐_Project-Page-1f6feb"></a>
  <a href="https://huggingface.co/MeiGen-AI/GenEvolve">
    <img alt="Weights" src="https://img.shields.io/badge/🤗_HuggingFace-GenEvolve-FFD21E"></a>
  <a href="https://github.com/Ephemeral182/GenEvolve">
    <img alt="GitHub" src="https://img.shields.io/badge/💾_GitHub-Code-181717"></a>
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

The released `GenEvolve` policy is based on Qwen3-VL-8B and is designed to be **generator-transferable**: the same agent output can be rendered by the open Qwen-Image-Edit backend or by a stronger proprietary renderer such as Nano Banana Pro.

## 🎁 What's released

| Component | Where |
|---|---|
| 🧠 Trained agent policy `GenEvolve` (Qwen3-VL-8B-based) | 🤗 [`MeiGen-AI/GenEvolve`](https://huggingface.co/MeiGen-AI/GenEvolve) |
| ⚡ Standalone inference runtime (`GenEvolveAgent`, OpenAI-compatible) | this repo |
| 🛠️ Three tools (`search`, `image_search`, `query_knowledge`) | this repo |
| 📚 The eight skill markdown files used at training time | this repo |
| 🎨 Reference-conditioned generator wrappers (Qwen-Image-Edit + Nano Banana Pro) | this repo |
| 📦 SFT trajectories (9,000 records) | 🤗 [`MeiGen-AI/GenEvolve-Data-Bench`](https://huggingface.co/datasets/MeiGen-AI/GenEvolve-Data-Bench) / `GenEvolve-Data-SFT/` |
| 🎯 Self-evolution prompts + GT images (3,175 records) | 🤗 [`MeiGen-AI/GenEvolve-Data-Bench`](https://huggingface.co/datasets/MeiGen-AI/GenEvolve-Data-Bench) / `GenEvolve-Data-RL/` |
| 📊 Held-out evaluation benchmark (594 prompts + GT images) | 🤗 [`MeiGen-AI/GenEvolve-Data-Bench`](https://huggingface.co/datasets/MeiGen-AI/GenEvolve-Data-Bench) / `GenEvolve-Bench/` |

## 📋 Requirements

GenEvolve has a main runtime environment for policy serving, agent rollouts, tool execution, and benchmark inference. This is not the only process used in a full image-generation pipeline: for reproducible Qwen rendering, run Qwen-Image-Edit as a separate FastAPI/diffusers service and call it from GenEvolve through `--service-url`.

### Main GenEvolve runtime - `genevolve`

Use this environment for the released agent code path: serving `GenEvolve`, running the agent, calling tools, using the Nano client, and calling a Qwen service endpoint. Install it once using the Quickstart commands below.

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 |
| CUDA stack | CUDA 12.x; our logs used PyTorch CUDA 12.8 wheels |
| `torch` / `torchvision` | `2.8.0` / `0.23.0` |
| `transformers` | `4.57.1` |
| `vllm` | `0.11.0` |
| `ray` | `2.54.1` |
| `flash-attn` | `2.8.3` |

This environment does not install or launch external services such as Qwen-Image-Edit, Serper, or the Google image API. Those are configured separately.

### External services

| Service | Variable | Used for |
|---|---|---|
| [serper.dev](https://serper.dev) | `SERPER_API_KEY` | required for `search` and `image_search` |
| [Google Generative Language API](https://ai.google.dev/api) | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | only for `--backend nano-banana-pro` |
| Qwen-Image-Edit FastAPI service | `--service-url` | only for `--backend qwen-image-edit-service` |

### Qwen-Image-Edit service environment

For Qwen rendering, use a separate service environment instead of mixing the diffusion stack into the vLLM server. A typical working stack is Python 3.11, PyTorch/torchvision `2.6.0`/`0.21.0` with CUDA 12.4 wheels, `diffusers>=0.38`, `transformers>=4.57`, `accelerate`, `fastapi`, `uvicorn`, `pillow`, and `requests`.

```bash
conda create -n qwenimage python=3.11 -y
conda activate qwenimage
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install "diffusers>=0.38" "transformers>=4.57" accelerate fastapi uvicorn pillow requests
```

Start any Qwen-Image-Edit FastAPI service compatible with `POST /generate`; a common deployment is one Qwen pipeline per visible GPU, with one HTTP endpoint such as `http://host:8001`. GenEvolve sends requests with `--backend qwen-image-edit-service --service-url http://host:8001`.

## 🚀 Quickstart

### 1. Install the main GenEvolve runtime

```bash
git clone https://github.com/Ephemeral182/GenEvolve.git
cd GenEvolve

conda create -n genevolve python=3.11 -y
conda activate genevolve
pip install -U pip setuptools wheel packaging psutil ninja
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation -r requirements.txt
pip install -e .
```

This installs only the main GenEvolve runtime: vLLM serving, the agent tools, and lightweight generator clients/wrappers. It does not install or start the separate Qwen-Image-Edit service; set up that service from the Qwen environment section above when using `--backend qwen-image-edit-service`.

### 2. Serve the released checkpoint

Put the Hugging Face checkpoint directory in `MODEL_PATH`. The serving scripts support both tensor parallelism (`TP`) and data parallel replicas (`DP`).

- `TP` shards one model replica across multiple GPUs.
- `DP` launches multiple model replicas to improve throughput for many concurrent prompts.
- Total GPU usage is `TP × DP`.
- Use a larger `DP` when `scripts/run_agent.py --parallel` is large and each request fits on one GPU.
- Use a larger `TP` when one model replica needs more memory or longer context than one GPU can provide.

```bash
# Single GPU / single replica.
MODEL_PATH=MeiGen-AI/GenEvolve PORT=8000 TP=1 DP=1 bash scripts/serve_vllm.sh

# Higher throughput on one 8-GPU node: 8 replicas, one GPU per replica.
MODEL_PATH=MeiGen-AI/GenEvolve PORT=8000 TP=1 DP=8 bash scripts/serve_vllm.sh

# If one replica needs more memory: 4 replicas, two GPUs per replica.
MODEL_PATH=MeiGen-AI/GenEvolve PORT=8000 TP=2 DP=4 bash scripts/serve_vllm.sh
```

For example, `TP=8 DP=1` is one model replica sharded over 8 GPUs. It is not 8 independent services. For throughput on one 8-GPU node, prefer `TP=1 DP=8` if the model fits on one GPU; use `TP=2 DP=4` or `TP=4 DP=2` when each replica needs multiple GPUs.

### 3. Run an end-to-end example

```bash
export SERPER_API_KEY=<your_key>             # required for search and image_search
export GOOGLE_API_KEY=<your_key>             # or GEMINI_API_KEY; only for Nano Banana Pro

python examples/quickstart.py \
    --backend nano-banana-pro \
    --base-url http://localhost:8000/v1 \
    --model GenEvolve \
    --prompt "A 1990s travel-magazine cover of two backpackers in front of the Eiffel Tower at golden hour, the title \"PARIS\" rendered in bold serif type." \
    --output paris.png
```

For the open-generator path, use `--backend qwen-image-edit-service` with one or more Qwen-Image-Edit service endpoints:

```bash
python examples/quickstart.py \
    --backend qwen-image-edit-service \
    --service-url http://your-qwen-service:8001 \
    --base-url http://localhost:8000/v1 \
    --model GenEvolve \
    --output paris_qwen.png
```

`--backend qwen-image-edit` is kept only as a local diffusers debug path when the Qwen-Image-Edit dependencies are installed in the active environment.

### 4. Batch pipeline

The agent rollout and the heavy image rendering are split into two stages so they can run on different machines.

```bash
# Stage 1: agent rollouts -> results.json.
python scripts/run_agent.py \
    --input examples/example_prompts.jsonl \
    --output-dir runs/example \
    --base-url http://localhost:8000/v1 \
    --model GenEvolve \
    --parallel 4

# Stage 2a: render through one or more Qwen-Image-Edit services.
# Repeating --service-url enables round-robin dispatch; --parallel sends
# concurrent requests so multiple service workers can be used.
python scripts/generate_images.py \
    --input runs/example/results.json \
    --output-dir runs/example_qwen_service \
    --backend qwen-image-edit-service \
    --service-url http://your-qwen-service-1:8001 \
    --service-url http://your-qwen-service-2:8001 \
    --parallel 8

# Stage 2b: render with Nano Banana Pro.
python scripts/generate_images.py \
    --input runs/example/results.json \
    --output-dir runs/example_nano \
    --backend nano-banana-pro \
    --parallel 4
```

Current script support:

| Stage | Script | Scaling knobs |
|---|---|---|
| Agent model serving | `scripts/serve_vllm.sh` | `TP`, `DP`, `PORT`, `MAX_MODEL_LEN`, `MODEL_PATH` |
| Agent rollouts | `scripts/run_agent.py` | `--parallel`, `--base-url`, `--model` |
| Remote Qwen rendering | `scripts/generate_images.py --backend qwen-image-edit-service` | repeat `--service-url` and set `--parallel` |
| Local Qwen debug rendering | `scripts/generate_images.py --backend qwen-image-edit` | single local process; requires a Qwen-compatible diffusers environment |
| Nano rendering | `scripts/generate_images.py --backend nano-banana-pro` | `--parallel`, subject to API quota/rate limits |

### 5. Benchmark scoring

To reproduce benchmark metrics, download the public dataset and pass the
benchmark JSONL directly to the agent runner. The public benchmark uses
`question` as the prompt field; `scripts/run_agent.py` accepts both `question`
and `prompt`, preserves extra fields such as `gt_image`, `eval_type`,
`category`, and `difficulty`, and the rendering script copies them into its
output `results.json`.

The scorer in `scripts/evaluate_images.py` is the paper-compatible Gemini judge:
it uses the same rubric prompt, the same image order (Image 1 = generated,
Image 2 = GT), the same OpenAI-compatible multimodal chat-completions call, and
the same score normalization and weighted overall formula used for the reported
benchmark numbers. No service endpoint or API key is hard-coded.

Public benchmark row format:

```jsonl
{"id": "0", "question": "A detailed image-generation request...", "gt_image": "images/case_00000.jpg", "eval_type": "Knowledge-Anchored", "category": "architecture_landmark", "difficulty": "hard"}
```

Run the same two-stage pipeline, then score the rendered images with Gemini:

```bash
huggingface-cli download MeiGen-AI/GenEvolve-Data-Bench \
    --repo-type dataset \
    --local-dir ./GenEvolve-Data-Bench

# Stage 1: agent rollouts.
python scripts/run_agent.py \
    --input ./GenEvolve-Data-Bench/GenEvolve-Bench/test.jsonl \
    --output-dir runs/bench_agent \
    --base-url http://localhost:8000/v1 \
    --model GenEvolve \
    --parallel 16

# Stage 2: render images, for example through Qwen-Image-Edit services.
python scripts/generate_images.py \
    --input runs/bench_agent/results.json \
    --output-dir runs/bench_qwen \
    --backend qwen-image-edit-service \
    --service-url http://your-qwen-service:8001 \
    --parallel 16

# Stage 3: Gemini judge.
# Use an OpenAI-compatible Gemini chat-completions endpoint.
export OPENAI_API_KEY=<your_eval_api_key>
export OPENAI_API_BASE=<your_openai_compatible_base_url>
python scripts/evaluate_images.py \
    --results runs/bench_qwen/results.json \
    --gt-root ./GenEvolve-Data-Bench/GenEvolve-Bench \
    --model gemini-3.1-pro-preview \
    --max-workers 16 \
    --rpm 60 \
    --resume
```

`scripts/evaluate_images.py` writes:

| File | Contents |
|---|---|
| `results_eval.json` | per-sample judge output and rationale |
| `summary.json` | aggregate metrics |
| `summary.csv` | the same metrics in table form |

`results_eval.json` also appends benchmark split summaries such as
`eval_type:Knowledge-Anchored`, `eval_type:Quality-Anchored`, and
`overall_avg`.

The reported metrics are `faithfulness`, `visual_correctness`,
`text_accuracy`, `aesthetics`, and the weighted `overall` score:

```text
overall = 0.1 * faithfulness
        + 0.4 * visual_correctness
        + 0.4 * text_accuracy
        + 0.1 * aesthetics
```

`overall_missing_zero` keeps the full denominator and treats missing or failed
cases as zero. The summary also reports metrics by `eval_type`, `category`,
and `difficulty` when those fields are present.

## 🧩 Optional Python Usage

If you only want to run the provided scripts, you can skip this section. This is for users who want to call the agent and renderer directly from their own Python pipeline instead of going through `scripts/run_agent.py` and `scripts/generate_images.py`.

```python
from genevolve import GenEvolveAgent
from genevolve.generator import QwenImageEditServiceGenerator  # or NanoBananaProGenerator

agent = GenEvolveAgent(
    model="GenEvolve",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)
result = agent.run("A cyberpunk version of the Sydney Opera House at sunset.")

# z = (gen_prompt, reference_images)
print(result.gen_prompt)
for r in result.reference_images:
    print(r["img_id"], r["local_path"], r["note"])

backend = QwenImageEditServiceGenerator(["http://your-qwen-service:8001"])
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

We release the training data and benchmark in one Hugging Face dataset repository: [`MeiGen-AI/GenEvolve-Data-Bench`](https://huggingface.co/datasets/MeiGen-AI/GenEvolve-Data-Bench). The total trajectory data is too large for GitHub but installs in one line via 🤗 `datasets` / `huggingface-cli`.

| Dataset | Records | Size | Purpose |
|---|---|---|---|
| `GenEvolve-Data-SFT/` | 9,000 records | ~7.4 GB | Multi-turn tool-orchestrated trajectories used for the SFT cold start. Each record: `messages` (chat-format ReAct trajectory ending in `<answer>{gen_prompt, reference_images}`) + `images` (reference jpegs). |
| `GenEvolve-Data-RL/` | 3,175 records | ~680 MB | Open-ended user requests paired with curated GT images. Used for GRPO + Visual Experience Distillation, where multiple agent rollouts per prompt are scored against the GT. |
| `GenEvolve-Bench/` | 594 prompts | ~120 MB | Held-out evaluation benchmark. Contains both **Knowledge-Anchored** (335) and **Quality-Anchored** (259) tracks plus per-prompt category, difficulty, and skill metadata. |

### Quick load

```bash
pip install -U huggingface_hub datasets

huggingface-cli download MeiGen-AI/GenEvolve-Data-Bench \
    --repo-type dataset \
    --local-dir ./GenEvolve-Data-Bench
```

```python
from datasets import load_dataset

repo_id = "MeiGen-AI/GenEvolve-Data-Bench"

bench = load_dataset(repo_id, "bench", split="test")
print(bench[0]["question"], bench[0]["gt_image"])

rl = load_dataset(repo_id, "rl", split="train")
sft = load_dataset(repo_id, "sft", split="train")
print(sft[0]["messages"])
print(sft[0]["images"])
```

All paths inside the datasets are relative, for example `images/case_00512.jpg` or `images/traj_00213/IMG_001.jpg`; resolve them against the dataset directory you downloaded to. Per-dataset usage notes live on each dataset's Hub page.

The full training scripts are not included in this repository, but the released SFT/RL datasets, model weights, tools, and runtime let you reproduce the path from a user request to a rendered image.

## 🖼️ Visual results

<p align="center"><img src="assets/visual_comparison.png" alt="Qualitative comparison" width="100%"></p>

<p align="center"><sub>The same <code>GenEvolve</code> policy paired with two different reference-conditioned generators. <span style="color:#D97706">Orange</span> marks external/uncommon knowledge, <span style="color:#2563EB">blue</span> marks internal generation-knowledge requirements.</sub></p>

### 🎨 Extended gallery - paired with Nano Banana Pro

<p align="center"><img src="assets/gallery_nano.jpg" alt="GenEvolve + Nano Banana Pro gallery" width="100%"></p>

<p align="center"><sub>Additional qualitative results of <code>GenEvolve</code> with Nano Banana Pro as the downstream renderer. The agent autonomously orchestrates search, reference selection, and skill activation across diverse open-ended categories: spatial layout, text rendering, quantity counting, attribute binding, anatomy/pose, creative transfer, material physics, and aesthetic drawing.</sub></p>

### 🎨 Extended gallery - paired with Qwen-Image-Edit (open)

<p align="center"><img src="assets/gallery_qwen.jpg" alt="GenEvolve + Qwen-Image-Edit gallery" width="100%"></p>

<p align="center"><sub>The same trained agent policy paired with the open-source Qwen-Image-Edit-2511 renderer. Consistent quality across both generators demonstrates that <code>GenEvolve</code> learns generator-transferable tool orchestration rather than overfitting to one specific renderer.</sub></p>

## ⚙️ Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_BASE_URL` | OpenAI-compatible chat-completions endpoint | `http://localhost:8000/v1` |
| `OPENAI_API_KEY` | API key for the inference server or the OpenAI-compatible evaluator endpoint | `EMPTY` for local inference |
| `OPENAI_API_BASE` | OpenAI-compatible Gemini judge endpoint used by `scripts/evaluate_images.py` | provider-specific |
| `SERPER_API_KEY` | [serper.dev](https://serper.dev) key for text and image search | required |
| `SERPER_BASE_URL` | Override for Serper-compatible gateways | `https://google.serper.dev` |
| `IMAGE_DOWNLOAD_DIR` | Local cache for `image_search` downloads | `/tmp/genevolve_images` |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Google Generative Language API key | required for Nano backend |

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
| Agent cannot connect to the model | Confirm the vLLM server is running and `OPENAI_BASE_URL` or `--base-url` ends with `/v1`. |
| Qwen local renderer fails at import time | Use a separate Qwen-Image-Edit service environment and call it with `qwen-image-edit-service`; avoid mixing incompatible `xformers` / `flash-attn` combinations into the renderer env. |
| Qwen renderer says it needs a reference image | Qwen-Image-Edit is reference-conditioned; rerun the agent or use Nano Banana Pro for no-reference prompts. |
| `evaluate_images.py` cannot find GT images | Keep `gt_image` in each input record and pass `--gt-root` pointing to the downloaded benchmark directory. |
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
│   ├── run_agent.py           # batch agent rollouts -> results.json
│   ├── generate_images.py     # render images from results.json
│   └── evaluate_images.py     # Gemini judge scoring and metric summary
├── examples/
│   ├── quickstart.py          # single-prompt end-to-end example
│   └── example_prompts.jsonl
├── assets/                    # README figures
├── requirements.txt
├── setup.py
└── README.md
```

## 🙏 Acknowledgements

We thank the authors and maintainers of **[Gen-Searcher](https://github.com/RUCAIBox/Gen-Searcher)**, **Qwen3-VL**, **Qwen-Image-Edit**, **vLLM**, Serper.dev, and the Google Generative Language API.

## 📝 Citation

```bibtex
@inproceedings{chen2026genevolve,
  title     = {GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation},
  author    = {Chen, Sixiang and Xing, Zhaohu and Ye, Tian and Geng, Xinyu and Lin, Yunlong and Lai, Jianyu and He, Xuanhua and Zhai, Fuxiang and Gao, Jialin and Zhu, Lei},
  booktitle = {xxxx},
  year      = {2026}
}
```

## 📜 License

Code is released under the [Apache 2.0](LICENSE) license. Released model weights inherit the upstream license of `Qwen3-VL-8B-Instruct`. Search results returned by Serper.dev and images rendered by Nano Banana Pro / Qwen-Image-Edit are governed by the respective upstream service terms.
