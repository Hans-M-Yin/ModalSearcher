<div align="center">

  <h1>ModalSearcher</h1>
  <p><b>Weaving Visual Evidence into Autonomous Cross-Modal Deep Search</b></p>

  [![arXiv: coming soon](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?style=for-the-badge)](http://arxiv.org/abs/2611.06131)
  [![Code](https://img.shields.io/badge/GitHub-ModalSearcher-181717?style=for-the-badge)](https://github.com/Hans-M-Yin/ModalSearcher)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

  [Model](https://huggingface.co/hansQAQ/ModalSearcher-8B) | [SFT-only Model](https://huggingface.co/hansQAQ/ModalSearcher-8B-SFT) | [Training Data](https://huggingface.co/datasets/hansQAQ/ModalSearch-21K-Full) | [Benchmark](https://huggingface.co/datasets/hansQAQ/ModalSearch-Bench)

  [English](README.md) | [简体中文](README.zh-CN.md)

</div>

## Contents

- [🔍 Overview](#overview)
- [🛠️ Setup](#setup)
  - [1. Tool setup](#1-tool-setup)
  - [2. Inference setup](#2-inference-setup)
  - [3. Training setup](#3-training-setup)
- [📊 Evaluation](#evaluation)
  - [Evaluation data](#evaluation-data)
  - [Run benchmark inference](#run-benchmark-inference)
  - [LLM-as-judge](#llm-as-judge)
  - [Outputs and reports](#outputs-and-reports)
- [🤝 Acknowledgements and licenses](#acknowledgements-and-licenses)

<a id="overview"></a>

## 🔍 Overview

ModalSearcher is a multimodal deep-search agent that autonomously decides **what to search, which modality to use, and when to switch modalities** as its information needs evolve. It alternates between text search, image search, reverse-image search, and webpage reading to collect evidence during multi-hop reasoning.

This repository provides inference, evaluation, shared tools, and SFT/RLOO training frameworks for **ModalSearcher: Weaving Visual Evidence into Autonomous Cross-Modal Deep Search**.

<a id="setup"></a>

## 🛠️ Setup

```bash
git clone https://github.com/Hans-M-Yin/ModalSearcher.git
cd ModalSearcher
```

The workflow below serves our trained ModalSearcher checkpoint with SGLang and calls it from the inference runner. The auxiliary tool model is served separately; SFT/RL use the training Docker environment. Unless stated otherwise, run commands from the repository root.

### 1. Tool setup

ModalSearcher uses the following tools:

| Tool | Purpose |
| --- | --- |
| `t2t_search` | Retrieve textual evidence and candidate webpages |
| `t2i_search` | Search for images from a text description |
| `i2i_search` | Reverse-image search, with optional region selection |
| `read_url` | Read webpages/PDFs or inspect retrieved image resources |

#### Service configuration

All three search tools use Serper. `read_url` uses Jina Reader or a self-hosted Enhanced Reader. Reverse-image search over local or cropped images first needs a publicly reachable image URL; the project uses Alibaba Cloud OSS for these uploads.

A fill-in [`.env`](.env) is included directly in the repository. Enter your API keys, OSS settings, and service configuration without copying another environment file. The main fields are:

```bash
export SERPER_SEARCH_URL="https://google.serper.dev/search"
export SERPER_IMAGES_URL="https://google.serper.dev/images"
export SERPER_LENS_URL="https://google.serper.dev/lens"
export SERPER_API_KEY=""
export JINA_READER_URL="https://r.jina.ai/"
export JINA_API_KEY=""

export OSS_ACCESS_KEY_ID=""
export OSS_ACCESS_KEY_SECRET=""
export OSS_ENDPOINT=""
export OSS_BUCKET_NAME=""
export SYNTHESIS_MODEL_CONFIG="synthesis/models.json"
export SFT_QWEN_MODEL_ALIAS="multimodal_process"
```

`JINA_API_KEY` is needed only when your Reader service requires authentication. Provider URLs use official defaults; supply your own OSS regional endpoint, bucket name, and credentials. After editing `.env`, load it in each shell that runs inference, evaluation, or RL:

```bash
source .env
export PYTHONPATH="$PWD"
```

#### Optional: Enhanced Reader and Firecrawl

Enhanced Reader wraps the open-source Jina Reader with persistent file caching, Markdown cleanup, and multi-instance request dispatch. The `markdown_clean` strategy uses local rules and requires no ReaderLM model or GPU. In our tests, it successfully fetched approximately 90–95% of URLs returned by search.

To use it, install and build Reader following the [official source-installation instructions](https://github.com/jina-ai/reader#local-development), then run the following from this repository's root (`READER_DIR` is your Reader source directory):

```bash
python -m pip install -r setup/requirements-tools.txt
touch utils/reader_backend_registry.txt
READER_DIR=/path/to/reader ENHANCED_READER_FETCH_STRATEGY=markdown_clean \
  bash utils/start_enhanced_reader_stack.sh
```

Set `ENHANCED_READER_URL="http://127.0.0.1:8004/"` in `.env` to use it (replace the host for a remote deployment), or leave the variable empty to use Jina Reader.

Some websites may fail to load because of access restrictions or anti-bot mechanisms. The tool can fall back to Firecrawl when the selected Reader errors, returns empty content, or is blocked. To use this fallback, fill in:

```bash
export FIRECRAWL_API_KEY=""
```

Firecrawl is optional; leave it unset to use only the selected Reader.

#### Deploy the auxiliary tool model

`read_url` uses `multimodal_process` to extract relevant evidence and summarize webpage text. Our configuration uses Qwen3-VL-32B. First complete the [inference environment and weight preparation](#2-inference-setup), then start this service on its reserved GPUs. The example uses tensor parallelism of 4 and port `6658`:

```bash
# Prepare the Qwen3-VL-32B weights, then select the GPUs for this service.
MODEL_PATH=./models/qwen3-vl-32b \
  bash utils/serve_qwen3_32b_loop_sglang.sh 4 true
```

Select GPUs through `CUDA_VISIBLE_DEVICES`. If the agent and auxiliary services share a machine, allocate separate GPUs to each. Change `MODEL_PATH` when the weights are stored elsewhere.

Configure `models.multimodal_process` in [`synthesis/models.json`](synthesis/models.json). The example below shows only this entry; retain other model entries that you use:

```json
{
  "models": {
    "multimodal_process": {
      "client_type": "openai",
      "served_model": "Qwen3-VL-32B",
      "base_url": "http://127.0.0.1:6658/v1",
      "api_key": "EMPTY",
      "timeout_s": 120,
      "sampling_params": {
        "greedy": false,
        "top_p": 0.8,
        "top_k": 20,
        "temperature": 0.7,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "out_seq_length": 8192
      }
    }
  }
}
```

The model name must match the service's `served-model-name`. The default address assumes a service on the same machine; change `base_url` for a remote deployment.

#### Quick tool checks

After installing the inference dependencies, filling the configuration, and starting the auxiliary model, run the [tool-check script](scripts/check_tools.py). It automatically loads the root `.env` and calls the existing wrappers to check the auxiliary model, text search, image search, and webpage reading:

```bash
python scripts/check_tools.py
```

The script prints each check's result and displays `All tool checks passed.` when every check succeeds.

### 2. Inference setup

#### Install dependencies

Create a separate environment with Python 3.10 or newer. These requirements include SGLang, the inference client, and local model support. They do not require verl, Megatron-LM, or LLaMA-Factory, or a separate choice of model-serving package versions:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r setup/requirements-inference-serving.txt
python -m pip install -r setup/requirements-inference-extra.txt
source .env
export PYTHONPATH="$PWD"
```

`requirements-inference-serving.txt` installs the client and GPU dependencies; `requirements-inference-extra.txt` adds PDF, SVG, and image-processing support. CairoSVG also needs the system Cairo library.

#### Weights and benchmarks

Model weights: [ModalSearcher-8B](https://huggingface.co/hansQAQ/ModalSearcher-8B) (SFT + RL) and [ModalSearcher-8B-SFT](https://huggingface.co/hansQAQ/ModalSearcher-8B-SFT) (SFT-only). Evaluation benchmark: [ModalSearch-Bench](https://huggingface.co/datasets/hansQAQ/ModalSearch-Bench).

Place the agent checkpoint in `models/ModalSearcher-8B/` and the benchmark file at `data/benchmarks/benchmark.parquet`. The deployment and evaluation examples below use these paths.

Download the Qwen3-VL-32B auxiliary model separately if you have not already prepared it:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-32B-Instruct",
    local_dir="./models/qwen3-vl-32b",
)
PY
```

#### Deploy ModalSearcher-8B

Serve the checkpoint:

```bash
# Select the GPUs reserved for this service through CUDA_VISIBLE_DEVICES.
nohup python -m sglang.launch_server \
  --model-path ./models/ModalSearcher-8B \
  --served-model-name ModalSearcher-8B \
  --tp 2 --port 6657 --context-length 50000 \
  > /tmp/sglang.log 2>&1 &
```

Change `--model-path` to your checkpoint directory. `--tp 2` uses two visible GPUs, and logs are written to `/tmp/sglang.log`. Set the corresponding fields in `.env`:

```bash
export AGENT_BASE_URL="http://127.0.0.1:6657/v1"
export AGENT_MODEL_NAME="ModalSearcher-8B"
export AGENT_API_KEY="EMPTY"
```

These addresses assume a local deployment; use a reachable service address for another machine. Then run the full benchmark as described in [Evaluation](#evaluation).

### 3. Training setup

Training data: [ModalSearch-21K-Full](https://huggingface.co/datasets/hansQAQ/ModalSearch-21K-Full).

SFT starts from the Qwen3-VL-8B base model; RL starts from the SFT checkpoint. Before training, set the data paths and dataset registrations for your training files, check model paths, adjust the configuration for your cluster, and start Ray.

#### Environment

Use the training Dockerfiles for SFT and RL. Build from the repository root; the final `.` supplies the current source as the build context:

```bash
# Primary SFT/RL environment.
docker build -f setup/train.DOCKERFILE \
  --build-arg ROLLOUT_BACKEND=vllm -t opensearch-vl:train .

# Alternative RL environment.
docker build -f RL/Dockerfile.sglang \
  --build-arg ROLLOUT_BACKEND=sglang -t opensearch-vl:sglang .
```

The training environment uses PyTorch 2.8.0, CUDA 12.8, and the bundled verl 0.6.1. Each image installs one rollout backend; provide data, weights, and runtime configuration when running the container.

RL shares the tools, `.env`, and `synthesis/models.json` with inference. The training images already include the main tool dependencies; do not add the SGLang inference-serving requirements to the vLLM training environment. Before RL, fill the tool configuration and `JUDGE_BASE_URL`, `JUDGE_API_KEY`, and `JUDGE_MODEL` in `.env`, and provide model services reachable from all workers.

#### SFT

Run inside the training container. The default configuration uses 2 nodes × 8 GPUs and trains for 4 epochs:

```bash
(
  cd SFT
  USE_RAY=1 llamafactory-cli train examples/agentic_full/qwen3_vl_full_sft_8b_ray_2node16gpu_searchweave_fvqa_livevqa_16004_epoch4.yaml
)
```

#### RL

Run inside the training container. The default configuration uses 4 nodes × 8 GPUs, with 4 RLOO training epochs and 2 PPO epochs per batch. The command uses the paper's 96-prompt training batch:

```bash
source .env
TRAIN_PROMPT_BSZ=96 bash RL/rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-4node-vllm-rloo-baseline-v3-ppo-epoch2.sh
```

<a id="evaluation"></a>

## 📊 Evaluation

Evaluation has two steps: run the agent and save its trajectories, then use an LLM judge to score the saved final answers. Use the environment from [Setup](#2-inference-setup), with `.env` filled in and loaded.

### Evaluation data

Evaluation uses `data/benchmarks/benchmark.parquet`.

ModalSearch-Bench contains 300 expert-verified questions with 2-, 3-, and 4-hop reference paths:

| Category | Questions | Reference visual evidence |
| --- | --- | --- |
| Input-only | 100 | All visual evidence required by the reference solution is provided in the initial input |
| Search-required | 200 | The reference solution requires new visual evidence during reasoning, with or without an initial image |

IO/SR labels describe reference evidence paths and do not restrict the agent's tools. Overall accuracy is computed over all 300 questions, weighted by question count.

### Run benchmark inference

Once the agent and auxiliary tool model are deployed, run the full benchmark. The command uses temperature `0.0`, `top_p=1.0`, `presence_penalty=0`, and 16 concurrent examples. Omitting `--limit` processes every sample in the input file:

```bash
source .env
SFT_QWEN_MODEL_ALIAS=multimodal_process python opensearch_vl/run_infer.py \
  --model 8b --backend api \
  --base-url "$AGENT_BASE_URL" \
  --served-model-name "$AGENT_MODEL_NAME" \
  --data-path ./data/benchmarks/benchmark.parquet \
  --output-dir ./outputs/benchmark_8b \
  --dataset test --start 0 --parallel-workers 16 \
  --max-tokens 4096 --api-timeout 600 --api-max-retries 3 \
  --temperature 0.0 --top-p 1.0 --top-k 20 \
  --repetition-penalty 1.0 --presence-penalty 0 --log-level INFO
```

`--max-tokens 4096` limits each generation turn; the server context length is set by `--context-length 50000` in the deployment command. `--dataset test` is a run label, not a Hugging Face split selector.

### LLM-as-judge

The paper uses **GPT-4o** with the Figure 13 prompt and passes only the **final summarized answer** to the judge. Fill `models.gpt4o` in the same [`synthesis/models.json`](synthesis/models.json), alongside `multimodal_process`. The example below shows only the judge entry:

```json
{
  "models": {
    "gpt4o": {
      "client_type": "openai",
      "served_model": "gpt-4o",
      "base_url": "https://api.openai.com/v1",
      "api_key": "",
      "timeout_s": 120,
      "sampling_params": {
        "max_tokens": 1024
      }
    }
  }
}
```

Configure your actual model/deployment name, endpoint, and API key. Offline evaluation uses credentials in this registry; the `JUDGE_*` variables in `.env` configure the RL reward model.

Score the completed trajectories from the preceding step:

```bash
source .env
python opensearch_vl/eval_infer_with_llm.py \
  --traj-dir ./outputs/benchmark_8b \
  --answer-file ./data/benchmarks/benchmark.parquet \
  --judge-model-alias gpt4o \
  --prompt-style official --max-workers 8
```

Inference and judging use the same `./outputs/benchmark_8b` directory.

After changing the judge deployment or prompt, add `--rerun` to rescore answers instead of reusing cached results.

### Outputs and reports

The commands above write trajectories and judging reports to:

```text
outputs/benchmark_8b/
├── run_config.json                 # initial inference configuration
├── run_invocations.jsonl           # invocation history
├── <sample_id>_trajectory.json     # completed agent trajectories
├── failures/                      # failed-case diagnostics, when present
├── llm_eval_report.json            # summary after running the judge
└── llm_eval_report_details.jsonl   # per-sample judgments
```

Judge requests are made only for completed `*_trajectory.json` files. Accuracy uses all samples in `--answer-file` as its denominator; samples without a completed trajectory and judge exceptions count as incorrect. Inference-failure diagnostics are saved under `failures/`, and judge exceptions also contribute to `error_count`.

The summary `llm_eval_report.json` contains:

| Field | Meaning |
| --- | --- |
| `total` | All samples in the benchmark file; the accuracy denominator |
| `evaluated` | Completed trajectories selected for judging |
| `missing` | Samples not evaluated, counted as incorrect |
| `correct` | Answers judged correct |
| `accuracy` | `correct / total`, expressed as a percentage |
| `error_count` | Failed judge calls |
| `judge_model` | Model alias used for judging |
| `judge_prompt_style` | Evaluation prompt style |

`llm_eval_report_details.jsonl` stores per-sample judgments.

To reproduce the paper's results, run the complete benchmark **independently five times**, save each run in its own directory, and score all runs with the same GPT-4o deployment and prompt. Report the mean of the five accuracies and record the inference parameters and judge deployment.

<a id="acknowledgements-and-licenses"></a>

## 🤝 Acknowledgements and licenses

Built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
[rLLM](https://github.com/rllm-org/rllm),
[verl](https://github.com/volcengine/verl),
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
[mbridge](https://github.com/ISEEKYAN/mbridge),
[vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang), and
[Qwen-VL](https://github.com/QwenLM/Qwen3-VL).
We also thank [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and
[Vision-DeepResearch](https://github.com/Alibaba-NLP/Vision-DeepResearch).

See the root [LICENSE](LICENSE) and the licenses and copyright notices retained for the bundled upstream components.
