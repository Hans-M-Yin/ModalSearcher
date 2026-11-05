<div align="center">

  <h1>ModalSearcher</h1>
  <p><b>将视觉证据融入自主跨模态深度搜索</b></p>

  [![arXiv](https://img.shields.io/badge/arXiv-2611.06131-b31b1b?style=for-the-badge))](http://arxiv.org/abs/2611.06131)
  [![Code](https://img.shields.io/badge/GitHub-ModalSearcher-181717?style=for-the-badge)](https://github.com/Hans-M-Yin/ModalSearcher)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

  [模型权重](https://huggingface.co/hansQAQ/ModalSearcher-8B) | [SFT-only 权重](https://huggingface.co/hansQAQ/ModalSearcher-8B-SFT) | [训练数据](https://huggingface.co/datasets/hansQAQ/ModalSearch-21K-Full) | [Benchmark](https://huggingface.co/datasets/hansQAQ/ModalSearch-Bench)

  [English](README.md) | [简体中文](README.zh-CN.md)

</div>

## 目录

- [Overview](#overview)
- [Setup](#setup)
  - [1. 工具配置](#1-工具配置)
  - [2. 推理配置](#2-推理配置)
  - [3. 训练配置](#3-训练配置)
- [Evaluation](#evaluation)
  - [评估数据](#评估数据)
  - [启动评估推理](#启动评估推理)
  - [LLM-as-judge](#llm-as-judge)
  - [输出结果与报告](#输出结果与报告)
- [致谢与许可证](#致谢与许可证)

## Overview

ModalSearcher 是一个多模态深度搜索智能体，会根据不断变化的信息需求，自主决定**搜索什么、使用哪种模态，以及何时切换模态**。它交替使用文本检索、图像检索、反向图像搜索和网页阅读，在多跳推理过程中收集证据。

本项目为论文 **ModalSearcher: Weaving Visual Evidence into Autonomous Cross-Modal Deep Search** 提供推理、评估、共享工具以及 SFT/RLOO 训练框架。

## Setup

```bash
git clone https://github.com/Hans-M-Yin/ModalSearcher.git
cd ModalSearcher
```

以下流程使用 SGLang 部署我们训练好的 ModalSearcher checkpoint，再由推理脚本调用该服务。工具处理模型单独部署；SFT/RL 使用训练 Docker 环境。除非另有说明，命令均从项目根目录执行。

### 1. 工具配置

ModalSearcher 使用以下工具：

| 工具 | 用途 |
| --- | --- |
| `t2t_search` | 检索文本证据和候选网页 |
| `t2i_search` | 根据文本描述检索候选图像 |
| `i2i_search` | 反向图像搜索，支持可选的区域选择 |
| `read_url` | 阅读网页/PDF，或查看检索得到的图像资源 |

#### 服务配置

三种搜索使用 Serper；`read_url` 使用 Jina Reader 或自建的 Enhanced Reader。对本地或裁剪后的图片进行反向搜索时，需要先生成可供搜索服务访问的公网图片 URL；项目使用阿里云 OSS 完成上传。

仓库直接提供待填写的 [`.env`](.env)。填写自己的 API Key、OSS 和服务配置，无需复制其他环境文件。主要字段如下：

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

`JINA_API_KEY` 仅在所使用的 Reader 服务需要认证时填写。服务地址采用官方默认值；OSS 的区域 Endpoint、Bucket 名称与访问凭证由用户提供。修改 `.env` 后，在运行推理、评估或 RL 的终端加载：

```bash
source .env
export PYTHONPATH="$PWD"
```

#### 可选：Enhanced Reader 与 Firecrawl

Enhanced Reader 基于开源 Jina Reader，提供持久化文件缓存、Markdown 清洗和多实例请求调度。`markdown_clean` 使用本地规则清洗，无需 ReaderLM 模型或 GPU。在我们的测试中，该方案可成功抓取约 90%–95% 的搜索结果 URL。

如需使用，先按 [Jina Reader 官方源码安装说明](https://github.com/jina-ai/reader#local-development)完成安装与构建，再从本仓库根目录运行以下命令（`READER_DIR` 替换为 Reader 源码目录）：

```bash
python -m pip install -r setup/requirements-tools.txt
touch utils/reader_backend_registry.txt
READER_DIR=/path/to/reader ENHANCED_READER_FETCH_STRATEGY=markdown_clean \
  bash utils/start_enhanced_reader_stack.sh
```

在 `.env` 中设置 `ENHANCED_READER_URL="http://127.0.0.1:8004/"` 即可使用（远程部署时替换主机地址），留空则使用 Jina Reader。

部分网站可能因访问限制或反爬机制而读取失败。项目支持在所选 Reader 出错、内容为空或被拦截时，回退到 Firecrawl。需要此能力时，在 `.env` 填写：

```bash
export FIRECRAWL_API_KEY=""
```

Firecrawl 为可选配置；不填写时，仅使用选定的 Reader。

#### 部署工具处理模型

`read_url` 会通过 `multimodal_process` 对网页文本做证据提取和摘要。我们的配置使用 Qwen3-VL-32B。先完成下方[推理环境安装与权重准备](#2-推理配置)，然后在预留的 GPU 上启动该服务。下面使用 4 路张量并行，默认端口为 `6658`：

```bash
# Prepare the Qwen3-VL-32B weights, then select the GPUs for this service.
MODEL_PATH=./models/qwen3-vl-32b \
  bash utils/serve_qwen3_32b_loop_sglang.sh 4 true
```

通过 `CUDA_VISIBLE_DEVICES` 选择 GPU；如果与智能体服务位于同一台机器，请为两个服务分配各自的 GPU。权重不在示例目录时，修改 `MODEL_PATH`。

在 [`synthesis/models.json`](synthesis/models.json) 的 `models.multimodal_process` 中填写部署配置。下面只展示该条目，保留文件中其他仍需使用的模型配置：

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

模型名称应与服务的 `served-model-name` 一致。默认配置对应同机服务；跨机器部署时修改 `base_url`。

#### 工具快速测试

安装推理依赖、填写配置并启动工具处理模型后，运行[工具检查脚本](scripts/check_tools.py)。脚本会自动加载项目根目录的 `.env`，调用现有封装检查摘要模型、文本搜索、图像搜索和网页阅读：

```bash
python scripts/check_tools.py
```

脚本会逐项输出检查结果；全部通过后显示 `All tool checks passed.`。

### 2. 推理配置

#### 安装环境

使用 Python 3.10 或更高版本创建独立环境。以下依赖包含 SGLang、推理客户端和本地模型支持，无需安装 verl、Megatron-LM 或 LLaMA-Factory，也无需单独选择模型服务框架版本：

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

`requirements-inference-serving.txt` 会安装所需的客户端与 GPU 依赖；`requirements-inference-extra.txt` 补充 PDF、SVG 和图像处理支持。CairoSVG 还需要操作系统提供 Cairo 库。

#### 权重与基准

模型权重：[ModalSearcher-8B](https://huggingface.co/hansQAQ/ModalSearcher-8B)（SFT + RL）和 [ModalSearcher-8B-SFT](https://huggingface.co/hansQAQ/ModalSearcher-8B-SFT)（仅 SFT）。评估基准：[ModalSearch-Bench](https://huggingface.co/datasets/hansQAQ/ModalSearch-Bench)。

将智能体权重放在 `models/ModalSearcher-8B/`，将评估文件放在 `data/benchmarks/benchmark.parquet`；下方部署与评估示例使用这些路径。

尚未准备工具处理模型的用户，可以单独下载 Qwen3-VL-32B：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-32B-Instruct",
    local_dir="./models/qwen3-vl-32b",
)
PY
```

#### 部署 ModalSearcher-8B

启动 checkpoint：

```bash
# Select the GPUs reserved for this service through CUDA_VISIBLE_DEVICES.
nohup python -m sglang.launch_server \
  --model-path ./models/ModalSearcher-8B \
  --served-model-name ModalSearcher-8B \
  --tp 2 --port 6657 --context-length 50000 \
  > /tmp/sglang.log 2>&1 &
```

将 `--model-path` 替换为实际 checkpoint 目录；`--tp 2` 使用两张可见 GPU。日志写入 `/tmp/sglang.log`。服务启动后，在 `.env` 中填写对应配置：

```bash
export AGENT_BASE_URL="http://127.0.0.1:6657/v1"
export AGENT_MODEL_NAME="ModalSearcher-8B"
export AGENT_API_KEY="EMPTY"
```

以上为本机部署示例；模型运行在其他机器时，使用可达的服务地址。接着按 [Evaluation](#evaluation) 中的命令运行完整基准。

### 3. 训练配置

训练数据：[ModalSearch-21K-Full](https://huggingface.co/datasets/hansQAQ/ModalSearch-21K-Full)。

SFT 使用 Qwen3-VL-8B 基座模型，RL 使用 SFT 输出的 checkpoint。运行前为训练文件设置数据路径和数据集注册配置，确认模型路径，按实际集群调整参数，并启动 Ray 集群。

#### 安装环境

SFT 与 RL 使用训练 Dockerfile。从项目根目录构建，命令末尾的 `.` 表示包含当前项目代码的构建上下文：

```bash
# Primary SFT/RL environment.
docker build -f setup/train.DOCKERFILE \
  --build-arg ROLLOUT_BACKEND=vllm -t opensearch-vl:train .

# Alternative RL environment.
docker build -f RL/Dockerfile.sglang \
  --build-arg ROLLOUT_BACKEND=sglang -t opensearch-vl:sglang .
```

训练环境使用 PyTorch 2.8.0、CUDA 12.8 和项目内置的 verl 0.6.1。每个镜像只安装一种 rollout 后端；数据、权重和运行时配置需要在运行容器时提供。

RL 使用与推理相同的工具、`.env` 和 `synthesis/models.json`；训练镜像已包含主要工具依赖，无需将 SGLang 推理部署依赖再安装到 vLLM 训练环境。运行 RL 前，请填写 `.env` 中的工具配置及 `JUDGE_BASE_URL`、`JUDGE_API_KEY`、`JUDGE_MODEL`，并在所有工作节点准备可访问的模型服务。

#### SFT

在训练容器中运行，默认使用 2 节点 × 8 GPU，训练 4 个 epoch：

```bash
(
  cd SFT
  USE_RAY=1 llamafactory-cli train examples/agentic_full/qwen3_vl_full_sft_8b_ray_2node16gpu_searchweave_fvqa_livevqa_16004_epoch4.yaml
)
```

#### RL

在训练容器中运行，默认使用 4 节点 × 8 GPU，进行 4 个 epoch 的 RLOO 训练，每批执行 2 个 PPO epoch。下面使用论文中的 96 个问题训练批次：

```bash
source .env
TRAIN_PROMPT_BSZ=96 bash RL/rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-4node-vllm-rloo-baseline-v3-ppo-epoch2.sh
```

## Evaluation

评估分为两步：先运行智能体并保存轨迹，再用 LLM Judge 为保存的最终答案评分。使用 [Setup](#2-推理配置) 中安装的环境，并先填写、加载 `.env`。

### 评估数据

评估使用 `data/benchmarks/benchmark.parquet`。

ModalSearch-Bench 包含 300 道经过专家验证的问题，覆盖 2、3、4 跳参考路径：

| 类别 | 题目数量 | 参考视觉证据 |
| --- | --- | --- |
| Input-only | 100 | 参考解答所需的全部视觉证据都在初始输入中提供 |
| Search-required | 200 | 参考解答需要在推理过程中获取新的视觉证据，可以有初始图像，也可以没有 |

IO/SR 标签描述参考证据路径，不限制智能体可以调用的工具。总体准确率在全部 300 题上计算，按题目数量加权。

### 启动评估推理

确认智能体和工具处理模型均已部署，然后执行完整基准。以下命令使用温度 `0.0`、`top_p=1.0`、`presence_penalty=0` 和 16 个并行任务；不设置 `--limit`，处理输入文件中的全部样本：

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

`--max-tokens 4096` 是每轮生成上限；服务端上下文长度由部署命令中的 `--context-length 50000` 设置。`--dataset test` 是运行标签，不会选择 Hugging Face 数据划分。

### LLM-as-judge

论文使用 **GPT-4o** 和图 13 的提示词，仅将**最终汇总答案**传给裁判。填写同一个 [`synthesis/models.json`](synthesis/models.json) 中的 `models.gpt4o` 条目，与 `multimodal_process` 并列。下面只展示 Judge 条目：

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

根据实际服务填写模型/部署名称、地址和 API Key。离线 Judge 使用注册表中的凭证，`.env` 中的 `JUDGE_*` 字段用于 RL 奖励模型。

对上一步生成的完整轨迹运行评分：

```bash
source .env
python opensearch_vl/eval_infer_with_llm.py \
  --traj-dir ./outputs/benchmark_8b \
  --answer-file ./data/benchmarks/benchmark.parquet \
  --judge-model-alias gpt4o \
  --prompt-style official --max-workers 8
```

推理和评分使用同一个 `./outputs/benchmark_8b` 目录。

更换 Judge 部署或提示词后，添加 `--rerun` 重新判分，避免复用已有结果。

### 输出结果与报告

上述命令将轨迹和评分报告写入：

```text
outputs/benchmark_8b/
├── run_config.json                 # initial inference configuration
├── run_invocations.jsonl           # invocation history
├── <sample_id>_trajectory.json     # completed agent trajectories
├── failures/                      # failed-case diagnostics, when present
├── llm_eval_report.json            # summary after running the judge
└── llm_eval_report_details.jsonl   # per-sample judgments
```

裁判只对已完成的 `*_trajectory.json` 文件调用 Judge，准确率以 `--answer-file` 中的完整样本数为分母；未生成完整轨迹的样本和 Judge 调用异常均按错误答案计分。推理失败的诊断信息保存在 `failures/`，Judge 调用异常另计入 `error_count`。

汇总报告 `llm_eval_report.json` 包含：

| 字段 | 含义 |
| --- | --- |
| `total` | 基准文件中的完整样本数，即准确率分母 |
| `evaluated` | 参与判分的完整轨迹数量 |
| `missing` | 未参与判分的样本数，按错误答案计分 |
| `correct` | 判为正确的数量 |
| `accuracy` | 正确数量除以 `total`，以百分比表示 |
| `error_count` | Judge 调用失败的数量 |
| `judge_model` | 判分使用的模型别名 |
| `judge_prompt_style` | 评估提示词类型 |

`llm_eval_report_details.jsonl` 保存逐样本的判分结果。

复现论文结果时，对完整基准**独立推理五次**，分别保存到不同目录，使用相同的 GPT-4o 部署和提示词评分，报告五次准确率的均值，并记录推理参数与 Judge 部署。

## 致谢与许可证

本项目基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、
[rLLM](https://github.com/rllm-org/rllm)、
[verl](https://github.com/volcengine/verl)、
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、
[mbridge](https://github.com/ISEEKYAN/mbridge)、
[vLLM](https://github.com/vllm-project/vllm)、
[SGLang](https://github.com/sgl-project/sglang) 和
[Qwen-VL](https://github.com/QwenLM/Qwen3-VL) 构建。同时感谢
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) 和
[Vision-DeepResearch](https://github.com/Alibaba-NLP/Vision-DeepResearch)。

许可证详见根目录的 [LICENSE](LICENSE)，以及各上游组件保留的许可证和版权声明。
