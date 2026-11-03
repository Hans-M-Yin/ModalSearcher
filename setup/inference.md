# Inference setup

Inference and LLM-based evaluation have their own Python environment.
Training frameworks are not required. The root README serves our trained
checkpoint with SGLang; client-only and in-process model-loading options
are also available.
All commands below run from the repository root.

## API client setup

Use Python 3.10+ and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r setup/requirements-inference.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

This installs HTTP/model SDKs, parquet/image handling, and the shared search
and upload clients. It does not install PyTorch, Transformers, vLLM, verl,
Ray, or DeepSpeed. The model server and any auxiliary model endpoints run
in their own environments.

Configure the provider keys and model aliases described below, then call
an existing model endpoint:

```bash
python opensearch_vl/run_infer.py \
  --model 8b --backend api --base-url http://127.0.0.1:8000/v1 \
  --served-model-name ModalSearcher-8B \
  --data-path ./data/benchmarks/benchmark.parquet \
  --dataset test --output-dir ./outputs/benchmark_8b_api
```

The client machine does not load the agent checkpoint. Its tool clients
and the judge still need the services configured for the selected run.

## Local GPU setup

In the inference environment, install the CUDA 12.8 PyTorch wheels and
local model-loading dependencies:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r setup/requirements-inference-local.txt
```

The local requirements include the API-client requirements, PyTorch 2.8.0,
torchvision 0.23.0, Transformers 4.57.1, Accelerate, and Qwen-VL utilities.
They do not install the training frameworks or require a model server.
The CUDA installation above targets a compatible NVIDIA GPU/driver setup.

After obtaining the model weights and benchmark files:

```bash
python opensearch_vl/run_infer.py \
  --model 8b --checkpoint ./models/ModalSearcher-8B --gpus 0 \
  --data-path ./data/benchmarks/benchmark.parquet \
  --dataset test --output-dir ./outputs/benchmark_8b
```

The paper reports ModalSearcher-8B. Other compatible checkpoints can use
`32b` or `30b-a3b` with the matching variant flag.
GPU memory requirements depend on the model, context, and image workload.
See [artifact downloads](../README.md#weights-and-benchmarks).

## Optional file-format support

```bash
python -m pip install -r setup/requirements-inference-extra.txt
```

CairoSVG enables SVG rasterization, PyMuPDF enables direct PDF handling,
and headless OpenCV supports the retained image-enhancement helpers.
CairoSVG also requires the system Cairo library. These are optional extras
rather than training dependencies.

## Configure tool clients and auxiliary models

```bash
# Fill the provided .env and synthesis/models.json first.
source .env
```

The agent checkpoint, tool summarizer, and evaluation judge are independently
configured. `synthesis/models.json` must define the aliases used by your run.
See [tool configuration](../README.md#1-tool-setup) in the root README.

Using the public Reader endpoint requires no local Reader deployment.
For self-hosted Enhanced Reader, follow
[tool service setup](tools.md), then set the worker-side endpoint variables
before starting inference. Localhost URLs refer to the machine running that
process; use a worker-reachable address when services run elsewhere.

## Evaluation and serving

The same API-client environment can score saved trajectories. For the paper
protocol, configure `gpt4o` in `synthesis/models.json` for a GPT-4o deployment,
as shown in the [root evaluation guide](../README.md#llm-as-judge):

```bash
python opensearch_vl/eval_infer_with_llm.py \
  --traj-dir ./outputs/benchmark_8b \
  --answer-file ./data/benchmarks/benchmark.parquet \
  --judge-model-alias gpt4o \
  --prompt-style official --max-workers 8
```

For the recommended SGLang workflow, use [checkpoint serving](#checkpoint-serving)
below.
This API-client environment and local model-loading setup are independent
of the [SFT/RL training Docker environment](training.md).

## Checkpoint serving

The root README deploys the released ModalSearcher checkpoint, then calls
its OpenAI-compatible endpoint. Install the serving requirements in the
inference environment:

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

Prepare the released weights as described in
[weights and benchmarks](../README.md#weights-and-benchmarks), then start
the model service:

```bash
# Select the GPUs reserved for this service through CUDA_VISIBLE_DEVICES.
nohup python -m sglang.launch_server \
  --model-path ./models/ModalSearcher-8B \
  --served-model-name ModalSearcher-8B \
  --tp 2 --port 6657 --context-length 50000 \
  > /tmp/sglang.log 2>&1 &
```

Fill `AGENT_BASE_URL`, `AGENT_MODEL_NAME`, and `AGENT_API_KEY` in the root `.env`
for this service. The local example uses `http://127.0.0.1:6657/v1`,
`ModalSearcher-8B`, and `EMPTY`, respectively. Run the
[full benchmark command](../README.md#run-benchmark-inference) after loading
`.env`. The tool processing model is configured separately in
`synthesis/models.json`.
