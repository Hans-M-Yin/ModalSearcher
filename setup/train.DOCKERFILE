# syntax=docker/dockerfile:1

# ==============================================================================
# OpenSearch-VL Full Environment
#
# Includes:
#   - Base CUDA / PyTorch environment
#   - vLLM or SGLang (selected with ROLLOUT_BACKEND)
#   - VERL / rLLM
#   - Megatron-LM
#   - mbridge
#   - LLaMA-Factory / SFT
#   - DeepSpeed
#
# Important:
#   - Uses the public verl CUDA 12.8 / PyTorch 2.8.0 base and official Ray.
#   - Installs the repository-local verl 0.6.1 source.
#   - The rollout backends require incompatible outlines-core versions, so
#     choose one per image with --build-arg ROLLOUT_BACKEND=vllm|sglang.
#   - flash-attn and transformer-engine are provided by the base image.
# ==============================================================================


# ------------------------------------------------------------------------------
# 0. Base image
# ------------------------------------------------------------------------------

ARG BASE_IMAGE=verlai/verl:base-verl0.6-cu128-cudnn9.8-torch2.8.0-fa2.7.4
FROM ${BASE_IMAGE}


# ------------------------------------------------------------------------------
# 1. Build arguments
# ------------------------------------------------------------------------------

ARG PROJECT_ROOT=/workspace/project
ENV PROJECT_ROOT=${PROJECT_ROOT}

ARG PIP_INDEX=https://pypi.org/simple

# Install repository-local verl.
ARG INSTALL_LOCAL_VERL=1

# Select the rollout backend. Build a separate image to use the other backend.
ARG ROLLOUT_BACKEND=vllm
ARG VLLM_VERSION=0.11.0
ARG SGLANG_VERSION=0.5.5

# Only enable if the base image really lacks transformer-engine / flash-attn.
ARG INSTALL_GPU_EXTENSIONS=0


# ------------------------------------------------------------------------------
# 2. Runtime environment
# ------------------------------------------------------------------------------

ENV ROLLOUT_NAME=${ROLLOUT_BACKEND} \
    DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    MAX_JOBS=16 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    TOKENIZERS_PARALLELISM=false \
    GRADIO_SERVER_PORT=7860 \
    API_PORT=8000

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace


# ------------------------------------------------------------------------------
# 3. System dependencies
# ------------------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------------------------
# 4. Python build tools
#
# Avoid blindly upgrading packaging / setuptools to the newest versions,
# because the base image contains packages with tighter constraints.
# ------------------------------------------------------------------------------

RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir \
        pip \
        wheel \
        "setuptools>=77.0.3,<80" \
        editables \
        "hatchling>=1.18.0"


# ------------------------------------------------------------------------------
# 5. Copy the current public-release checkout
#
# Build from the repository root:
#   docker build -f setup/train.DOCKERFILE -t opensearch-vl:train .
# Check out the desired revision before building. The root .dockerignore
# excludes private configuration, datasets, model weights, and run artifacts.
# ------------------------------------------------------------------------------

COPY . ${PROJECT_ROOT}/


# ------------------------------------------------------------------------------
# 6. Verify the checkout's rLLM compatibility with newer IPython
#
# The compatibility fix is already maintained in the local source. Check it
# without rewriting the checkout or relying on an older upstream code block.
# ------------------------------------------------------------------------------

RUN python - <<'PY'
import ast
import os
from pathlib import Path

path = Path(os.environ["PROJECT_ROOT"]) / "RL/rllm/rllm/rewards/code_utils/pyext2.py"
tree = ast.parse(path.read_text(), filename=str(path))
if not any(
    isinstance(node, ast.ImportFrom)
    and node.module == "IPython.core"
    and any(alias.name == "oinspect" for alias in node.names)
    for node in ast.walk(tree)
):
    raise RuntimeError(
        "rLLM source must explicitly import oinspect from IPython.core."
    )
print("rLLM IPython compatibility: OK")
PY


# ------------------------------------------------------------------------------
# 7. Record important base-image package versions
#
# These constraints protect the existing GPU/runtime stack from accidental
# replacement by later pip dependency resolution.
#
# The public base supplies torch and compiled GPU extensions. The matching
# torchvision/torchaudio wheels and rollout backend are installed below.
# ------------------------------------------------------------------------------

RUN python - <<'PY'
import importlib.metadata as md
import torch

assert torch.__version__.split("+", 1)[0] == "2.8.0", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda

packages = [
    "torch",
    "flash-attn",
    "transformer-engine",
]

constraints = [
    "torchvision==0.23.0",
    "torchaudio==2.8.0",
    "transformers==4.57.1",
    "ray>=2.48.0,<3",
]

print("===== Base image protected packages =====")

for name in packages:
    try:
        version = md.version(name)
        constraints.append(f"{name}=={version}")
        print(f"{name}=={version}")
    except md.PackageNotFoundError:
        print(f"{name}: NOT INSTALLED")

with open("/tmp/base-constraints.txt", "w") as f:
    f.write("\n".join(constraints))
    f.write("\n")

print()
print("===== Generated constraints =====")
print(open("/tmp/base-constraints.txt").read())
PY

ENV PIP_CONSTRAINT=/tmp/base-constraints.txt

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torchvision==0.23.0" "torchaudio==2.8.0"


# ==============================================================================
# RL stack
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Install local VERL + rLLM
#
# --no-deps is intentional.
#
# rLLM's package metadata declares dependencies such as:
#   ray
#   torch
#   torchvision
#
# The runtime dependencies, including official Ray, are installed explicitly
# below so these editable installs cannot replace the GPU stack.
# ------------------------------------------------------------------------------

WORKDIR ${PROJECT_ROOT}/RL/rllm

RUN if [[ "${INSTALL_LOCAL_VERL}" == "1" ]]; then \
        pip install \
            --no-cache-dir \
            --no-build-isolation \
            --no-deps \
            -e ./verl; \
    fi && \
    pip install \
        --no-cache-dir \
        --no-build-isolation \
        --no-deps \
        -e .


# ------------------------------------------------------------------------------
# 9. Install Megatron-LM
# ------------------------------------------------------------------------------

WORKDIR ${PROJECT_ROOT}/RL/Megatron-LM

RUN pip install \
    --no-cache-dir \
    --no-build-isolation \
    --no-deps \
    -e .


# ------------------------------------------------------------------------------
# 10. Install mbridge
# ------------------------------------------------------------------------------

WORKDIR ${PROJECT_ROOT}/RL/mbridge

RUN pip install \
    --no-cache-dir \
    --no-build-isolation \
    --no-deps \
    -e .


# ==============================================================================
# SFT stack
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Install LLaMA-Factory / SFT source
#
# Again use --no-deps so its torch / torchvision dependencies do not modify
# the base GPU stack.
# ------------------------------------------------------------------------------

WORKDIR ${PROJECT_ROOT}/SFT

RUN pip install \
    --no-cache-dir \
    --no-build-isolation \
    --no-deps \
    -e .


# ==============================================================================
# Shared RL + SFT dependencies
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Install common Python dependencies
#
# Deliberately NOT installed here:
#
#   torch
#   transformer-engine
#   flash-attn
#
# The selected rollout backend and official Ray are installed together with
# the shared dependencies so pip resolves their requirements consistently.
# ------------------------------------------------------------------------------

RUN case "${ROLLOUT_BACKEND}" in \
        vllm) set -- "vllm==${VLLM_VERSION}" ;; \
        sglang) set -- "sglang==${SGLANG_VERSION}" ;; \
        *) echo "ROLLOUT_BACKEND must be vllm or sglang" >&2; exit 1 ;; \
    esac && \
    pip install \
    --no-cache-dir \
    --upgrade-strategy only-if-needed \
    -c /tmp/base-constraints.txt \
    \
    "$@" \
    "ray[default]>=2.48.0,<3" \
    \
    "transformers>=4.55.0,<=5.2.0,!=4.52.0,!=4.57.0" \
    "datasets>=2.20.0,<=4.0.0" \
    "accelerate>=1.3.0,<=1.11.0" \
    "peft>=0.18.0,<=0.18.1" \
    "trl>=0.18.0,<=0.24.0" \
    "torchdata>=0.10.0,<=0.11.0" \
    \
    codetiming \
    dill \
    hydra-core \
    omegaconf \
    \
    "numpy<2.0.0" \
    pandas \
    polars \
    pillow \
    scipy \
    \
    "pyarrow>=19.0.0" \
    pybind11 \
    pylatexenc \
    \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    \
    wandb \
    tensorboard \
    packaging \
    \
    mcp \
    eval-protocol \
    openai \
    fastapi \
    uvicorn \
    tqdm \
    PyYAML \
    pydantic \
    \
    wrapt \
    "asgiref>=3.7.0" \
    json5 \
    aiosqlite \
    oss2 \
    "firecrawl>=4.34.0,<5" \
    "requests==2.32.3" \
    huggingface_hub \
    \
    qwen-vl-utils \
    "av>=10.0.0,<=16.0.0" \
    decord \
    safetensors \
    einops \
    pymupdf \
    \
    "gradio>=4.38.0,<=5.50.0" \
    "matplotlib>=3.7.0" \
    "tyro<0.9.0" \
    \
    sentencepiece \
    tiktoken \
    modelscope \
    hf-transfer \
    fire \
    protobuf \
    sse-starlette \
    \
    sympy \
    "antlr4-python3-runtime==4.9.3" \
    \
    nltk \
    jieba \
    rouge-chinese \
    \
    "deepspeed>=0.10.0,<=0.18.4"


# ==============================================================================
# Optional GPU extensions
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Optional transformer-engine / flash-attn installation
#
# Default is OFF.
#
# The public verl base image already contains compatible builds.
# Reinstalling these unnecessarily can introduce CUDA / PyTorch ABI problems.
# ------------------------------------------------------------------------------

RUN if [[ "${INSTALL_GPU_EXTENSIONS}" == "1" ]]; then \
        pip install \
            --no-cache-dir \
            --no-build-isolation \
            transformer_engine \
            flash-attn; \
    fi


# ==============================================================================
# Python paths
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. PYTHONPATH
# ------------------------------------------------------------------------------

ENV PYTHONPATH=${PROJECT_ROOT}:\
${PROJECT_ROOT}/SFT/src:\
${PROJECT_ROOT}/RL/rllm:\
${PROJECT_ROOT}/RL/Megatron-LM:\
${PROJECT_ROOT}/RL/mbridge:\
${PYTHONPATH}


# ==============================================================================
# Build-time validation
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Print package versions
#
# This is informational only.
# ------------------------------------------------------------------------------

RUN python - <<'PY'
import importlib.metadata as md

packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "vllm",
    "verl",
    "sglang",
    "deepspeed",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "torchdata",
    "ipython",
]

print()
print("==========================================")
print("        Installed package versions")
print("==========================================")

for name in packages:
    try:
        print(f"{name:20s}: {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name:20s}: NOT INSTALLED")

print("==========================================")
PY


# ------------------------------------------------------------------------------
# 16. Runtime import smoke test
#
# Validate the selected backend and the shared training imports.
# ------------------------------------------------------------------------------

RUN python - <<'PY'
import importlib
import os
import traceback

modules = [
    "torch",
    "torchvision",
    "ray",
    "rllm",
    "verl",
    "mbridge",
    "llamafactory",
    "deepspeed",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    os.environ["ROLLOUT_NAME"],
]

failed = []

print()
print("==========================================")
print("             Runtime smoke test")
print("==========================================")

for name in modules:
    print()
    print(f"--- importing {name} ---")

    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")

        print(f"{name}: OK")
        print(f"{name} version: {version}")

    except Exception:
        print(f"{name}: FAILED")
        traceback.print_exc()
        failed.append(name)

print()
print("==========================================")

if failed:
    raise RuntimeError(
        "Runtime import test failed: "
        + ", ".join(failed)
    )

print("All important runtime imports: OK")
PY


# ------------------------------------------------------------------------------
# 17. Verify the official Ray distribution and module
# ------------------------------------------------------------------------------

RUN python - <<'PY'
import importlib.metadata as md
import ray

assert md.version("ray") == ray.__version__

print()
print("==========================================")
print("               Ray runtime")
print("==========================================")

print("ray import: OK")
print("ray version:", getattr(ray, "__version__", "unknown"))
print("ray path:", ray.__file__)

print("==========================================")
PY


# ------------------------------------------------------------------------------
# 18. PyTorch environment report
#
# torch.cuda.is_available() may be False during docker build because build stages
# normally do not receive access to physical GPUs.
#
# Therefore CUDA availability is printed but NOT asserted.
# ------------------------------------------------------------------------------

RUN python - <<'PY'
import importlib.metadata as md
import torch

assert torch.__version__.split("+", 1)[0] == "2.8.0", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert md.version("verl") == "0.6.1", md.version("verl")

print()
print("==========================================")
print("            PyTorch environment")
print("==========================================")

print("torch version:", torch.__version__)
print("torch CUDA version:", torch.version.cuda)
print(
    "torch.cuda.is_available():",
    torch.cuda.is_available(),
)

print("==========================================")
PY


# ==============================================================================
# Runtime configuration
# ==============================================================================


# LLaMA-Factory Web UI
EXPOSE 7860

# API service
EXPOSE 8000


# Default working directory
WORKDIR ${PROJECT_ROOT}


CMD ["/bin/bash"]
