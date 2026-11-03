# Training setup

Use the existing training Dockerfiles for SFT and RL. Inference has its
own [lighter setup](inference.md), and optional tool services have a
[separate deployment guide](tools.md).

## Target environment

| Component | Version |
| --- | --- |
| PyTorch / CUDA | 2.8.0 / 12.8 |
| torchvision / torchaudio | 0.23.0 / 2.8.0 |
| Bundled verl | 0.6.1 |
| Transformers | 4.57.1 |
| Primary rollout backend | vLLM 0.11.0 |
| Alternative rollout backend | SGLang 0.5.5, in a separate image |

The full image installs the bundled SFT, rLLM, verl, Megatron-LM, and
mbridge code along with Ray, DeepSpeed, and the training dependencies.
These packages are not required in the inference client environment.

## Build from this checkout

Run from the repository root. The final `.` is the source build context:

```bash
# Primary SFT/RL environment; matches the retained vLLM RL recipe.
docker build -f setup/train.DOCKERFILE \
  --build-arg ROLLOUT_BACKEND=vllm -t opensearch-vl:train .

# Alternative RL environment using SGLang.
docker build -f RL/Dockerfile.sglang \
  --build-arg ROLLOUT_BACKEND=sglang -t opensearch-vl:sglang .
```

Both Dockerfiles copy the current checkout. Build the revision you intend
to run. Each image installs one rollout backend because vLLM and SGLang
have conflicting dependency requirements at these versions.
The retained RL recipe uses vLLM; an SGLang image does not change that recipe.

The root [`.dockerignore`](../.dockerignore) excludes private environment
files, private model configuration, data, weights, checkpoints, and run
artifacts. Provide model/data and private configuration at runtime. Training
also needs writable output and tokenized-cache paths.

## Prepare the training run

- SFT: prepare the ShareGPT data and registry, model weights, and a
  shared cache/output location.
- RL: register the training/validation parquets and configure tool/model
  services on all workers.
- Start Ray on the intended training nodes and configure network interfaces
  for that cluster. The recipes' node/GPU counts describe the retained runs.

Use the launch commands in the [root training setup](../README.md#3-training-setup).

The release contains one 4-epoch Qwen3-VL-8B SFT recipe and one outcome-only
RLOO RL recipe with two PPO epochs per batch.
Training data will be released later.

Complete image builds and GPU training smoke runs still require validation
in the target environment. The current checks cover source/configuration
and isolated command behavior.
