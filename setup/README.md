# Setup guides

Choose an environment for the work you want to run. All commands in these
guides start at the repository root unless a different directory is stated.

| Task | Guide | Environment |
| --- | --- | --- |
| Inference against an existing model endpoint; LLM evaluation | [Inference](inference.md#api-client-setup) | Lightweight Python client and shared tool clients |
| Load ModalSearcher weights in the inference process | [Local inference](inference.md#local-gpu-setup) | Inference dependencies plus PyTorch, Transformers, and Qwen-VL utilities |
| SFT or RL training | [Training](training.md) | Existing CUDA training Dockerfiles and bundled training frameworks |
| Serve our trained checkpoints | [Checkpoint serving](inference.md#checkpoint-serving) | GPU inference environment with SGLang |
| Optional webpage extraction services | [Tool services](tools.md) | Jina Reader and Enhanced Reader, deployed independently |

The inference setup is separate from the SFT/RL training frameworks.
`requirements-inference-serving.txt` includes SGLang for serving our
checkpoints; client-only and in-process model-loading dependencies remain
available as separate requirement files.

Provider credentials and auxiliary model aliases are shared by inference
and RL; see [tool configuration](../README.md#1-tool-setup).
Download model weights and evaluation benchmarks from
[Hugging Face](../README.md#weights-and-benchmarks).
Training data will be released later.
