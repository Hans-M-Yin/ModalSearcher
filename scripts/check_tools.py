#!/usr/bin/env python3
"""Quick checks for the auxiliary model and shared inference/RL tools.

Run from the repository root:
    python scripts/check_tools.py
"""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from synthesis.env import load_env_file

    # Load configuration before importing clients that read it at startup.
    # Explicitly exported environment variables take precedence.
    load_env_file(PROJECT_ROOT / ".env")

    from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest
    from synthesis.sft import ToolRuntimeContext, execute_tool_call
    from synthesis.sft.tools import get_sft_qwen_model_alias

    model_alias = get_sft_qwen_model_alias()
    print(f"Checking auxiliary model: {model_alias}", flush=True)
    reply = LLM_WORKER.generate(
        ModelRequest(
            model=model_alias,
            messages=[ModelMessage(role="user", content="Reply with OK.")],
            max_tokens=128,
        )
    )
    if not reply.content or not reply.content.strip():
        raise RuntimeError("The auxiliary model returned an empty response.")
    print(f"{model_alias}: {reply.content}")

    with TemporaryDirectory(prefix="modalsearcher-tools-") as work_dir:
        context = ToolRuntimeContext(working_dir=work_dir)
        for name, arguments in [
            ("t2t_search", {"query": "Eiffel Tower official website", "top_k": 2}),
            ("t2i_search", {"query": "Eiffel Tower Paris", "top_k": 2}),
            (
                "read_url",
                {"url": "https://example.com", "goal": "What is this page used for?"},
            ),
        ]:
            print(f"\nChecking {name}...", flush=True)
            result = execute_tool_call(name, arguments, context)
            print(result.output_text[:1200])
            if result.output.get("ok") is False:
                raise RuntimeError(f"{name} failed: {result.output_text}")
            if name == "read_url" and result.output.get("content") in (
                None,
                "",
                "Unable to read the requested page.",
            ):
                raise RuntimeError(f"{name} returned no usable content.")

    print("\nAll tool checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

