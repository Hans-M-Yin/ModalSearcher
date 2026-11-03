"""Exercise rollout and cleanup methods without loading Torch or Ray."""

import ast
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock


TRAINER_PATH = (
    Path(__file__).resolve().parents[1]
    / "RL/rllm/rllm/trainer/verl/agent_workflow_trainer.py"
)


def trainer_method(name, namespace):
    tree = ast.parse(TRAINER_PATH.read_text())
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentWorkflowPPOTrainer"
    )
    method = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(TRAINER_PATH), "exec"), namespace)
    return namespace[name]


class TrainerLifecycleTests(unittest.TestCase):
    def test_training_and_validation_wait_for_results_and_forward_progress(self):
        loop = object()
        submit = MagicMock()
        timer = MagicMock(side_effect=lambda *args, **kwargs: nullcontext())
        method = trainer_method("generate_trajectories", {
            "marked_timer": timer,
            "asyncio": SimpleNamespace(run_coroutine_threadsafe=submit),
        })
        for batch in ("train", "validation"):
            with self.subTest(batch=batch):
                result = object()
                coroutine = object()
                execute = MagicMock(return_value=coroutine)
                future = Future()
                future.set_result(result)
                submit.reset_mock()
                submit.return_value = future
                progress = MagicMock()
                trainer = SimpleNamespace(
                    agent_execution_engine=SimpleNamespace(execute_tasks_verl=execute),
                    _tracking_logger=SimpleNamespace(live_enabled=True, log_live=progress),
                    _loop=loop,
                )
                timing = {}
                self.assertIs(method(trainer, batch, timing_raw=timing, seed=7), result)
                execute.assert_called_once_with(batch, seed=7, progress_callback=progress)
                submit.assert_called_once_with(coroutine, loop)
                timer.assert_called_with("generate_trajectories", timing, color="red")

    def test_rollout_failure_reaches_the_caller(self):
        coroutine = object()
        execute = MagicMock(return_value=coroutine)
        future = Future()
        future.set_exception(ValueError("rollout failed"))
        submit = MagicMock(return_value=future)
        trainer = SimpleNamespace(
            agent_execution_engine=SimpleNamespace(execute_tasks_verl=execute),
            _loop=object(),
        )
        method = trainer_method(
            "generate_trajectories",
            {
                "marked_timer": lambda *args, **kwargs: nullcontext(),
                "asyncio": SimpleNamespace(run_coroutine_threadsafe=submit),
            },
        )
        with self.assertRaisesRegex(ValueError, "rollout failed"):
            method(trainer, "train")
        execute.assert_called_once_with("train")
        submit.assert_called_once_with(coroutine, trainer._loop)

    def test_shutdown_cleans_up_engine_loop_thread_and_tracking(self):
        engine, loop, thread, tracking = (MagicMock() for _ in range(4))
        loop.is_running.return_value = True
        trainer = SimpleNamespace(
            agent_execution_engine=engine, _loop=loop, _thread=thread,
            _tracking_logger=tracking,
        )
        trainer_method("shutdown", {})(trainer)
        engine.shutdown.assert_called_once_with()
        loop.call_soon_threadsafe.assert_called_once_with(loop.stop)
        thread.join.assert_called_once_with()
        tracking.finish.assert_called_once_with()
        self.assertIsNone(trainer.agent_execution_engine)
        self.assertIsNone(trainer._tracking_logger)

    def test_shutdown_accepts_partially_initialized_trainer(self):
        trainer_method("shutdown", {})(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
