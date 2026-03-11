"""
Reusable SDPO code environment for LiveCodeBench tasks.

Provides:
- Task dataclass and load_tasks() for loading DeepCoder problems
- Prompt templates for SDPO teacher/student flows
- format_feedback() for converting sandbox results to LeetCode-style feedback
- SDPOCodeEnv (Env subclass) for use with tinker-cookbook RL rollouts
- SDPOCodeGroupBuilder / SDPOCodeDataset / SDPOCodeDatasetBuilder for batched training

The environment returns reward=0.0 always -- SDPO training uses KL divergence
against a teacher model (computed in the training loop) as the sole loss signal.
Grading results and feedback are carried via StepResult.logs for downstream use.

Prerequisites:
    docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
    export SANDBOX_URL=http://localhost:8080/run_code
"""

from __future__ import annotations

import ast
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

import chz
import tinker
from datasets import Dataset, concatenate_datasets, load_dataset
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.code_rl.code_env import (
    _build_question,
    _ensure_dict,
    _load_deepcoder_split,
    _normalize_tests,
)
from tinker_cookbook.recipes.code_rl.lcb_utils import fetch_live_code_bench_system_prompt
from tinker_cookbook.recipes.code_rl.code_grading import extract_code_from_model
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)

import sandbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDPO prompt templates (matching SDPO paper Table 2)
# ---------------------------------------------------------------------------

CODE_PROMPT = (
    "You are a coding expert. You will be given a coding problem, and you need "
    "to write a correct Python program that matches the specification and passes "
    "all tests. The time limit is 1 second. You may start by outlining your "
    "thought process. In the end, please provide the complete code in a code "
    "block enclosed with ``` ```.\n\n{problem}"
)

SOLUTION_SECTION = (
    "\n"
    "Correct solution:\n\n"
    "{solution}\n\n"
)

FEEDBACK_SECTION = (
    "\n"
    "The following is feedback from your unsuccessful earlier attempt:\n\n"
    "{feedback}\n\n"
)

TEACHER_PROMPT = (
    "{prompt}{solution}{feedback}"
    "Correctly solve the original question.\n"
)

# ---------------------------------------------------------------------------
# Task + loading
# ---------------------------------------------------------------------------


@dataclass
class Task:
    problem: str
    tests: list[dict[str, Any]]
    starter_code: str | None


_DEEPCODER_SUBSETS: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {"default": ("primeintellect", "taco", "lcbv5"), "lcbv5": ("lcbv5",), "codeforces": ()},
    "test": {"default": ("codeforces", "lcbv5"), "lcbv5": ("lcbv5",), "codeforces": ("codeforces",)},
}


def _load_deepcoder_subset(
    names: tuple[str, ...],
    split: Literal["train", "test"],
) -> Dataset:
    """Load and concatenate specific DeepCoder sub-datasets."""
    datasets = []
    for name in names:
        logger.info(f"  Loading {name} ({split})...")
        ds = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split=split)
        datasets.append(cast(Dataset, ds))
    return cast(Dataset, concatenate_datasets(datasets))


def _tasks_from_deepcoder_ds(ds: Dataset, n: int | None = None) -> list[Task]:
    """Convert a HuggingFace Dataset of DeepCoder rows into Task objects."""
    tasks: list[Task] = []
    for row in ds:
        if n is not None and len(tasks) >= n:
            break
        example: dict[str, Any] = row  # type: ignore[assignment]
        metadata = _ensure_dict(example.get("metadata", {}))
        tests = _normalize_tests(example.get("tests") or example.get("ground_truth"), metadata)
        if not tests:
            continue
        question = _build_question(example)
        if question is None:
            continue
        starter = example.get("starter_code")
        if isinstance(starter, str) and not starter.strip():
            starter = None
        tasks.append(Task(problem=question, tests=tests, starter_code=starter))
    return tasks


def _load_lcbv6_tasks(
    split: Literal["train", "test"],
    n: int | None = None,
) -> list[Task]:
    """Load tasks from the bzz2/live_code_bench_v6_lite_sdpo dataset.

    This dataset is uniform LiveCodeBench -- every ground_truth is a dict with
    keys {inputs, outputs, testtype, fn_name, time_limit}.  We convert directly
    instead of routing through ``_normalize_tests`` / ``taco_to_lcb_format``.
    """
    logger.info("Loading lcbv6 (%s)...", split)
    ds = load_dataset("bzz2/live_code_bench_v6_lite_sdpo", "parquet", split=split)

    tasks: list[Task] = []
    for row in cast(Dataset, ds):
        if n is not None and len(tasks) >= n:
            break
        example: dict[str, Any] = row  # type: ignore[assignment]
        gt = json.loads(example["reward_model"]["ground_truth"])

        fn_name = gt.get("fn_name") or ""
        is_functional = bool(fn_name)
        tests: list[dict[str, Any]] = []
        for inp, out in zip(gt["inputs"], gt["outputs"]):
            test: dict[str, Any] = {
                "input": str(inp),
                "output": str(out),
                "testtype": "functional" if is_functional else "stdin_stdout",
                "metadata": {"func_name": fn_name} if is_functional else {},
            }
            tests.append(test)
        if not tests:
            continue

        description = example["extra_info"]["description"]
        question = fetch_live_code_bench_system_prompt(f"{description}\n\n")
        tasks.append(Task(problem=question, tests=tests, starter_code=None))
    return tasks


def load_tasks(
    n: int | None = None,
    split: str = "test",
    seed: int = 42,
    dataset_name: str | None = None,
) -> list[Task]:
    """Load coding tasks, optionally capped at *n* problems.

    Args:
        dataset_name: None for all DeepCoder sub-datasets, or one of
            "lcbv5", "codeforces", "lcbv6".
    """
    if dataset_name == "lcbv6":
        return _load_lcbv6_tasks(split, n)  # type: ignore[arg-type]

    if dataset_name is not None:
        split_map = _DEEPCODER_SUBSETS.get(split, {})
        names = split_map.get(dataset_name)
        if names is None or len(names) == 0:
            logger.info("No %s split for dataset_name=%s, returning empty task list", split, dataset_name)
            return []
    else:
        names = None  # use _load_deepcoder_split (loads all)

    if names is not None:
        ds = _load_deepcoder_subset(names, split)  # type: ignore[arg-type]
    else:
        ds = _load_deepcoder_split(split)  # type: ignore[arg-type]

    if split == "train":
        ds = ds.shuffle(seed=seed)

    return _tasks_from_deepcoder_ds(ds, n)



# ---------------------------------------------------------------------------
# Feedback formatting
# ---------------------------------------------------------------------------


def _parse_stdout(s: str) -> Any:
    """Parse sandbox stdout (Python repr from ``print(dict)``).

    ``ast.literal_eval`` is safe -- it only parses literal values
    (strings, numbers, dicts, lists, bools, None), never executes code.
    """
    try:
        return ast.literal_eval(s.strip())
    except (ValueError, SyntaxError):
        return s


def format_feedback(details: dict[str, Any], max_length: int = 2000) -> str:
    """Format sandbox grading details into LeetCode-style feedback for the teacher prompt.

    Adapted from sdpo/replication-package/verl/utils/reward_score/feedback/code.py:format_test_feedback
    """
    def _truncate(s: str, limit: int) -> str:
        return s if len(s) <= limit else s[:limit] + "..."

    run = details.get("run_result")
    if not isinstance(run, dict):
        error = details.get("error")
        if error:
            return f"Runtime Error\n{error}"
        return "No test execution information available."

    stdout_raw = run.get("stdout", "")
    meta = _parse_stdout(stdout_raw) if isinstance(stdout_raw, str) else stdout_raw

    if not isinstance(meta, dict):
        return_code = run.get("return_code", 0)
        if return_code != 0 and isinstance(stdout_raw, str) and stdout_raw.strip():
            return f"Runtime Error\n{_truncate(stdout_raw.strip(), 500)}"
        return "No test execution information available."

    error_msg = meta.get("error_message", "")
    error_code = meta.get("error_code", 0)

    parts: list[str] = []

    if error_code == -1:
        parts.append("Compilation Error")
        if meta.get("error"):
            parts.append(_truncate(str(meta["error"]), 500))
    elif error_code == -3:
        parts.append("Time Limit Exceeded")
        if meta.get("inputs"):
            parts.append("")
            parts.append("Last Executed Input")
            parts.append(_truncate(str(meta["inputs"]), 250))
    elif error_code == -4:
        parts.append("Runtime Error")
        if meta.get("error"):
            parts.append(_truncate(str(meta["error"]), 500))
        if meta.get("inputs"):
            parts.append("")
            parts.append("Last Executed Input")
            parts.append(_truncate(str(meta["inputs"]), 250))
    elif error_code == -2 or error_msg == "Wrong Answer":
        parts.append("Wrong Answer")
        if meta.get("inputs") is not None:
            parts.append("")
            parts.append("Input")
            parts.append(_truncate(str(meta["inputs"]), 250))
        if meta.get("output") is not None:
            parts.append("")
            parts.append("Output")
            parts.append(_truncate(str(meta["output"]), 250))
        if meta.get("expected") is not None:
            parts.append("")
            parts.append("Expected")
            parts.append(_truncate(str(meta["expected"]), 250))
    else:
        parts.append(error_msg or "Unknown Error")

    result = "\n".join(parts)
    if len(result) > max_length:
        result = result[:max_length]
    return result


# ---------------------------------------------------------------------------
# SDPOCodeEnv -- single-turn code env with async sandbox grading
# ---------------------------------------------------------------------------


class SDPOCodeEnv(Env):
    """Single-turn code environment for SDPO on-policy distillation.

    Always returns ``reward=0.0`` -- SDPO supervision comes from the KL
    penalty against a teacher model, computed in the training loop.

    Grading results flow through the standard rollout data path:
    - ``StepResult.metrics``: ``correct`` (1/0), ``has_code`` (1/0)
    - ``StepResult.logs``: ``feedback`` (formatted string for teacher prompt)

    Instance state (``self.passed``, ``self.feedback``, ``self.code``) is
    also available after rollout for use in ``compute_group_rewards``.
    """

    def __init__(
        self,
        task: Task,
        renderer: renderers.Renderer,
        timeout: int = 6,
        backend: str = "sandboxfusion",
    ):
        self.task = task
        self.renderer = renderer
        self.timeout = timeout
        self.backend = backend
        # Populated by step()
        self.passed: bool = False
        self.feedback: str = ""
        self.code: str | None = None
        self.response_text: str = ""

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        prompt_text = CODE_PROMPT.format(problem=self.task.problem)
        messages: list[renderers.Message] = [{"role": "user", "content": prompt_text}]
        return self.renderer.build_generation_prompt(messages), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, _parse_success = self.renderer.parse_response(action)
        self.response_text = message["content"]
        self.code = extract_code_from_model(self.response_text)

        self.passed = False
        self.feedback = ""
        metrics: dict[str, Any] = {}
        if self.code is not None:
            with timed("grade", metrics):
                passed, details = await sandbox.grade_code(self.task.tests, self.code, timeout=self.timeout, backend=self.backend)
            self.passed = passed
            if not passed:
                self.feedback = format_feedback(details)
        else:
            self.feedback = "No code block found in response."
        metrics["correct"] = float(self.passed)
        metrics["has_code"] = float(self.code is not None)

        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
            logs={
                "feedback": self.feedback,
            },
        )


# ---------------------------------------------------------------------------
# SDPO teacher input builder (shared by play and training)
# ---------------------------------------------------------------------------


def build_sdpo_teacher_inputs(
    task: Task,
    trajectories_G: list[Trajectory],
    renderer: renderers.Renderer,
    tokenizer: Any,
) -> list[tinker.ModelInput | None]:
    """Build SDPO re-prompted teacher ModelInputs for a group of trajectories.

    For each **failed** trajectory, assembles the teacher prompt (original
    question + optional passing-sibling solution + sandbox feedback) and
    appends the student's response tokens.  The resulting ``ModelInput`` is
    ready to be passed to ``SamplingClient.compute_logprobs_async``; the
    caller slices the last ``len(student_tokens)`` logprobs.

    Returns a parallel list (one entry per trajectory).  Passing trajectories
    get ``None`` (no teacher logprobs needed).
    """
    passed_G = [bool(t.transitions[0].metrics.get("correct", 0)) for t in trajectories_G]
    feedback_G = [str(t.transitions[0].logs.get("feedback", "")) for t in trajectories_G]

    sibling_solution: str | None = None
    for i, (traj, passed) in enumerate(zip(trajectories_G, passed_G)):
        if passed:
            sibling_solution = tokenizer.decode(traj.transitions[0].ac.tokens)
            break

    teacher_inputs_G: list[tinker.ModelInput | None] = []
    for traj, passed, feedback in zip(trajectories_G, passed_G, feedback_G):
        if passed:
            teacher_inputs_G.append(None)
            continue

        solution_section = SOLUTION_SECTION.format(solution=sibling_solution) if sibling_solution else ""
        feedback_section = FEEDBACK_SECTION.format(feedback=feedback) if feedback else ""
        teacher_content = TEACHER_PROMPT.format(
            prompt=task.problem,
            solution=solution_section,
            feedback=feedback_section,
        )
        teacher_messages: list[renderers.Message] = [{"role": "user", "content": teacher_content}]
        prompt = renderer.build_generation_prompt(teacher_messages)
        prompt_ids = prompt.to_ints()
        student_tokens = list(traj.transitions[0].ac.tokens)
        full_ids = prompt_ids + student_tokens
        teacher_inputs_G.append(tinker.ModelInput.from_ints(full_ids))

    return teacher_inputs_G


# ---------------------------------------------------------------------------
# Group builder / dataset / dataset builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SDPOCodeGroupBuilder(EnvGroupBuilder):
    """Builds a group of SDPOCodeEnv instances for a single task.

    Pass *renderer* directly to reuse an existing renderer (e.g. from
    ``play_w_code_env.py``).  When *renderer* is ``None`` (the default,
    typical in training), one is created from *model_name* /
    *renderer_name*.
    """

    task: Task
    group_size: int
    model_name: str = ""
    renderer_name: str | None = None
    renderer: renderers.Renderer | None = None
    timeout: int = 6
    backend: str = "sandboxfusion"

    async def make_envs(self) -> Sequence[SDPOCodeEnv]:
        if self.renderer is not None:
            r = self.renderer
        else:
            tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
            name = self.renderer_name or self.model_name
            r = renderers.get_renderer(name, tokenizer)
        return [
            SDPOCodeEnv(task=self.task, renderer=r, timeout=self.timeout, backend=self.backend)
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        envs = [e for e in env_group if isinstance(e, SDPOCodeEnv)]
        n_passed = sum(1 for e in envs if e.passed)
        n_total = len(envs)
        return [
            (0.0, {"group_pass_rate": n_passed / n_total if n_total else 0.0, "group_has_sibling": float(0 < n_passed < n_total)})
            for _ in trajectory_group
        ]

    def logging_tags(self) -> list[str]:
        return ["sdpo_code"]


class SDPOCodeDataset(RLDataset):
    """Dataset that produces batches of SDPOCodeGroupBuilder."""

    def __init__(
        self,
        builders: list[SDPOCodeGroupBuilder],
        batch_size: int,
    ):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        index = index % len(self)
        start = index * self.batch_size
        end = start + self.batch_size
        return self.builders[start:end]

    def __len__(self) -> int:
        return math.ceil(len(self.builders) / self.batch_size)


@chz.chz
class SDPOCodeDatasetBuilder(RLDatasetBuilder):
    """Build RL datasets over coding tasks for SDPO training."""

    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    timeout: int = 6
    backend: str = "sandboxfusion"
    n_tasks: int | None = None
    n_eval_tasks: int | None = None
    dataset_name: str | None = None
    seed: int = 42

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_tasks = load_tasks(split="train", seed=self.seed, dataset_name=self.dataset_name)
        if self.n_tasks is not None:
            train_tasks = train_tasks[:self.n_tasks]
        train_builders = [
            SDPOCodeGroupBuilder(
                task=task,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                timeout=self.timeout,
                backend=self.backend,
            )
            for task in train_tasks
        ]
        train_dataset = SDPOCodeDataset(builders=train_builders, batch_size=self.batch_size)

        test_tasks = load_tasks(split="test", seed=self.seed, dataset_name=self.dataset_name)
        if self.n_eval_tasks is not None:
            test_tasks = test_tasks[:self.n_eval_tasks]
        test_builders = [
            SDPOCodeGroupBuilder(
                task=task,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=1,
                timeout=self.timeout,
                backend=self.backend,
            )
            for task in test_tasks
        ]
        test_dataset = SDPOCodeDataset(builders=test_builders, batch_size=self.batch_size)

        return train_dataset, test_dataset
