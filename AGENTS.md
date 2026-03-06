# SDPO Agents

- All the dependencies source code is available in .venv dir, consult to validate any hypothesis or suggestion you might have about using them.

- When working with Tinker API, follow code conventions and design patterns from tinker-cookbook.

- For Tinker API, consult the documentation in tinker-llms-full.txt file in tinker-cookbook.

## Directory layout

- `env.py` -- Reusable SDPO code environment. Contains `Task` dataclass, `load_tasks()`, `grade()`, SDPO prompt templates (`CODE_PROMPT`, `TEACHER_PROMPT`, etc.), `format_feedback()`, and tinker-cookbook RL abstractions: `SDPOCodeEnv(Env)`, `SDPOCodeGroupBuilder`, `SDPOCodeDataset`, `SDPOCodeDatasetBuilder`.
- `play_w_code_env.py` -- Interactive LiveCodeBench explorer with Rich TUI. Uses `do_group_rollout` with `SDPOCodeEnv` for the `[g]enerate` flow, exercising the same env + rollout path used by training. Teacher logprob visualization is SDPO-specific TUI code.
- `sdpo_on_policy_distillation.py` -- On-policy SDPO training loop.
- `train_on_policy.py` -- Training entrypoint (fork of tinker-cookbook's `on_policy_distillation.py`).
- `replication-package/` -- Git submodule containing the original SDPO paper's codebase (verl-based).

## Key imports from tinker_cookbook

- `tinker_cookbook.renderers` / `tokenizer_utils` -- model-family-specific prompt rendering + tokenizers.
- `tinker_cookbook.recipes.code_rl.code_env` -- dataset loading (`_load_deepcoder_split`), problem construction (`_build_question`), test normalisation.
- `tinker_cookbook.recipes.code_rl.code_grading` -- code extraction from model output (`extract_code_from_model`) and sandbox grading (`sandbox_check_correctness`).
- `tinker_cookbook.completers` -- `TinkerTokenCompleter` (standard policy used for rollouts).
- `tinker_cookbook.rl.types` -- `Env`, `EnvGroupBuilder`, `RLDataset`, `RLDatasetBuilder`, `StepResult` (with `logs` field).
- `tinker_cookbook.rl.rollouts` -- `do_single_rollout`, `do_group_rollout`.

## Runtime prerequisites

- Docker sandbox: `docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609`
- Set `SANDBOX_URL=http://localhost:8080/run_code` (or pass `--sandbox-url`).

## env.py architecture

- `SDPOCodeEnv(Env)` wraps a `Task`, subclasses `Env` directly (not `ProblemEnv`). Returns `reward=0.0` always -- SDPO supervision comes from KL divergence against a teacher model in the training loop.
- `step()` runs async sandbox grading. Results flow via `StepResult.metrics` (`correct`, `has_code`) and `StepResult.logs` (`feedback` string for teacher prompt).
- `SDPOCodeGroupBuilder` builds a group of envs and reports group-level pass rate in `compute_group_rewards`.
- `SDPOCodeDatasetBuilder` loads all DeepCoder tasks and produces train/test `SDPOCodeDataset` instances.
- Prompt templates match SDPO paper Table 2: `CODE_PROMPT`, `SOLUTION_SECTION`, `FEEDBACK_SECTION`, `TEACHER_PROMPT`.
- `build_sdpo_teacher_inputs(task, trajectories_G, renderer, tokenizer)` -- shared pure function that builds SDPO re-prompted teacher `ModelInput`s from trajectory data. Returns a parallel list (`None` for passing trajectories). Used by both `play_w_code_env.py` and (future) SDPO-aware `incorporate_kl_penalty` in `train_on_policy.py`.

## play_w_code_env.py architecture

- `_handle_generate` uses `do_group_rollout(builder, TinkerTokenCompleter(...))` to run the real `SDPOCodeEnv`. This exercises the same env + rollout path that training uses.
- After rollout, reads all display data directly from trajectory transitions: `metrics["correct"]`, `metrics["has_code"]`, `logs["feedback"]`, `ac.tokens`, `ac.logprobs`.
- Teacher logprob computation uses `build_sdpo_teacher_inputs` (from env.py) + `sampling_client.compute_logprobs_async` -- the same code path that training will use.
- UI uses `rich` for TUI: `Panel` for side-by-side rollout display, `Progress` bars, `Syntax` for code/JSON, `Text` with per-token RGB styles for logprob visualisation.
