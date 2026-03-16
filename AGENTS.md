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
- `tinker_cookbook.recipes.code_rl.code_grading` -- code extraction from model output (`extract_code_from_model`); `sandbox_check_correctness` and `postprocess_lcb_sample` used by `sandbox.py`.
- `tinker_cookbook.completers` -- `TinkerTokenCompleter` (standard policy used for rollouts).
- `tinker_cookbook.rl.types` -- `Env`, `EnvGroupBuilder`, `RLDataset`, `RLDatasetBuilder`, `StepResult` (with `logs` field).
- `tinker_cookbook.rl.rollouts` -- `do_single_rollout`, `do_group_rollout`.

## Runtime prerequisites

- Docker sandbox (default): `docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609`, then `export SANDBOX_URL=http://localhost:8080/run_code`.
- **local** sandbox (`--sandbox local`): no prerequisites; runs code as a plain subprocess. No isolation — only use with trusted code.
- **bwrap** sandbox (`--sandbox bwrap`): requires `bwrap` ≥ 0.4 in PATH (`apt install bubblewrap` / `dnf install bubblewrap`). Provides unprivileged Linux namespace isolation (no network, read-only host fs) without Docker.

## Token cost

Every call to `_handle_generate` (interactive `[g]` command **or** `--generate` mode) sends requests to the Tinker API and consumes real tokens.  Keep `--n-tasks` and `-n` small when running for validation purposes.

## Validating changes non-interactively

### Tiny overfit run on LiveCodeBenchV5

```sh
python sdpo_on_policy_distillation.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    teacher_model=Qwen/Qwen3-4B-Instruct-2507 \
    dataset_name=lcbv5 \
    n_tasks=4 groups_per_batch=2 group_size=4 \
    max_step=2 eval_every=2 \
    max_eval_tasks=5 \
    learning_rate=1e-4 \
    lora_rank=128 \
	chkpt_name_prefix=sdpo_overfit \
    sandbox_backend=local
```

### Visualize logprob difference for teacher and student responses for a few samples LiveCodeBenchV6

`play_w_code_env.py` supports a `--generate` flag that runs the full generation + grading flow for all loaded tasks without any interactive prompts, then exits.  Use this to smoke-test changes without manually driving the TUI:

```bash
# Local sandbox — no Docker required; ~2 tasks × 2 samples
python play_w_code_env.py --n-tasks 2 --model Qwen/Qwen3-4B-Instruct-2507 -n 2 --seed 44 --dataset lcbv6 \
    --generate --sandbox local
```

The Rich output is identical to the interactive `[g]enerate` flow: rollout panels, grading summary, and (on failures) teacher logprob visualisation.

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

## Cursor Cloud specific instructions

- **Python venv**: The project uses a `.venv` virtualenv at `/workspace/.venv`. Always activate it before running commands: `source /workspace/.venv/bin/activate`.
- **Tinker API**: Requires `TINKER_API_KEY` environment variable (injected as a secret). All generation and training commands call the remote Tinker API and consume real tokens.
- **Dataset caveat**: The `lcbv6` dataset (`bzz2/live_code_bench_v6_lite_sdpo`) is private on HuggingFace. Use `--dataset lcbv5` (public DeepCoder/lcbv5 split) for smoke tests instead. First load downloads ~100 MB from HuggingFace Hub and caches locally.
- **Sandbox**: Use `--sandbox local` (or `sandbox_backend=local`) for cloud agent runs — no Docker or bubblewrap needed. The local backend runs code as a subprocess.
- **Lint**: Run `ruff check *.py` from the workspace root. No project-level ruff config exists; default rules apply. Pre-existing lint issues exist (unused imports in `experiment_teacher_prompts.py`, shadowed `sandbox` import in `play_w_code_env.py`).
- **Tests**: No automated test suite exists. Validation is done by running the entry points (see "Validating changes non-interactively" above).
- **Smoke test command** (recommended for validating changes):
  ```bash
  python play_w_code_env.py --n-tasks 2 --model Qwen/Qwen3-4B-Instruct-2507 -n 2 --seed 44 --dataset lcbv5 --generate --sandbox local
  ```
