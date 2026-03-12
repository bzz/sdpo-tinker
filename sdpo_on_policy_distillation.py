"""
On-policy training for LiveCodeBench with multiple experiment modes.

Modes (controlled by use_execution_reward and use_sdpo_teacher_inputs):

  SDPO (default):  token-level KL divergence against a feedback-conditioned teacher.
  GRPO:            Execution reward with per-group advantage centering.
  Distillation:    KL divergence against a teacher scoring student output directly.
  Combined:        Execution reward + distillation KL.

Example — SDPO (original):
    python sdpo_on_policy_distillation.py \
        model_name=Qwen/Qwen3-8B group_size=8 groups_per_batch=8 \
        learning_rate=1e-6 lora_rank=32 max_step=50

Example — GRPO with execution rewards:
    python sdpo_on_policy_distillation.py \
        model_name=Qwen/Qwen3-8B group_size=8 groups_per_batch=8 \
        use_execution_reward=True kl_penalty_coef=0

Example — On-policy distillation from a bigger teacher:
    python sdpo_on_policy_distillation.py \
        model_name=Qwen/Qwen3-4B-Instruct-2507 teacher_model=Qwen/Qwen3-32B \
        use_sdpo_teacher_inputs=False
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import chz
from tinker.types import LossFnType
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    TeacherConfig,
)

import train_on_policy
from env import SDPOCodeDatasetBuilder


logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for on-policy distillation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"  # Student model
    lora_rank: int = 128
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None  # Student checkpoint

    # Teacher configuration
    teacher_model: str = "Qwen/Qwen3-8B"
    teacher_checkpoint: str | None = None

    # Training hyperparameters
    group_size: int = 4  # Number of rollouts per prompt
    groups_per_batch: int = 1024
    learning_rate: float = 1e-4
    max_tokens: int = 4096
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # When True, build SDPO feedback-conditioned teacher inputs.
    # When False, teacher scores the student's exact output (standard distillation).
    use_sdpo_teacher_inputs: bool = True

    # When True, env returns execution reward (pass=1, fail=0) for GRPO-style training.
    use_execution_reward: bool = False

    # Optimizer configuration
    num_substeps: int = 1

    # Loss function and configuration.
    # See https://tinker-docs.thinkingmachines.ai/losses
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evaluation and checkpointing
    eval_every: int = 20
    save_every: int = 20
    max_step: int | None = None
    n_tasks: int | None = None
    max_eval_tasks: int | None = None

    # Dataset selection: None (all DeepCoder), "lcbv5", "lcbv6", "codeforces"
    dataset_name: str | None = None

    # If True, run evaluators once on the test set and exit without training.
    eval_only: bool = False

    # Sandbox backend for code execution; matches play_w_code_env.py --sandbox
    sandbox_backend: str = "sandboxfusion"  # sandboxfusion | local | bwrap

    # Service configuration
    base_url: str | None = None

    # TTL for checkpoints in seconds (None = no expiry).
    ttl_seconds: int | None = 259200  # 3 days

    # Prefix for checkpoint names, e.g. "sdpo-code-qwen3-8b-"
    chkpt_name_prefix: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get renderer name
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"sdpo-code-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/distillation/{run_name}")

    # Create wandb name if not specified
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = os.path.basename(log_path)

    # SDPO code env: DeepCoder problems graded via sandbox, feedback fed to teacher.
    # batch_size must equal groups_per_batch for CompositeDataset alignment.
    dataset_builder = SDPOCodeDatasetBuilder(
        model_name_for_tokenizer=cli_config.model_name,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        backend=cli_config.sandbox_backend,
        n_tasks=cli_config.n_tasks,
        n_eval_tasks=cli_config.max_eval_tasks,
        dataset_name=cli_config.dataset_name,
        use_execution_reward=cli_config.use_execution_reward,
    )

    # Create teacher config
    teacher_config = TeacherConfig(
        base_model=cli_config.teacher_model,
        load_checkpoint_path=cli_config.teacher_checkpoint,
    )

    # Create distillation dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=cli_config.groups_per_batch,
    )

    # Create full config
    config = train_on_policy.Config(
        learning_rate=cli_config.learning_rate,
        dataset_configs=[dataset_config],
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        use_sdpo_teacher_inputs=cli_config.use_sdpo_teacher_inputs,
        use_execution_reward=cli_config.use_execution_reward,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_step=cli_config.max_step,
        eval_only=cli_config.eval_only,
        ttl_seconds=cli_config.ttl_seconds,
        chkpt_name_prefix=cli_config.chkpt_name_prefix,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await train_on_policy.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
