"""
On-policy distillation training loop — SDPO fork of tinker-cookbook's
tinker_cookbook/distillation/train_on_policy.py (commit aae9dac).

SDPO-specific changes:
  * _get_renderer_for_builder() and build_teacher_inputs_for_batch() build
    feedback-conditioned teacher inputs from the code environment after rollout.
  * incorporate_kl_penalty() accepts pre-built teacher_inputs_D instead of
    constructing them from datum.model_input.  For passing trajectories the
    input is None and the KL contribution is zero.  For failing trajectories
    the teacher sees [SDPO reprompt][student tokens] and [-n_action:] logprobs
    are extracted to align with the response suffix.
  * prepare_minibatch() calls build_teacher_inputs_for_batch() and threads the
    result through into incorporate_kl_penalty().
  * do_sync_training() sets do_remove_constant_reward_groups=False because the
    SDPO code env always returns reward=0.
  * main() fixes upstream double-creation of the teacher sampling client.
  * colorize_kl_example() and incorporate_kl_penalty() print one failing
    sample per step with tokens colored by per-token reverse KL
    (student_lp - teacher_lp), the actual SDPO learning signal.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Sequence

import chz
import tinker
import torch

from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.display import colorize_example, to_ints
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    DistillationDatasetConfig,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
try:
    from tinker_cookbook.rl.train import _format_chkpt_name
except ImportError:
    # From https://github.com/bzz/tinker-cookbook/tree/bzz/rl-prompt-template-rlvr-openchar
    def _format_chkpt_name(name: str, prefix: str | None) -> str:
        return f"{prefix}_{name}" if prefix else name

from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.utils.trace import scope, update_scope_context, trace_init

from env import SDPOCodeGroupBuilder, build_sdpo_teacher_inputs


logger = logging.getLogger(__name__)


def colorize_kl_example(
    datum: tinker.Datum,
    reverse_kl: torch.Tensor,
    tokenizer: Any,
) -> str:
    """Return ANSI-colorized text where tokens are colored by per-token reverse KL.

    Green = teacher more confident (student will be pushed toward),
    Yellow = outside mask / equal,
    Red = student more confident (student will be pushed away).
    """
    int_tokens = [
        token for chunk in datum.model_input.chunks
        for token in to_ints(chunk, tokenizer)
    ] + [datum.loss_fn_inputs["target_tokens"].tolist()[-1]]
    weights = [0.0] + (-reverse_kl).tolist()
    return format_colorized(int_tokens, weights, tokenizer)


def _get_renderer_for_builder(
    builder: SDPOCodeGroupBuilder,
    tokenizer: Any,
    cache: dict,
) -> renderers.Renderer:
    """Return a renderer for an SDPOCodeGroupBuilder, reusing cached instances."""
    if getattr(builder, "renderer", None) is not None:
        return builder.renderer  # type: ignore[return-value]
    name = builder.renderer_name or builder.model_name
    if name not in cache:
        cache[name] = renderers.get_renderer(name, tokenizer)
    return cache[name]


def build_teacher_inputs_for_batch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    metadata_D: list[dict],
    data_D: list[tinker.Datum],
    tokenizer: Any,
) -> list[tinker.ModelInput | None]:
    """Build per-datum teacher ModelInputs for the SDPO code environment.

    Returns a list aligned with data_D / metadata_D where each entry is:
      - None  for passing trajectories (no KL contribution)
      - ModelInput containing [SDPO reprompt][student tokens] for failing ones
    """
    renderer_cache: dict[str, renderers.Renderer] = {}
    teacher_inputs_by_group: dict[int, list[tinker.ModelInput | None]] = {}

    for i_group, (builder, traj_group) in enumerate(
        zip(env_group_builders_P, trajectory_groups_P)
    ):
        if not isinstance(builder, SDPOCodeGroupBuilder):
            raise TypeError(
                f"Expected SDPOCodeGroupBuilder, got {type(builder).__name__}"
            )
        renderer = _get_renderer_for_builder(builder, tokenizer, renderer_cache)
        teacher_inputs_by_group[i_group] = build_sdpo_teacher_inputs(
            task=builder.task,
            trajectories_G=traj_group.trajectories_G,
            renderer=renderer,
            tokenizer=tokenizer,
        )

    teacher_inputs_D: list[tinker.ModelInput | None] = []
    for datum, metadata in zip(data_D, metadata_D):
        gi = metadata["group_idx"]
        ti = metadata["traj_idx"]
        teacher_inputs_D.append(teacher_inputs_by_group[gi][ti])

    return teacher_inputs_D


@scope
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    teacher_clients_D: List[tinker.SamplingClient],
    teacher_inputs_D: List[tinker.ModelInput | None],
    dataset_indices_D: List[int],
    kl_penalty_coef: float,
    kl_discount_factor: float,
    tokenizer: Any = None,
    use_sdpo_teacher_inputs: bool = True,
) -> Dict[str, float | int]:
    """
    Compute reverse KL between the student (log p) and the teacher model (log q), computed as
    log p - log q. We then adjust the advantages in-place as the negative reverse KL.

    teacher_inputs_D is a parallel list of ModelInputs, one per datum:
      - None means no KL contribution (passing SDPO trajectory)
      - ModelInput is [SDPO reprompt][student tokens] (use_sdpo_teacher_inputs=True)
        or [student prompt + response] (use_sdpo_teacher_inputs=False);
        [-n_action:] logprobs align with the response suffix.

    Args:
        data_D: List of datums to compute KL for
        teacher_clients_D: List of teacher sampling clients, one per datum
        teacher_inputs_D: Pre-built teacher ModelInputs, one per datum
        dataset_indices_D: List of dataset indices, one per datum
        kl_penalty_coef: Coefficient for KL penalty
        kl_discount_factor: Discount factor for future KL
        use_sdpo_teacher_inputs: Whether SDPO-style reprompted teacher inputs are used
    """
    # Compute the teacher's logprobs for each element of the batch; skip None entries.
    teacher_logprobs_D = await asyncio.gather(
        *[
            teacher_client.compute_logprobs_async(teacher_input)
            if teacher_input is not None
            else asyncio.sleep(0, result=None)
            for teacher_client, teacher_input in zip(teacher_clients_D, teacher_inputs_D)
        ]
    )
    # The reverse KL is computed as KL[p||q] = log p - log q, where
    #   - p: sampled_logprobs
    #   - q: teacher_logprobs
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    reverse_kl = []
    for teacher_logprobs, sampled_logprobs, mask in zip(
        teacher_logprobs_D, sampled_logprobs_D, float_masks
    ):
        if teacher_logprobs is None:
            # Passing trajectory: zero KL contribution.
            reverse_kl.append(torch.zeros_like(sampled_logprobs))
            continue
        # Slice [-n_action:] to align teacher logprobs with the response suffix.
        n_action = int(mask.sum().item())
        teacher_action_lp = torch.tensor(teacher_logprobs[-n_action:], dtype=sampled_logprobs.dtype)
        teacher_lp_aligned = torch.zeros_like(sampled_logprobs)
        teacher_lp_aligned[-n_action:] = teacher_action_lp
        reverse_kl.append((sampled_logprobs - teacher_lp_aligned) * mask)

    kl_label = "SDPO KL debug" if use_sdpo_teacher_inputs else "Distill KL debug"
    if tokenizer is not None:
        for i, (datum, rkl) in enumerate(zip(data_D, reverse_kl)):
            if teacher_inputs_D[i] is not None:
                n_action = int(float_masks[i].sum().item())
                avg_kl = rkl.sum().item() / max(n_action, 1)
                logger.info(
                    "%s (avg per-token reverse_kl=%.3f, %d action tokens):\n%s",
                    kl_label, avg_kl, n_action,
                    colorize_kl_example(datum, rkl, tokenizer),
                )
                break

    # Track per-dataset KL for logging
    # dataset_idx -> (sum of KL, sum of mask)
    per_dataset_kl: Dict[int, tuple[float, float]] = {}

    for i, datum in enumerate(data_D):
        # The advantage is the negative reverse KL. We can optionally apply a discount factor.
        kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i]
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

        # Accumulate per-dataset KL
        dataset_idx = dataset_indices_D[i]
        kl_sum = reverse_kl[i].sum().item()
        mask_sum = float_masks[i].sum().item()
        if dataset_idx not in per_dataset_kl:
            per_dataset_kl[dataset_idx] = (0.0, 0.0)
        prev_kl_sum, prev_mask_sum = per_dataset_kl[dataset_idx]
        per_dataset_kl[dataset_idx] = (prev_kl_sum + kl_sum, prev_mask_sum + mask_sum)

    # Compute average reverse KL over the batch for logging purposes
    avg_logp_diff = sum([diff.sum() for diff in reverse_kl]) / max(
        sum([mask.sum() for mask in float_masks]).item(), 1e-8
    )

    # Count teacher KL prefill tokens (total tokens sent for logprob forward passes)
    teacher_kl_prefill_tokens = sum(
        ti.length for ti in teacher_inputs_D if ti is not None
    )

    # Compute per-dataset metrics
    metrics = {
        "teacher_kl": float(avg_logp_diff),
        "teacher_kl_prefill_tokens": teacher_kl_prefill_tokens,
    }
    for dataset_idx, (kl_sum, mask_sum) in per_dataset_kl.items():
        if mask_sum > 0:
            metrics[f"teacher_kl/dataset_{dataset_idx}"] = float(kl_sum / mask_sum)

    return metrics


@chz.chz
class Config:
    learning_rate: float
    dataset_configs: List[DistillationDatasetConfig]
    model_name: str
    renderer_name: str | None = None
    max_tokens: int
    temperature: float = 1.0
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # When True, build SDPO feedback-conditioned teacher inputs (reprompting
    # with execution feedback + sibling solution).  When False, the teacher
    # scores the student's exact output — standard on-policy distillation.
    use_sdpo_teacher_inputs: bool = True

    # When True, the env returns execution reward (pass=1, fail=0) enabling
    # GRPO-style training.  When False, reward is always 0.
    use_execution_reward: bool = False

    # Loss function and configuration.
    # See https://tinker-docs.thinkingmachines.ai/losses
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False

    eval_every: int = 20
    save_every: int = 20
    load_checkpoint_path: str | None = None

    # Maximum number of training steps. If None, train on the full dataset.
    max_step: int | None = None

    # If True, run evaluators once and exit without training.
    eval_only: bool = False

    # TTL for checkpoints in seconds (None = no expiry).
    ttl_seconds: int | None = 259200  # 3 days

    # Prefix for checkpoint names, e.g. "sdpo-code-qwen3-8b-"
    chkpt_name_prefix: str | None = None

    @property
    def checkpoint_kwargs(self) -> dict[str, Any]:
        """Optional args for RL save_checkpoint_and_get_sampling_client / compute_full_batch_metrics_and_get_sampling_client.
        From https://github.com/bzz/tinker-cookbook/tree/bzz/rl-prompt-template-rlvr-openchar
        """
        kw: dict[str, Any] = {}
        if self.chkpt_name_prefix is not None:
            kw["chkpt_name_prefix"] = self.chkpt_name_prefix
        return kw


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    dataset_indices_P: List[int],
    teacher_clients: List[tinker.SamplingClient],
    kl_penalty_coef: float,
    kl_discount_factor: float,
    use_sdpo_teacher_inputs: bool = True,
    use_execution_reward: bool = False,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Count train tokens: total tokens in each datum and masked (action) tokens
    # contributing to the loss.
    train_total_tokens = sum(datum.model_input.length for datum in data_D)
    train_action_tokens = sum(
        int(datum.loss_fn_inputs["mask"].to_torch().sum().item()) for datum in data_D
    )
    metrics["train_total_tokens"] = train_total_tokens
    metrics["train_action_tokens"] = train_action_tokens

    # Print one datum per dataset
    printed_datasets = set()
    for datum, metadata in zip(data_D, metadata_D):
        dataset_idx = dataset_indices_P[metadata["group_idx"]]
        if dataset_idx not in printed_datasets:
            logger.info(colorize_example(datum, tokenizer, key="mask"))
            printed_datasets.add(dataset_idx)

    # Print advantage-colored sample when execution rewards produce non-trivial advantages
    if use_execution_reward:
        printed_datasets_adv: set[int] = set()
        for datum, metadata in zip(data_D, metadata_D):
            dataset_idx = dataset_indices_P[metadata["group_idx"]]
            if dataset_idx not in printed_datasets_adv:
                logger.info(
                    "GRPO advantage-colored sample:\n%s",
                    colorize_example(datum, tokenizer, key="advantages"),
                )
                printed_datasets_adv.add(dataset_idx)

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        with timed("build_teacher_inputs", metrics):
            if use_sdpo_teacher_inputs:
                teacher_inputs_D = build_teacher_inputs_for_batch(
                    env_group_builders_P,
                    trajectory_groups_P,
                    metadata_D,
                    data_D,
                    tokenizer,
                )
            else:
                teacher_inputs_D = [
                    datum.model_input.append_int(
                        int(datum.loss_fn_inputs["target_tokens"].data[-1])
                    )
                    for datum in data_D
                ]

        with timed("compute_kl_penalty", metrics):
            # Map each datum to its teacher sampling client and dataset index using metadata
            #   - metadata_D contains group_idx which indexes into trajectory_groups_P
            #   - dataset_indices_P[group_idx] gives us the dataset index
            #   - teacher_clients[dataset_idx] gives us the teacher
            teacher_clients_D = [
                teacher_clients[dataset_indices_P[metadata["group_idx"]]] for metadata in metadata_D
            ]
            dataset_indices_D = [
                dataset_indices_P[metadata["group_idx"]] for metadata in metadata_D
            ]
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                teacher_clients_D,
                teacher_inputs_D,
                dataset_indices_D,
                kl_penalty_coef,
                kl_discount_factor,
                tokenizer=tokenizer,
                use_sdpo_teacher_inputs=use_sdpo_teacher_inputs,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    dataset_indices_P: List[int],
    teacher_clients: List[tinker.SamplingClient],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    update_scope_context({"step": i_batch})

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        dataset_indices_P,
        teacher_clients,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        use_sdpo_teacher_inputs=cfg.use_sdpo_teacher_inputs,
        use_execution_reward=cfg.use_execution_reward,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D=data_D,
            training_client=training_client,
            learning_rate=cfg.learning_rate,
            num_substeps=cfg.num_substeps,
            loss_fn=cfg.loss_fn,
            loss_fn_config=cfg.loss_fn_config,
            metrics=metrics,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
        ttl_seconds=cfg.ttl_seconds,
        **cfg.checkpoint_kwargs,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: CompositeDataset,
    teacher_clients: List[tinker.SamplingClient],
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements fully synchronous on-policy training"""

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every,
        ttl_seconds=cfg.ttl_seconds,
        **cfg.checkpoint_kwargs,
    )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if (cfg.eval_every > 0 and i_batch % cfg.eval_every == 0) or i_batch == end_batch - 1:
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    if hasattr(evaluator, "env_group_builders_P"):
                        logger.info(f"Evaluating on {len(evaluator.env_group_builders_P)} tasks")
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Get batch and sample trajectories
        env_group_builders_P, dataset_indices_P = dataset.get_batch(i_batch)
        # With execution rewards, drop groups where all rollouts got the same
        # reward (all-pass or all-fail) since GRPO advantages would be zero.
        # Without execution rewards the KL penalty is the sole signal, so we
        # must keep constant-reward groups.
        drop_constant = cfg.use_execution_reward and cfg.kl_penalty_coef == 0
        with timed("sample", metrics):
            trajectory_groups_P = await asyncio.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            temperature=cfg.temperature,
                            max_tokens=cfg.max_tokens,
                            do_remove_constant_reward_groups=drop_constant,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
            )
        trajectory_groups_P = [
            trajectory_group
            for trajectory_group in trajectory_groups_P
            if trajectory_group is not None
        ]

        # Train step
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            dataset_indices_P,
            teacher_clients,
        )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(
    cfg: Config,
):
    """Main training loop for on-policy distillation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
        do_configure_logging_module=False,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    mode_parts = []
    if cfg.use_execution_reward:
        mode_parts.append("execution_reward")
    if cfg.kl_penalty_coef > 0:
        mode_parts.append(f"kl(coef={cfg.kl_penalty_coef}, sdpo={cfg.use_sdpo_teacher_inputs})")
    mode_label = " + ".join(mode_parts) or "no signal (check config)"
    logger.info(f"Experiment mode: {mode_label}")

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, cfg.renderer_name)

    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank, user_metadata=user_metadata
    )

    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, load_state_path, cfg.renderer_name
        )
        future = await training_client.load_state_with_optimizer_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create datasets and teacher sampling clients from configs
    datasets = []
    teacher_clients = []
    groups_per_batch_list = []
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]

    for dataset_config in cfg.dataset_configs:
        # Create dataset
        dataset, maybe_test_dataset = await dataset_config.dataset_builder()
        datasets.append(dataset)
        groups_per_batch_list.append(dataset_config.groups_per_batch)

        # Add test dataset evaluator if present
        if maybe_test_dataset is not None:
            evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

        # Create teacher sampling client
        teacher_config = dataset_config.teacher_config
        if teacher_config.load_checkpoint_path is not None:
            teacher_client = service_client.create_sampling_client(
                base_model=teacher_config.base_model,
                model_path=teacher_config.load_checkpoint_path,
            )
        else:
            teacher_client = service_client.create_sampling_client(
                base_model=teacher_config.base_model,
            )
        teacher_clients.append(teacher_client)
        logger.info(
            f"Created teacher sampling client for {teacher_config.base_model} "
            f"(checkpoint: {teacher_config.load_checkpoint_path})"
        )

    # Eval-only mode: run evaluators once then exit.
    if cfg.eval_only:
        sampling_client = (
            service_client.create_sampling_client(
                base_model=cfg.model_name, model_path=load_state_path,
            )
            if load_state_path
            else service_client.create_sampling_client(base_model=cfg.model_name)
        )
        metrics: dict[str, Any] = {}
        with timed("run_evals", metrics):
            for evaluator in evaluators:
                if hasattr(evaluator, "env_group_builders_P"):
                    logger.info(f"Evaluating on {len(evaluator.env_group_builders_P)} tasks")
                eval_metrics = await evaluator(sampling_client)
                metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})
        ml_logger.log_metrics(metrics, step=0)
        ml_logger.close()
        logger.info("Eval-only run completed")
        return

    # Wrap datasets in CompositeDataset
    composite_dataset = CompositeDataset(datasets, groups_per_batch_list)
    dataset_len = len(composite_dataset)
    # Use max_step directly (not min) so it can *exceed* the dataset length,
    # enabling wrap-around for overfit runs (e.g. n_tasks=2, max_step=20).
    num_batches = cfg.max_step if cfg.max_step is not None else dataset_len
    logger.info(f"Will train on {num_batches} batches (dataset has {dataset_len})")

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=composite_dataset,
        teacher_clients=teacher_clients,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=_format_chkpt_name("final", cfg.chkpt_name_prefix),
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
            ttl_seconds=cfg.ttl_seconds,
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
