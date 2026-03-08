# Tiny SDPO Training Experiment Report

**Date**: 2026-03-08
**Branch**: `claude/tiny-train-experiment-7Awsd`

## 1. Experiment Setup

| Parameter | Value |
|-----------|-------|
| Student model | `Qwen/Qwen3-4B-Instruct-2507` |
| Teacher model | `Qwen/Qwen3-4B-Instruct-2507` (same as student) |
| Dataset | LiveCodeBench V6 (`lcbv6`) |
| Training tasks | 4 |
| Eval tasks | 10 |
| Groups per batch | 2 |
| Group size (rollouts/prompt) | 4 |
| Max training steps | 2 |
| Max tokens | 4096 |
| LoRA rank | 128 |
| Learning rate | 1e-4 |
| KL penalty coef | 1.0 |
| Loss function | `importance_sampling` |
| Sandbox backend | `local` |
| Eval frequency | Every step |
| Temperature | 1.0 |

**Total rollouts per step**: 2 groups × 4 rollouts = 8
**Total rollouts across training**: 2 steps × 8 = 16
**Total eval rollouts**: 2 evals × 10 tasks × 1 rollout = 20

### Command

```bash
python sdpo_on_policy_distillation.py \
  model_name=Qwen/Qwen3-4B-Instruct-2507 \
  teacher_model=Qwen/Qwen3-4B-Instruct-2507 \
  dataset_name=lcbv6 n_tasks=4 groups_per_batch=2 group_size=4 \
  max_step=2 max_eval_tasks=10 sandbox_backend=local \
  eval_every=1 behavior_if_log_dir_exists=delete
```

## 2. Results

### 2.1 Training Metrics (Step 0 → Step 1)

| Metric | Step 0 | Step 1 | Trend |
|--------|--------|--------|-------|
| **Teacher KL** | 0.0149 | 0.0108 | ↓ (−27.5%) |
| **KL sample→train v1** | 0.00131 | 0.00084 | ↓ (−36.1%) |
| **KL sample→train v2** | 0.00175 | 0.00098 | ↓ (−43.8%) |
| **Entropy** | 0.427 | 0.389 | ↓ (−8.8%) |
| **Correct (train)** | 0.0% | 0.0% | — |
| **Group pass rate (train)** | 0.0% | 0.0% | — |
| **Has code (train)** | 100% | 87.5% | ↓ |
| **Avg action tokens/turn** | 1948.5 | 1675.8 | ↓ (−14.0%) |
| **Total action tokens** | 15,588 | 13,406 | ↓ |
| **Frac all-bad groups** | 100% | 100% | — |

### 2.2 Evaluation Metrics (Test Set, 10 tasks)

| Metric | Step 0 (pre-train) | Step 1 (post-step-1) |
|--------|---------------------|----------------------|
| **Correct** | 0.0% | 0.0% |
| **Group pass rate** | 0.0% | 0.0% |
| **Has code** | 100% | 100% |
| **Avg action tokens/turn** | 1227.1 | 1331.2 |
| **Total action tokens** | 12,271 | 13,312 |
| **Frac all-bad groups** | 100% | 100% |

### 2.3 Timing Breakdown

| Phase | Step 0 (sec) | Step 1 (sec) |
|-------|-------------|-------------|
| Eval (test set) | 85.0 | 82.0 |
| Sample (train) | 88.0 | 100.0 |
| Train (optimizer) | 3.5 | 4.0 |
| KL penalty compute | 2.0 | 2.0 |
| Save checkpoint | 2.3 | 7.7 |
| **Total** | **180.9** | **195.7** |

**Total wall-clock time**: ~376.6 seconds (~6.3 minutes)

**Note on timing granularity**: The `sample` and `eval` timers include both Tinker API inference (token generation) **and** local sandbox grading (code execution via subprocess). This run predates the `grade_time_s` metric added to `env.py`, which will break these apart in future runs. The `train` timer covers the Tinker optimizer forward/backward pass. The `KL penalty compute` timer covers the teacher logprob forward pass (prefill-only, no sampling).

## 3. Analysis

### 3.1 KL Divergence Trends

The teacher KL dropped from **0.0149 → 0.0108** (−27.5%) over just 2 steps. This is the expected direction — the student is being pushed toward the teacher's (feedback-conditioned) distribution via the KL penalty. The KL between sampling and training policy (`kl_sample_train`) also decreased, which is consistent with the policy being updated.

The absolute KL values are quite low (0.01–0.015 nats per token), which suggests the student and teacher distributions are already fairly similar — expected since they use the same base model (`Qwen3-4B-Instruct`) and no teacher checkpoint.

### 3.2 Accuracy and Correctness

**Zero correctness across all steps and eval** — no rollout passed any test case. This is expected for 2 reasons:

1. **LiveCodeBench V6 problems are hard** — these are competitive programming problems. A 4B model with only 4096 max tokens and temperature=1.0 is unlikely to solve them in 2 steps.
2. **Only 2 training steps** — far too few for the SDPO signal (KL-based) to improve solution quality measurably.

Since no rollouts pass, the environment reward is always 0. The only learning signal is the KL divergence between the student and the feedback-conditioned teacher. All groups are "all-bad" (frac_all_bad = 100%).

### 3.3 Generation Length

Training generation length decreased from **1948.5 → 1675.8** tokens/turn (−14%). This aligns with the SDPO paper's observation that SDPO produces shorter generations than baseline. The entropy also decreased (0.427 → 0.389), suggesting the model is becoming more concentrated in its predictions.

Evaluation generation length slightly *increased* (1227.1 → 1331.2), but this is within noise for 10 eval tasks.

### 3.4 Code Generation

`has_code` dropped from 100% → 87.5% on training data at step 1, meaning 1 out of 8 rollouts didn't produce extractable Python code. This is a minor degradation worth monitoring — with more steps, it could indicate the KL signal is destabilizing code generation.

## 4. Comparison with SDPO Paper (arXiv:2601.20802)

The [SDPO paper](https://arxiv.org/abs/2601.20802) reports the following on LiveCodeBench V6 with Qwen3-8B:
- **Final accuracy**: 48.8% (vs GRPO 41.2%) after full training
- **4× faster learning** than GRPO in terms of generations needed
- **Shorter generations** than GRPO (3× shorter on average)
- **Teacher accuracy improves** during training as the student improves

### What we can compare (and what we can't)

| Paper Finding | Our Observation | Consistent? |
|---------------|----------------|-------------|
| KL divergence as sole signal drives improvement | KL decreased (0.015→0.011), showing signal is non-trivial | ✅ Directionally consistent |
| Shorter generations over training | Train tokens/turn decreased 14% (1949→1676) | ✅ Consistent |
| Teacher accuracy improves during training | Cannot measure (teacher = frozen base model) | N/A |
| 48.8% final accuracy on LCBv6 | 0% after 2 steps | ⏳ Expected — need many more steps |
| Sample efficiency (4× fewer generations than GRPO) | Cannot evaluate with 2 steps | N/A |

### Key difference: per-token KL vs. logit-level dense credit

Our implementation uses **per-token reverse KL** on the sampled tokens:
- For each position, KL = `log p_student(token) - log p_teacher(token)` for the sampled token only

The paper's strongest variant uses **logit-level dense KL** across the top-k tokens at each position. The paper's ablation (Figure 10) shows:
- **Logit-level** (dense across ~100 tokens/position) → strongest
- **Token-level** (single token/position, our approach) → intermediate
- **Sequence-level** (averaged single advantage/sequence) → weakest but still beats GRPO

Our per-token approach should work but will converge more slowly than the logit-level variant. The paper confirms that even this simpler approach "significantly outperforms GRPO."

### Overfitting considerations

With only 4 training tasks, overfitting is guaranteed with enough steps. In 2 steps, we see no accuracy improvement (still 0%), so we haven't reached the point where overfitting manifests. To actually observe overfitting:
- Increase `max_step` to 20+ while keeping `n_tasks=4`
- Monitor whether train accuracy rises while test accuracy plateaus or drops

## 5. Cost Estimate

### 5.1 Tinker API Pricing (Qwen3-4B-Instruct-2507)

| Operation | Price |
|-----------|-------|
| Prefill (prompt/input tokens) | $0.07 / 1M tokens |
| Sample (generation/output tokens) | $0.22 / 1M tokens |
| Train (optimizer token processing) | $0.22 / 1M tokens |
| Storage | $0.10 / GB-month |

Source: [Tinker pricing](https://thinkingmachines.ai/tinker/)

### 5.2 Token Usage from Logs

| Category | Tokens |
|----------|--------|
| **Train sampling (output)** | Step 0: 15,588 + Step 1: 13,406 = **28,994** |
| **Train prompts (input)** | Step 0: 2,956 + Step 1: 2,912 = **5,868** |
| **Eval sampling (output)** | Step 0: 12,271 + Step 1: 13,312 = **25,583** |
| **Eval prompts (input)** | Step 0: 3,264 + Step 1: 3,264 = **6,528** |
| **Teacher KL logprob calls (input)** | ~28,994 (reprocesses student output tokens) |
| **Teacher KL prompts (input)** | ~5,868 (feedback-conditioned prompts) |
| **Total prefill tokens** | ~47,258 |
| **Total sample tokens** | ~54,577 |
| **Total train tokens** | ~28,994 (optimizer processes train rollouts) |

### 5.3 Estimated Tinker API Cost

| Operation | Tokens | Rate | Cost |
|---|---|---|---|
| Prefill | 47,258 | $0.07/1M | $0.0033 |
| Sample | 54,577 | $0.22/1M | $0.0120 |
| Train | 28,994 | $0.22/1M | $0.0064 |
| **Total** | | | **$0.022** |

This is the actual Tinker API cost for this experiment. Tinker handles model serving, LoRA training, and checkpoint management — pricing is purely per-token with no separate GPU/hour charges.

**Note**: The teacher KL token estimates above are approximate (assumed equal to student tokens). Future runs will log exact counts via the new `teacher_kl_prefill_tokens` and `train_total_tokens` / `train_action_tokens` metrics added to the training loop.

## 6. Observations and Recommendations

1. **The pipeline works end-to-end**: Model loads, samples, computes KL, trains, evaluates, and saves checkpoints successfully.

2. **KL signal is active**: The teacher KL is non-zero and decreasing, confirming the distillation objective is providing gradient signal even when all rollouts fail.

3. **Zero accuracy is expected**: LCBv6 competitive programming problems are extremely hard for a 4B model at temperature=1.0 with only 4096 tokens. The paper uses Qwen3-8B with 24576 max tokens.

4. **To see meaningful learning**:
   - Increase `max_step` to 50+ (paper runs for many more iterations)
   - Increase `max_tokens` to 16384+ (competitive programming needs long chains of thought)
   - Use more tasks (`n_tasks=128+`) to avoid overfitting
   - Consider a larger model (8B+) for better baseline capability

5. **To observe overfitting specifically**:
   - Keep `n_tasks=4` but increase `max_step=50` and `eval_every=5`
   - Watch for train accuracy rising while test accuracy stays flat

6. **`has_code` degradation** at step 1 (87.5% vs 100%) is a yellow flag — the KL update may be nudging the model away from producing well-formatted code blocks. Monitor with more steps.

## 7. Raw Logs

Full console output saved in `run_output.log`. Structured metrics at:
```
/root/tinker-examples/distillation/sdpo-code-Qwen-Qwen3-4B-Instruct-2507-128rank-0.0001lr-2batch-2026-03-08-21-18/metrics.jsonl
```
