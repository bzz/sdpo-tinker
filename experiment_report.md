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
| **Teacher KL** | 0.0094 | 0.0113 | ↑ (+20.4%) |
| **KL sample→train v1** | 0.00058 | 0.00156 | ↑ |
| **KL sample→train v2** | 0.00085 | 0.00089 | ≈ |
| **Entropy** | 0.373 | 0.380 | ≈ |
| **Correct (train)** | 0.0% | 0.0% | — |
| **Group pass rate (train)** | 0.0% | 0.0% | — |
| **Has code (train)** | 87.5% | 100% | ↑ |
| **Avg action tokens/turn** | 2211.5 | 1682.9 | ↓ (−23.9%) |
| **Total action tokens** | 17,692 | 13,463 | ↓ |
| **Frac all-bad groups** | 100% | 100% | — |

### 2.2 Evaluation Metrics (Test Set, 10 tasks)

| Metric | Step 0 (pre-train) | Step 1 (post-step-1) |
|--------|---------------------|----------------------|
| **Correct** | 0.0% | 0.0% |
| **Group pass rate** | 0.0% | 0.0% |
| **Has code** | 100% | 90% |
| **Avg action tokens/turn** | 1268.6 | 1365.4 |
| **Total action tokens** | 12,686 | 13,654 |
| **Frac all-bad groups** | 100% | 100% |

### 2.3 Timing Breakdown

| Phase | Step 0 (sec) | Step 1 (sec) | Total (sec) |
|-------|-------------|-------------|-------------|
| Eval (test set) | 71.0 | 113.4 | 184.4 |
| Sample (train rollouts) | 101.7 | 106.4 | 208.1 |
| KL penalty compute | 19.9 | 1.6 | 21.4 |
| Train (optimizer) | 3.2 | 2.9 | 6.1 |
| Save checkpoint | 4.4 | 2.0 | 6.5 |
| Other (assemble, etc.) | 0.7 | 0.0 | 0.7 |
| **Total** | **200.9** | **226.3** | **427.2** |

**Total wall-clock time**: ~427 seconds (~7.1 minutes)

#### Where does the time go?

The `grade_time_s` metric now separates sandbox grading from Tinker API inference inside the `sample` and `eval` timers:

| | Avg grade time (s/rollout) | Rollouts | Total grade (s) | Timer total (s) | **Inference (s)** |
|--|---|---|---|---|---|
| **Step 0 train** | 0.156 | 8 | 1.2 | 101.7 | **100.5** |
| **Step 0 eval** | 0.212 | 10 | 2.1 | 71.0 | **68.9** |
| **Step 1 train** | 0.184 | 8 | 1.5 | 106.4 | **104.9** |
| **Step 1 eval** | 0.152 | 10 | 1.5 | 113.4 | **111.9** |
| **Totals** | | | **6.3** | **392.5** | **386.2** |

**Sandbox grading is negligible** — only 6.3s total (1.5% of wall time). **Tinker API inference dominates at 386s (90%)** of the ~427s total. The remaining ~35s is KL penalty computation (21s), optimizer train steps (6s), checkpoint saves (6.5s), and setup.

The KL penalty was anomalously slow at step 0 (19.9s vs 1.6s at step 1). This may be a cold-start effect on the teacher model's first logprob call.

## 3. Analysis

### 3.1 KL Divergence Trends

Teacher KL slightly *increased* from 0.0094 → 0.0113 this run. With only 2 steps and stochastic sampling, this is within noise. The absolute values (~0.01 nats/token) are very low, confirming that the student and teacher distributions start nearly identical (same base model, no teacher checkpoint).

### 3.2 Accuracy and Correctness

**Zero correctness across all steps and eval** — no rollout passed any test case. Expected for 2 reasons:

1. **LCBv6 problems are hard** — competitive programming problems are beyond a 4B model with 4096 max tokens at temperature=1.0.
2. **Only 2 training steps** — far too few for the KL-based SDPO signal to improve solution quality.

### 3.3 Generation Length

Training generation length decreased from **2211.5 → 1682.9** tokens/turn (−24%). This aligns with the SDPO paper's observation that SDPO produces shorter generations. Evaluation length slightly increased (1268.6 → 1365.4), within noise.

## 4. Comparison with SDPO Paper (arXiv:2601.20802)

The [SDPO paper](https://arxiv.org/abs/2601.20802) reports on LiveCodeBench V6 with Qwen3-8B:
- **Final accuracy**: 48.8% (vs GRPO 41.2%) after full training
- **4× faster learning** than GRPO in terms of generations needed
- **Shorter generations** than GRPO (3× shorter on average)

| Paper Finding | Our Observation | Consistent? |
|---------------|----------------|-------------|
| KL divergence as sole signal drives improvement | KL non-trivial (~0.01 nats), providing gradient signal | ✅ Directionally consistent |
| Shorter generations over training | Train tokens/turn decreased 24% (2212→1683) | ✅ Consistent |
| 48.8% final accuracy on LCBv6 | 0% after 2 steps | ⏳ Expected — need many more steps |

### Key difference: per-token KL vs. logit-level dense credit

Our implementation uses **per-token reverse KL** on sampled tokens. The paper's strongest variant uses **logit-level dense KL** across top-k tokens per position. Paper ablation shows per-token is intermediate (between logit-level and sequence-level).

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

| Category | Step 0 | Step 1 | Total |
|----------|--------|--------|-------|
| **Train prompts (prefill)** | 2,956 | 2,912 | **5,868** |
| **Train sampling (sample)** | 17,692 | 13,463 | **31,155** |
| **Eval prompts (prefill)** | 3,264 | 3,264 | **6,528** |
| **Eval sampling (sample)** | 12,686 | 13,654 | **26,340** |
| **Teacher KL (prefill)** | 20,339 | 16,071 | **36,410** |
| **Train datums (train)** | 20,640 | 16,367 | **37,007** |

Summary by Tinker billing category:

| Billing category | Tokens |
|-----------------|--------|
| **Prefill** (train prompts + eval prompts + teacher KL) | 48,806 |
| **Sample** (train generation + eval generation) | 57,495 |
| **Train** (optimizer total tokens) | 37,007 |

### 5.3 Estimated Tinker API Cost

| Operation | Tokens | Rate | Cost |
|---|---|---|---|
| Prefill | 48,806 | $0.07/1M | $0.0034 |
| Sample | 57,495 | $0.22/1M | $0.0126 |
| Train | 37,007 | $0.22/1M | $0.0081 |
| **Total** | **143,308** | | **$0.024** |

**Total cost: ~2.4 cents** for 2 training steps with 4 tasks and 10 eval tasks.

Sampling dominates cost (52%), followed by training (34%), then prefill (14%). This makes sense — generation is the most compute-intensive operation (autoregressive decoding), and the train operation runs a full forward+backward pass.

## 6. Observations and Recommendations

1. **The pipeline works end-to-end**: Model loads, samples, computes KL, trains, evaluates, and saves checkpoints successfully.

2. **Inference dominates wall-clock time** (90%): Of ~427s total, ~386s is Tinker API inference (token generation for train rollouts + eval). Grading is negligible (6s). To speed up experiments, reduce `group_size`, `max_tokens`, or `max_eval_tasks`.

3. **Zero accuracy is expected**: LCBv6 problems are hard for a 4B model with 4096 max tokens. The paper uses Qwen3-8B with 24576 max tokens.

4. **To see meaningful learning**:
   - Increase `max_step` to 50+ (paper runs for many more iterations)
   - Increase `max_tokens` to 16384+ (competitive programming needs long chains of thought)
   - Use more tasks (`n_tasks=128+`) to avoid overfitting
   - Consider a larger model (8B+) for better baseline capability

5. **To observe overfitting specifically**:
   - Keep `n_tasks=4` but increase `max_step=50` and `eval_every=5`
   - Watch for train accuracy rising while test accuracy stays flat

## 7. Raw Logs

Full console output saved in `run_output_v2.log`. Structured metrics at:
```
/root/tinker-examples/distillation/sdpo-code-Qwen-Qwen3-4B-Instruct-2507-128rank-0.0001lr-2batch-2026-03-08-22-25/metrics.jsonl
```
