#!/usr/bin/env python3
"""Parse 10x experiment logs and produce a summary report."""

import re
import sys
import statistics
from pathlib import Path

RESULTS_DIR = Path("/home/user/sdpo-tinker/experiment_10x_results")

# Metrics to extract from Step 0 and Step 1 tables
METRICS = {
    "teacher_kl": r"│\s*teacher_kl\s*│\s*([\d.]+)\s*│",
    "entropy": r"│\s*optim/entropy\s*│\s*([\d.]+)\s*│",
    "kl_v1": r"│\s*optim/kl_sample_train_v1\s*│\s*([\d.]+)\s*│",
    "kl_v2": r"│\s*optim/kl_sample_train_v2\s*│\s*([\d.]+)\s*│",
    "train_correct": r"│\s*env/all/correct\s*│\s*([\d.]+)\s*│",
    "train_has_code": r"│\s*env/all/has_code\s*│\s*([\d.]+)\s*│",
    "train_ac_tokens": r"│\s*env/all/ac_tokens_per_turn\s*│\s*([\d.]+)\s*│",
    "train_frac_all_bad": r"│\s*env/all/by_group/frac_all_bad\s*│\s*([\d.]+)\s*│",
    "test_correct": r"│\s*test/test/env/all/correct\s*│\s*([\d.]+)\s*│",
    "test_has_code": r"│\s*test/test/env/all/has_code\s*│\s*([\d.]+)\s*│",
    "test_ac_tokens": r"│\s*test/test/env/all/ac_tokens_p…\s*│\s*([\d.]+)\s*│",
    "test_frac_all_bad": r"│\s*test/test/env/all/by_group/fr…\s*│\s*([\d.]+)\s*│",
    "time_total": r"│\s*time/total\s*│\s*([\d.]+)\s*│",
    "time_sample": r"│\s*time/sample\s*│\s*([\d.]+)\s*│",
    "time_eval": r"│\s*time/run_evals\s*│\s*([\d.]+)\s*│",
    "time_train": r"│\s*time/train\s*│\s*([\d.]+)\s*│",
    "time_kl": r"│\s*time/compute_kl_penalty\s*│\s*([\d.]+)\s*│",
    "train_action_tokens": r"│\s*train_action_tokens\s*│\s*([\d.]+)\s*│",
    "train_total_tokens": r"│\s*train_total_tokens\s*│\s*([\d.]+)\s*│",
}


def parse_log(log_path: Path) -> dict:
    """Parse a single run log and extract Step 0 and Step 1 metrics."""
    text = log_path.read_text()

    # Split by step headers
    step_sections = re.split(r"Step\s+(\d+)", text)
    # step_sections: [preamble, "0", step0_content, "1", step1_content, ...]

    result = {}
    for i in range(1, len(step_sections) - 1, 2):
        step_num = int(step_sections[i])
        section = step_sections[i + 1]

        for metric_name, pattern in METRICS.items():
            match = re.search(pattern, section)
            if match:
                key = f"step{step_num}_{metric_name}"
                result[key] = float(match.group(1))

    # Check for success
    result["success"] = "Training completed successfully" in text
    return result


def fmt(val, decimals=4):
    """Format a float."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def stats(values):
    """Return mean, std, min, max."""
    if not values:
        return None, None, None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std, min(values), max(values)


def main():
    logs = sorted(RESULTS_DIR.glob("run_*.log"))
    if not logs:
        print("No log files found!")
        sys.exit(1)

    print(f"Found {len(logs)} log files")

    all_runs = []
    for log in logs:
        parsed = parse_log(log)
        parsed["name"] = log.stem
        all_runs.append(parsed)
        print(f"  {log.stem}: success={parsed['success']}, "
              f"step0_teacher_kl={parsed.get('step0_teacher_kl', 'N/A')}, "
              f"step1_teacher_kl={parsed.get('step1_teacher_kl', 'N/A')}")

    successful = [r for r in all_runs if r["success"]]
    print(f"\n{len(successful)}/{len(all_runs)} runs succeeded")

    # Build summary table
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (across all successful runs)")
    print("=" * 80)

    key_metrics = [
        ("Teacher KL", "teacher_kl"),
        ("KL sample→train v1", "kl_v1"),
        ("KL sample→train v2", "kl_v2"),
        ("Entropy", "entropy"),
        ("Train correct", "train_correct"),
        ("Train has_code", "train_has_code"),
        ("Train ac_tokens/turn", "train_ac_tokens"),
        ("Train frac_all_bad", "train_frac_all_bad"),
        ("Test correct", "test_correct"),
        ("Test has_code", "test_has_code"),
        ("Test ac_tokens/turn", "test_ac_tokens"),
        ("Test frac_all_bad", "test_frac_all_bad"),
        ("Time total (s)", "time_total"),
        ("Time sample (s)", "time_sample"),
        ("Time eval (s)", "time_eval"),
        ("Time train (s)", "time_train"),
        ("Time KL (s)", "time_kl"),
    ]

    for step in [0, 1]:
        print(f"\n--- Step {step} ---")
        print(f"{'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 70)
        for label, metric_key in key_metrics:
            full_key = f"step{step}_{metric_key}"
            values = [r[full_key] for r in successful if full_key in r]
            mean, std, vmin, vmax = stats(values)
            if mean is not None:
                print(f"{label:<30} {fmt(mean):>10} {fmt(std):>10} {fmt(vmin):>10} {fmt(vmax):>10}")

    # Per-run detail table for key metrics
    print("\n" + "=" * 80)
    print("PER-RUN DETAIL")
    print("=" * 80)

    header = f"{'Run':<8}"
    detail_metrics = [
        ("s0_tkl", "step0_teacher_kl"),
        ("s1_tkl", "step1_teacher_kl"),
        ("s0_ent", "step0_entropy"),
        ("s1_ent", "step1_entropy"),
        ("s0_tcor", "step0_train_correct"),
        ("s1_tcor", "step1_train_correct"),
        ("s0_ecor", "step0_test_correct"),
        ("s1_ecor", "step1_test_correct"),
        ("s0_time", "step0_time_total"),
        ("s1_time", "step1_time_total"),
    ]
    for label, _ in detail_metrics:
        header += f" {label:>8}"
    print(header)
    print("-" * len(header))

    for run in successful:
        row = f"{run['name']:<8}"
        for _, key in detail_metrics:
            val = run.get(key)
            row += f" {fmt(val, 4) if val is not None else 'N/A':>8}"
        print(row)

    # Compute totals
    print("\n" + "=" * 80)
    print("AGGREGATE")
    print("=" * 80)
    total_time = sum(
        r.get("step0_time_total", 0) + r.get("step1_time_total", 0)
        for r in successful
    )
    print(f"Total wall-clock time across all runs: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average per run: {total_time/len(successful):.1f}s ({total_time/len(successful)/60:.1f} min)")

    # Teacher KL direction
    increases = 0
    decreases = 0
    for r in successful:
        s0 = r.get("step0_teacher_kl", 0)
        s1 = r.get("step1_teacher_kl", 0)
        if s1 > s0:
            increases += 1
        else:
            decreases += 1
    print(f"\nTeacher KL direction (step0→step1): {increases} increased, {decreases} decreased")


if __name__ == "__main__":
    main()
