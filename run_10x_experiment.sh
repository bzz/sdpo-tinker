#!/bin/bash
# Run the tiny SDPO training experiment 10 times, collecting metrics from each run.
# Each run: 2 training steps on lcbv6 with Qwen3-4B, eval on 10 test tasks.

set -uo pipefail

RESULTS_DIR="/home/user/sdpo-tinker/experiment_10x_results"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Starting 10x experiment runs ==="
echo "Start time: $(date)"

for i in $(seq 1 10); do
  echo ""
  echo "========================================="
  echo "  Run $i / 10 — $(date)"
  echo "========================================="

  LOG_FILE="$RESULTS_DIR/run_${i}.log"

  # Run the experiment, capturing output
  python /home/user/sdpo-tinker/sdpo_on_policy_distillation.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    teacher_model=Qwen/Qwen3-4B-Instruct-2507 \
    dataset_name=lcbv6 n_tasks=4 groups_per_batch=2 group_size=4 \
    max_step=2 max_eval_tasks=10 sandbox_backend=local \
    eval_every=1 behavior_if_log_dir_exists=delete \
    > "$LOG_FILE" 2>&1

  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Run $i completed successfully"
  else
    echo "Run $i FAILED (exit code $EXIT_CODE)"
  fi
done

echo ""
echo "=== All 10 runs completed ==="
echo "End time: $(date)"
echo "Logs saved in: $RESULTS_DIR"
