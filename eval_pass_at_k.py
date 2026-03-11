"""Estimate pass@k for a model on LiveCodeBench-style coding tasks.

Uses the OpenAI chat completions API (works with vLLM, SGLang, or any
compatible endpoint).  Results are cached to JSONL for resume safety --
re-run the same command after an interruption to pick up where you left off.

Example:
    python eval_pass_at_k.py \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --base-url http://localhost:8000/v1 \\
        --limit 30 -n 8 -k 1,2,4,8 \\
        --dataset-name lcbv5 --split train \\
        --temperature 1.0 --sandbox local \\
        --no-think --workers 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from tinker_cookbook.recipes.code_rl.code_grading import extract_code_from_model

import sandbox
from env import CODE_PROMPT, format_feedback, load_tasks

console = Console()


# ---------------------------------------------------------------------------
# pass@k estimator (Codex / HumanEval paper)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# JSONL cache helpers
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> dict[int, dict]:
    """Load completed results keyed by task_idx."""
    results: dict[int, dict] = {}
    if not path.exists():
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            results[entry["task_idx"]] = entry
    return results


def append_result(path: Path, result: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-problem evaluation
# ---------------------------------------------------------------------------

async def eval_one(
    task_idx: int,
    task: Any,
    client: AsyncOpenAI,
    model: str,
    n: int,
    temperature: float,
    max_tokens: int,
    sandbox_backend: str,
    extra_body: dict | None,
    sem: asyncio.Semaphore,
) -> dict:
    """Sample N completions, grade them, return a result dict."""
    prompt = CODE_PROMPT.format(problem=task.problem)
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    t_api = time.time()
    async with sem:
        response = await client.chat.completions.create(**kwargs)
    dt_api = time.time() - t_api

    async def grade_choice(choice: Any) -> dict:
        text = choice.message.content or ""
        n_tokens = choice.message.token_count if hasattr(choice.message, "token_count") else len(text.split())
        code = extract_code_from_model(text)
        has_code = code is not None
        passed = False
        feedback = ""
        if has_code:
            passed, details = await sandbox.grade_code(
                task.tests, code, timeout=6, backend=sandbox_backend,
            )
            if not passed:
                feedback = format_feedback(details)
        else:
            feedback = "No code block found in response."

        sample: dict[str, Any] = {
            "correct": passed,
            "has_code": has_code,
            "response": text,
            "n_tokens": n_tokens,
        }
        if feedback:
            sample["feedback"] = feedback
        return sample

    t_grade = time.time()
    samples = await asyncio.gather(*(grade_choice(c) for c in response.choices))
    dt_grade = time.time() - t_grade

    n_passed = sum(1 for s in samples if s["correct"])

    return {
        "task_idx": task_idx,
        "n": n,
        "n_passed": n_passed,
        "samples": samples,
        "dt_api": round(dt_api, 2),
        "dt_grade": round(dt_grade, 2),
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], ks: list[int]) -> None:
    n_total = len(results)
    if not n_total:
        return

    # pass@k
    console.print()
    console.print(Rule("pass@k Results", style="bold cyan"))
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    for k in ks:
        scores = [pass_at_k(r["n"], r["n_passed"], k) for r in results]
        mean = sum(scores) / len(scores)
        tbl.add_row(f"pass@{k}", f"{mean:.1%}")
    mean_pass_rate = sum(r["n_passed"] / r["n"] for r in results) / n_total
    tbl.add_row("mean pass rate", f"{mean_pass_rate:.1%}")
    console.print(tbl)

    # SDPO signal breakdown
    all_pass = sum(1 for r in results if r["n_passed"] == r["n"])
    all_fail = sum(1 for r in results if r["n_passed"] == 0)
    mixed = n_total - all_pass - all_fail

    console.print()
    console.print(Rule("SDPO Signal Summary", style="bold cyan"))
    stbl = Table(show_header=True, header_style="bold")
    stbl.add_column("Category", style="bold")
    stbl.add_column("Count", justify="right")
    stbl.add_column("Fraction", justify="right")
    stbl.add_column("SDPO Signal")
    stbl.add_row(
        "All pass", str(all_pass), f"{all_pass/n_total:.0%}",
        "[dim]No gradient (no failures)[/dim]",
    )
    stbl.add_row(
        "All fail", str(all_fail), f"{all_fail/n_total:.0%}",
        "[dim]KL signal but no sibling solution[/dim]",
    )
    stbl.add_row(
        "Mixed", str(mixed), f"{mixed/n_total:.0%}",
        "[bold green]Full SDPO signal (sibling + failures)[/bold green]",
    )
    stbl.add_row("", "", "", "")
    stbl.add_row(
        "[bold]Total[/bold]", f"[bold]{n_total}[/bold]", "",
        f"[bold]Mean pass rate: {mean_pass_rate:.1%}[/bold]",
    )
    console.print(stbl)

    # Timing breakdown
    api_times = [r.get("dt_api", 0) for r in results]
    grade_times = [r.get("dt_grade", 0) for r in results]
    if any(api_times) or any(grade_times):
        avg_api = sum(api_times) / n_total
        avg_grade = sum(grade_times) / n_total
        console.print(
            f"\n[dim]Avg per problem: api={avg_api:.1f}s  grade={avg_grade:.1f}s"
            f"  total={avg_api + avg_grade:.1f}s[/dim]"
        )

    if mixed == 0:
        console.print(
            "\n[bold red]Warning:[/bold red] No mixed groups. "
            "SDPO will have no sibling solutions. "
            "Consider a different model/dataset difficulty or larger group_size."
        )
    elif mixed / n_total < 0.2:
        console.print(
            f"\n[bold yellow]Warning:[/bold yellow] Only {mixed/n_total:.0%} of groups are mixed. "
            "Most training steps will have weak or no SDPO signal."
        )


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def default_output_path(args: argparse.Namespace) -> str:
    model_short = args.model.rsplit("/", 1)[-1]
    ds = args.dataset_name or "all"
    return f"eval_{model_short}_{ds}_{args.split}_n{args.n}.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate pass@k on LiveCodeBench-style coding tasks via OpenAI-compatible API",
    )
    p.add_argument("--model", required=True, help="Model name for the API")
    p.add_argument("--base-url", default=None, help="API base URL (e.g. http://localhost:8000/v1 for vLLM)")
    p.add_argument("--limit", type=int, default=None, help="Max number of tasks to evaluate")
    p.add_argument("-n", type=int, default=8, help="Samples per problem (N)")
    p.add_argument("-k", default="1", help="Comma-separated k values for pass@k (each <= N)")
    p.add_argument("--dataset-name", default=None, choices=["lcbv5", "lcbv6", "codeforces"])
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--sandbox", default="local", choices=["sandboxfusion", "local", "bwrap"])
    p.add_argument("--workers", type=int, default=4, help="Concurrent problems in flight")
    p.add_argument("-o", "--output", default=None, help="JSONL output path (default: auto-generated)")
    p.add_argument("--extra-body", default=None, help="JSON string passed as extra_body to the API")
    p.add_argument("--no-think", action="store_true", help="Disable thinking (sets chat_template_kwargs.enable_thinking=false)")
    return p.parse_args()


async def amain() -> None:
    args = parse_args()

    ks = [int(x) for x in args.k.split(",")]
    for k in ks:
        if k > args.n:
            console.print(f"[red]k={k} exceeds n={args.n}[/red]")
            return

    extra_body: dict[str, Any] | None = None
    if args.extra_body:
        extra_body = json.loads(args.extra_body)
    if args.no_think:
        extra_body = extra_body or {}
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    output_path = Path(args.output or default_output_path(args))
    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key)

    console.print(f"[bold]Model:[/bold] {args.model}")
    if args.base_url:
        console.print(f"[bold]Base URL:[/bold] {args.base_url}")
    console.print(f"[bold]Samples per problem:[/bold] {args.n}  [bold]k values:[/bold] {ks}")
    console.print(f"[bold]Workers:[/bold] {args.workers}  [bold]Sandbox:[/bold] {args.sandbox}")
    if extra_body:
        console.print(f"[bold]Extra body:[/bold] {json.dumps(extra_body)}")
    console.print(f"[bold]Output:[/bold] {output_path}")

    console.print(f"\n[bold]Loading tasks ({args.split} split)...[/bold]")
    tasks = load_tasks(n=args.limit, split=args.split, seed=args.seed, dataset_name=args.dataset_name)
    if not tasks:
        console.print("[red]No tasks loaded.[/red]")
        return
    console.print(f"[bold]Tasks loaded:[/bold] {len(tasks)}")

    cache = load_cache(output_path)
    n_cached = len(cache)
    if n_cached:
        console.print(f"[bold]Resuming:[/bold] {n_cached}/{len(tasks)} already completed in {output_path}")

    sem = asyncio.Semaphore(args.workers)
    results: list[dict] = list(cache.values())
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task("Evaluating", total=len(tasks), completed=n_cached)

        async def run_one(idx: int) -> None:
            if idx in cache:
                return
            result = await eval_one(
                task_idx=idx,
                task=tasks[idx],
                client=client,
                model=args.model,
                n=args.n,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                sandbox_backend=args.sandbox,
                extra_body=extra_body,
                sem=sem,
            )
            append_result(output_path, result)
            results.append(result)
            progress.advance(ptask)

            n_passed = result["n_passed"]
            n = result["n"]
            if n_passed == n:
                label = "[green]ALL PASS [/green]"
            elif n_passed == 0:
                label = "[red]ALL FAIL [/red]"
            else:
                frac = f"{n_passed}/{n}"
                label = f"[yellow]MIXED {frac:<3s}[/yellow]"
            w = len(str(len(tasks)))
            pk1 = pass_at_k(n, n_passed, 1)
            dt_api = result.get("dt_api", 0)
            dt_grade = result.get("dt_grade", 0)
            progress.console.print(
                f"  Problem {idx+1:>{w}}/{len(tasks)}:  {label}  pass@1={pk1:.2f}"
                f"  [dim]api={dt_api:5.1f}s  grade={dt_grade:5.1f}s[/dim]"
            )

        await asyncio.gather(*(run_one(i) for i in range(len(tasks))))

    elapsed = time.time() - t0
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")

    results.sort(key=lambda r: r["task_idx"])
    print_summary(results, ks)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
