"""
Interactive LiveCodeBench environment explorer with SDPO-style generation.

Loads problems from the DeepCoder dataset (LiveCodeBench splits), lets you
type Python solutions, and grades them via SandboxFusion.  The [g]enerate
option runs the real SDPOCodeEnv via ``do_group_rollout``, grades solutions,
and -- when they fail -- computes teacher-vs-student logprob deltas following
the SDPO self-teacher approach (https://arxiv.org/abs/2601.20802).

Prerequisites:
    docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
    export SANDBOX_URL=http://localhost:8080/run_code

Usage:
    python play_w_code_env.py                 # 3 test-split problems, manual only
    python play_w_code_env.py --n-tasks 5     # 5 problems
    python play_w_code_env.py --split train   # use train split
    python play_w_code_env.py --dataset-name lcbv6  # only LCB v6 problems
    # With Tinker generation (requires API key):
    python play_w_code_env.py --model Qwen/Qwen3-8B --renderer qwen3
    # Grouped rollout -- generate 8 samples, use passing siblings as teacher demos:
    python play_w_code_env.py --model Qwen/Qwen3-8B -n 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import tinker
from rich.color import Color
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.code_rl.code_grading import extract_code_from_model
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import Trajectory

from env import (
    SDPOCodeGroupBuilder,
    Task,
    _parse_stdout,
    build_sdpo_teacher_inputs,
    load_tasks,
)
import sandbox

console = Console()


# ---------------------------------------------------------------------------
# TUI helpers
# ---------------------------------------------------------------------------

def read_multiline(prompt: str) -> str:
    console.print(f"[yellow]{prompt}[/yellow] (paste code, then enter two blank lines to submit)")
    lines: list[str] = []
    consecutive_blanks = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            consecutive_blanks += 1
            if consecutive_blanks >= 2 and lines:
                break
        else:
            consecutive_blanks = 0
        lines.append(line)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def print_problem(index: int, total: int, task: Task) -> None:
    n_tests = len(task.tests)
    test_type = task.tests[0].get("testtype", "unknown") if task.tests else "unknown"

    subtitle_parts = [f"{n_tests} test case(s)", f"type: {test_type}"]
    if task.tests:
        preview = task.tests[0]
        inp_preview = preview["input"][:200]
        out_preview = preview["output"][:200]
        subtitle_parts.append(f"sample in: {inp_preview}")
        subtitle_parts.append(f"sample out: {out_preview}")

    console.print()
    console.print(Panel(
        task.problem,
        title=f"[bold]Problem {index + 1}/{total}[/bold]",
        subtitle=f"[dim]{' | '.join(subtitle_parts)}[/dim]",
        border_style="cyan",
        expand=True,
        padding=(1, 2),
    ))
    console.print()


def _pretty_details(details: dict[str, Any]) -> str:
    out = dict(details)
    run = out.get("run_result")
    if isinstance(run, dict) and run.get("stdout"):
        run = dict(run)
        run["stdout"] = _parse_stdout(run["stdout"])
        out["run_result"] = run
    return json.dumps(out, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Logprob visualisation
# ---------------------------------------------------------------------------

def _delta_to_rgb(delta: float) -> tuple[int, int, int]:
    """Map a teacher-student logprob delta to an RGB colour."""
    clamp = max(-4.0, min(4.0, delta))
    t = clamp / 4.0
    if t >= 0:
        r, g, b = int(180 * (1 - t)), int(180 + 75 * t), int(180 * (1 - t))
    else:
        r, g, b = int(180 + 75 * (-t)), int(180 * (1 + t)), int(180 * (1 + t))
    return r, g, b


def display_logprob_diff(
    tokenizer: Any,
    tokens: list[int],
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    label: str,
) -> None:
    """Print tokens coloured by teacher-student logprob delta."""
    console.print(Rule(label, style="bold cyan"))

    n = min(len(tokens), len(student_logprobs), len(teacher_logprobs))
    deltas: list[float] = []
    text = Text()

    for i in range(n):
        tok_str = tokenizer.decode([tokens[i]])
        d = teacher_logprobs[i] - student_logprobs[i]
        deltas.append(d)
        r, g, b = _delta_to_rgb(d)
        text.append(tok_str, style=Style(color=Color.from_rgb(r, g, b)))

    console.print(text)

    if deltas:
        avg = sum(deltas) / len(deltas)
        pos_frac = sum(1 for d in deltas if d > 0) / len(deltas)
        console.print(
            f"\n[dim]avg delta: {avg:+.3f}  |  "
            f"min: {min(deltas):+.3f}  max: {max(deltas):+.3f}  |  "
            f"teacher more confident: {pos_frac:.0%} of tokens[/dim]"
        )


def display_logprob_confidence(
    tokenizer: Any,
    tokens: list[int],
    logprobs: list[float],
    label: str,
) -> None:
    """Print tokens coloured by absolute logprob (confidence)."""
    console.print(Rule(label, style="bold cyan"))

    text = Text()
    for i in range(min(len(tokens), len(logprobs))):
        tok_str = tokenizer.decode([tokens[i]])
        lp = logprobs[i]
        t = max(0.0, min(1.0, -lp / 5.0))
        r = int(80 + 175 * t)
        g = int(220 - 140 * t)
        b = int(80)
        text.append(tok_str, style=Style(color=Color.from_rgb(r, g, b)))

    console.print(text)


# ---------------------------------------------------------------------------
# Panel display helpers
# ---------------------------------------------------------------------------

def _panels_grid(panels: list[Panel]) -> Table:
    table = Table.grid(expand=True)
    for _ in panels:
        table.add_column(ratio=1)
    table.add_row(*panels)
    return table


def _build_rollout_panels(
    trajectories: list[Trajectory],
    tokenizer: Any,
) -> Table:
    """Build side-by-side panels showing each rollout result."""
    n = len(trajectories)
    panels: list[Panel] = []
    for i, traj in enumerate(trajectories):
        t = traj.transitions[0]
        raw_text = tokenizer.decode(t.ac.tokens)
        body = Text(raw_text, overflow="fold")
        passed = bool(t.metrics.get("correct", 0))
        has_code = bool(t.metrics.get("has_code", 0))
        border = "green" if passed else ("yellow" if has_code else "red")
        n_tok = len(t.ac.tokens)
        title = f"[bold]\\[{i+1}/{n}][/bold] {n_tok} tok"
        if has_code:
            code = extract_code_from_model(raw_text)
            title += f" | code: {len(code)} chars" if code else ""
        else:
            title += " | [red]no code block[/red]"
        if passed:
            title += " | [green]PASS[/green]"
        elif has_code:
            title += " | [red]FAIL[/red]"
        panels.append(Panel(body, title=title, border_style=border, expand=True, padding=(0, 1)))
    return _panels_grid(panels)


# ---------------------------------------------------------------------------
# Generate flow (uses real Env + rollout)
# ---------------------------------------------------------------------------

async def _handle_generate(
    task: Task,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    tokenizer: Any,
    temperature: float,
    max_tokens: int,
    n: int = 1,
    sandbox: str = "sandboxfusion",
) -> None:
    """Run the [g]enerate flow using the real SDPOCodeEnv via do_group_rollout."""

    builder = SDPOCodeGroupBuilder(
        task=task,
        group_size=n,
        renderer=renderer,
        timeout=6,
        backend=sandbox,
    )
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # -- 1. Rollout (sample + grade via Env) -----------------------------------
    console.print(Rule(f"Step 1: Rollout (n={n})", style="bold cyan"))
    console.print(f"[dim]T={temperature}  max_tokens={max_tokens}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Running {n} rollout(s) through SDPOCodeEnv...", total=None)
        traj_group = await do_group_rollout(builder, policy)

    trajectories = traj_group.trajectories_G
    console.print(_build_rollout_panels(trajectories, tokenizer))

    # -- 2. Summarise results --------------------------------------------------
    console.print(Rule("Step 2: Grading summary", style="bold cyan"))

    passed_indices: list[int] = []
    failed_indices: list[int] = []

    for i, traj in enumerate(trajectories):
        t = traj.transitions[0]
        passed = bool(t.metrics.get("correct", 0))
        has_code = bool(t.metrics.get("has_code", 0))
        feedback = str(t.logs.get("feedback", ""))
        tag = f"\\[{i+1}/{n}]"
        if has_code:
            raw_text = tokenizer.decode(t.ac.tokens)
            code = extract_code_from_model(raw_text)
            if code:
                code_panel = Panel(
                    Syntax(code, "python", theme="monokai", word_wrap=True),
                    title=f"[bold]{tag}[/bold] {len(code)} chars",
                    border_style="green" if passed else "red",
                    expand=True,
                    padding=(0, 1),
                )
                console.print(code_panel)
        if passed:
            console.print(f"  {tag} [green bold]PASSED[/green bold]")
            passed_indices.append(i)
        else:
            console.print(f"  {tag} [red bold]FAILED[/red bold]")
            if feedback:
                console.print(f"  [dim]{feedback[:500]}[/dim]")
            failed_indices.append(i)

    pass_rate = len(passed_indices) / n if n else 0
    console.print(f"\n[dim]Pass rate: {len(passed_indices)}/{n} ({pass_rate:.0%})[/dim]")

    if not failed_indices:
        first_pass_traj = trajectories[passed_indices[0]]
        ac = first_pass_traj.transitions[0].ac
        if ac.maybe_logprobs is not None:
            display_logprob_confidence(
                tokenizer, ac.tokens, ac.logprobs,
                "First passing sample (coloured by student self-confidence, no teacher)",
            )
        return

    if passed_indices:
        console.print(f"\n[dim]Using sample \\[{passed_indices[0]+1}] as sibling solution for teacher prompt[/dim]")
    else:
        console.print("\n[dim]No passing sibling -- teacher prompt will use feedback only[/dim]")

    # -- 3. Teacher forward pass (SDPO re-prompt) ------------------------------
    teacher_inputs_G = build_sdpo_teacher_inputs(task, trajectories, renderer, tokenizer)

    first_failed = failed_indices[0]
    first_traj = trajectories[first_failed]
    first_ac = first_traj.transitions[0].ac
    student_tokens = first_ac.tokens
    student_logprobs = first_ac.maybe_logprobs
    teacher_input = teacher_inputs_G[first_failed]

    if student_logprobs is None:
        console.print("[yellow]No student logprobs available -- skipping teacher diff.[/yellow]")
        return
    if teacher_input is None:
        console.print("[yellow]No teacher input for this trajectory -- skipping.[/yellow]")
        return

    console.print(Rule(f"Step 3: Teacher logprobs for sample [{first_failed+1}]", style="bold cyan"))
    console.print(
        f"[dim]Computing teacher logprobs on {len(student_tokens)} student tokens "
        f"(teacher prompt: {teacher_input.length - len(student_tokens)} tok)...[/dim]"
    )

    teacher_full_text = tokenizer.decode(teacher_input.to_ints())
    console.print(Panel(
        Text(teacher_full_text, overflow="fold"),
        title=f"[bold]Full teacher input ({teacher_input.length} tok)[/bold]",
        border_style="magenta",
        expand=True,
        padding=(1, 2),
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(
            f"Computing teacher logprobs on {len(student_tokens)} tokens...",
            total=None,
        )
        all_lps = await sampling_client.compute_logprobs_async(teacher_input)

    n_student = len(student_tokens)
    teacher_lps_raw = all_lps[-n_student:] if all_lps else [None] * n_student
    teacher_lps = [lp if lp is not None else 0.0 for lp in teacher_lps_raw]

    # -- 4. Coloured diff ------------------------------------------------------
    console.print(Rule(f"Step 4: Logprob diff for sample [{first_failed+1}] (teacher - student)", style="bold cyan"))

    display_logprob_diff(
        tokenizer, student_tokens, student_logprobs, teacher_lps,
        f"Sample [{first_failed+1}] tokens coloured by teacher-student delta "
        "(green = teacher agrees, red = teacher disagrees)",
    )

    if len(failed_indices) > 1:
        console.print(f"\n[dim]({len(failed_indices) - 1} more failed sample(s) not shown)[/dim]")


# ---------------------------------------------------------------------------
# Non-interactive generate mode
# ---------------------------------------------------------------------------


async def generate_mode(
    tasks: list[Task],
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    tokenizer: Any,
    temperature: float,
    max_tokens: int,
    n: int,
    sandbox: str,
) -> None:
    """Run _handle_generate for each task non-interactively and exit."""
    _print_sandbox_info(sandbox, len(tasks))
    gen_hint = " [bold yellow]g[/bold yellow]enerate |" if sampling_client else ""
    console.print(f"Commands:{gen_hint} [bold yellow]n[/bold yellow]ext | [bold yellow]q[/bold yellow]uit | Enter to submit code\n")
    for i, task in enumerate(tasks):
        print_problem(i, len(tasks), task)
        await _handle_generate(
            task, sampling_client, renderer, tokenizer,
            temperature, max_tokens, n=n, sandbox=sandbox,
        )
    console.print("\n[bold green]All problems done.[/bold green]")


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------


def _print_sandbox_info(sandbox: str, n_tasks: int) -> None:
    if sandbox in ("local", "bwrap"):
        console.print(f"[bold]Sandbox backend:[/bold] {sandbox}")
    else:
        sandbox_url = os.getenv("SANDBOX_URL", "http://localhost:8080/run_code")
        console.print(f"[bold]Sandbox URL:[/bold] {sandbox_url}")
    console.print(f"[bold]Tasks loaded:[/bold] {n_tasks}")


async def interactive_loop(
    tasks: list[Task],
    sampling_client: tinker.SamplingClient | None = None,
    renderer: renderers.Renderer | None = None,
    tokenizer: Any = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    n: int = 1,
    sandbox_name: str = "sandboxfusion",
) -> None:
    _print_sandbox_info(sandbox_name, len(tasks))
    gen_hint = " [bold yellow]g[/bold yellow]enerate |" if sampling_client else ""
    console.print(f"Commands:{gen_hint} [bold yellow]n[/bold yellow]ext | [bold yellow]q[/bold yellow]uit | Enter to submit code\n")

    for i, task in enumerate(tasks):
        print_problem(i, len(tasks), task)

        while True:
            try:
                cmd = console.input("[yellow]>[/yellow] ").strip().lower()
            except EOFError:
                console.print("\nExiting.")
                return

            if cmd in ("q", "quit"):
                console.print("Exiting.")
                return
            if cmd in ("n", "next", "s", "skip"):
                console.print("[yellow]Skipped.[/yellow]")
                break
            if cmd in ("g", "gen", "generate") and sampling_client:
                assert renderer is not None and tokenizer is not None
                await _handle_generate(
                    task, sampling_client, renderer, tokenizer,
                    temperature, max_tokens, n=n, sandbox=sandbox_name,
                )
                continue

            if cmd:
                console.print(
                    f"[dim]Unknown command '{cmd}'. "
                    f"Press Enter to submit code, or use n/g/q.[/dim]"
                )
                continue

            raw = read_multiline("Paste your solution:")
            if not raw.strip():
                continue

            code = extract_code_from_model(raw) or raw

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(f"Grading ({len(task.tests)} test cases)...", total=None)
                passed, details = await sandbox_name.grade_code(task.tests, code, backend=sandbox_name)

            if passed:
                console.print("[green bold]PASSED -- all tests correct![/green bold]")
            else:
                console.print("[red bold]FAILED[/red bold]")
                detail_str = _pretty_details(details)
                if len(detail_str) > 3000:
                    detail_str = detail_str[:3000] + "\n  ...(truncated)"
                console.print(Syntax(detail_str, "json", theme="monokai", word_wrap=True))

    console.print("\n[bold green]All problems done.[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive LiveCodeBench explorer")
    parser.add_argument("--n-tasks", type=int, default=3, help="Number of problems to load")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--sandbox-url", default=None, help="Override SANDBOX_URL env var")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (train split)")
    parser.add_argument("--model", default=None, help="Tinker base model for [g]enerate (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--renderer", default="qwen3_disable_thinking", help="Renderer name matching model family (qwen3, llama3, ...)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for [g]enerate")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens for [g]enerate")
    parser.add_argument("-n", type=int, default=1, help="Number of samples per generation (grouped rollout)")
    parser.add_argument(
        "--generate", action="store_true",
        help="Non-interactive: run generation for all loaded tasks and exit. "
             "Requires --model.  NOTE: consumes Tinker API tokens.",
    )
    parser.add_argument(
        "--dataset-name", default=None,
        choices=["lcbv5", "lcbv6", "codeforces"],
        help="Sub-dataset to load (default: all DeepCoder). Matches sdpo_on_policy_distillation.py.",
    )
    parser.add_argument(
        "--sandbox", default="sandboxfusion",
        choices=["sandboxfusion", "local", "bwrap"],
        help="Sandbox backend for code grading. "
             "'local' and 'bwrap' do not require Docker. "
             "'bwrap' requires bubblewrap in PATH.",
    )
    args = parser.parse_args()

    if args.sandbox_url:
        os.environ["SANDBOX_URL"] = args.sandbox_url

    if args.generate and not args.model:
        console.print("[red]--generate requires --model.[/red]")
        sys.exit(1)

    sampling_client: tinker.SamplingClient | None = None
    renderer: renderers.Renderer | None = None
    tokenizer = None
    if args.model:
        console.print(f"[bold]Initialising Tinker for {args.model}...[/bold]")
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=args.model)
        tokenizer = tokenizer_utils.get_tokenizer(args.model)
        renderer = renderers.get_renderer(name=args.renderer, tokenizer=tokenizer)

    console.print(f"[bold]Loading {args.n_tasks} tasks from '{args.split}' split...[/bold]")
    tasks = load_tasks(args.n_tasks, split=args.split, seed=args.seed, dataset_name=args.dataset_name)
    if not tasks:
        console.print("[red]No tasks loaded. Check dataset availability.[/red]")
        sys.exit(1)

    if args.generate:
        asyncio.run(generate_mode(
            tasks,
            sampling_client=sampling_client,
            renderer=renderer,
            tokenizer=tokenizer,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=args.n,
            sandbox=args.sandbox,
        ))
    else:
        asyncio.run(interactive_loop(
            tasks,
            sampling_client=sampling_client,
            renderer=renderer,
            tokenizer=tokenizer,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=args.n,
            sandbox_name=args.sandbox,
        ))


if __name__ == "__main__":
    main()
