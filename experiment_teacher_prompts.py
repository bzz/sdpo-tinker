"""Experiment with teacher prompt designs for SDPO distillation signal.

Loads cached eval data (from eval_pass_at_k.py), reconstructs teacher prompts
under various designs, computes teacher logprobs via Tinker or vLLM, and compares
the resulting reverse-KL signals with colorized Rich output and a Markdown report.

Usage (vLLM -- for models served locally):
    python experiment_teacher_prompts.py \
        --cache eval_Qwen3-30B-A3B-Instruct-2507-FP8_lcbv5_train_n8.jsonl \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
        --vllm-url http://localhost:8003/v1 \
        --dataset-name lcbv5 --split train \
        --max-problems 5

Usage (Tinker -- for models on the Tinker service):
    python experiment_teacher_prompts.py \
        --cache eval_Qwen3-4B-Instruct-2507-FP8_lcbv5_train_n8.jsonl \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --dataset-name lcbv5 --split train

    # Run only specific variants:
    python experiment_teacher_prompts.py ... --variants baseline,feedback_only,multiturn

    # List available variants:
    python experiment_teacher_prompts.py --list-variants
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import AsyncOpenAI
from rich.color import Color
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.recipes.code_rl.code_grading import extract_code_from_model

from env import CODE_PROMPT, load_tasks

console = Console()


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    variant_name: str
    teacher_prompt_tokens: int
    student_tokens: int
    mean_reverse_kl: float
    teacher_agreement_frac: float
    signal_magnitude: float
    code_reverse_kl: float = 0.0
    code_magnitude: float = 0.0
    code_tokens: int = 0
    per_token_kl: list[float] = field(default_factory=list, repr=False)


@dataclass
class SampleResult:
    task_idx: int
    sample_idx: int
    student_response: str
    feedback: str
    sibling_code: str | None
    student_logprobs: list[float]
    student_token_ids: list[int]
    variants: dict[str, VariantResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt variant registry
# ---------------------------------------------------------------------------

VariantFn = Callable[
    [str, str | None, str, str, str | None],
    list[renderers.Message],
]

_VARIANTS: dict[str, tuple[str, VariantFn]] = {}


def variant(name: str, description: str):
    """Decorator to register a teacher prompt variant."""
    def decorator(fn: VariantFn) -> VariantFn:
        _VARIANTS[name] = (description, fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

@variant("baseline", "Paper Table 2: problem + solution + feedback + closing")
def _baseline(problem, sibling_response, feedback, student_response, sibling_code):
    solution_section = f"\nCorrect solution:\n\n{sibling_response}\n\n" if sibling_response else ""
    feedback_section = f"\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback}\n\n" if feedback else ""
    content = f"{problem}{solution_section}{feedback_section}Correctly solve the original question.\n"
    return [{"role": "user", "content": content}]


@variant("feedback_only", "Problem + feedback only (no sibling solution)")
def _feedback_only(problem, sibling_response, feedback, student_response, sibling_code):
    feedback_section = f"\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback}\n\n" if feedback else ""
    content = f"{problem}{feedback_section}Correctly solve the original question.\n"
    return [{"role": "user", "content": content}]


@variant("solution_only", "Problem + sibling solution only (no feedback)")
def _solution_only(problem, sibling_response, feedback, student_response, sibling_code):
    solution_section = f"\nCorrect solution:\n\n{sibling_response}\n\n" if sibling_response else ""
    content = f"{problem}{solution_section}Correctly solve the original question.\n"
    return [{"role": "user", "content": content}]


@variant("bare_reprompt", "Just the problem + closing instruction")
def _bare_reprompt(problem, sibling_response, feedback, student_response, sibling_code):
    return [{"role": "user", "content": f"{problem}\nCorrectly solve the original question.\n"}]


@variant("code_only_solution", "Problem + extracted code from sibling (no reasoning) + feedback")
def _code_only_solution(problem, sibling_response, feedback, student_response, sibling_code):
    solution_section = ""
    if sibling_code:
        solution_section = f"\nCorrect solution:\n\n```python\n{sibling_code}\n```\n\n"
    feedback_section = f"\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback}\n\n" if feedback else ""
    content = f"{problem}{solution_section}{feedback_section}Correctly solve the original question.\n"
    return [{"role": "user", "content": content}]


@variant("multiturn", "Multi-turn: user=problem, assistant=student, user=feedback+solution")
def _multiturn(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    correction = ""
    if sibling_response:
        correction += f"\nHere is a correct solution for reference:\n\n{sibling_response}\n\n"
    if feedback:
        correction += f"Your previous attempt had the following issue:\n\n{feedback}\n\n"
    correction += "Please solve the problem correctly."
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": student_response},
        {"role": "user", "content": correction},
    ]


@variant("system_user_split", "System=instruction, User=problem+solution+feedback")
def _system_user_split(problem, sibling_response, feedback, student_response, sibling_code):
    system = (
        "You are a coding expert. You will be given a coding problem, and you need "
        "to write a correct Python program that matches the specification and passes "
        "all tests. The time limit is 1 second."
    )
    body = problem
    if sibling_response:
        body += f"\n\nCorrect solution:\n\n{sibling_response}\n\n"
    if feedback:
        body += f"\nFeedback from an unsuccessful attempt:\n\n{feedback}\n\n"
    body += "\nCorrectly solve the original question.\n"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": body},
    ]


@variant("xml_tags", "Structured XML: <problem>, <solution>, <feedback>, <instruction>")
def _xml_tags(problem, sibling_response, feedback, student_response, sibling_code):
    parts = [f"<problem>\n{problem}\n</problem>"]
    if sibling_response:
        parts.append(f"<solution>\n{sibling_response}\n</solution>")
    if feedback:
        parts.append(f"<feedback>\n{feedback}\n</feedback>")
    parts.append("<instruction>Correctly solve the original problem.</instruction>")
    return [{"role": "user", "content": "\n\n".join(parts)}]


@variant("markdown_sections", "Markdown: ## Problem, ## Solution, ## Feedback")
def _markdown_sections(problem, sibling_response, feedback, student_response, sibling_code):
    parts = [f"## Problem\n\n{problem}"]
    if sibling_response:
        parts.append(f"## Correct Solution\n\n{sibling_response}")
    if feedback:
        parts.append(f"## Feedback on Your Attempt\n\n{feedback}")
    parts.append("## Task\n\nCorrectly solve the problem above.")
    return [{"role": "user", "content": "\n\n".join(parts)}]


@variant("error_type_only", "Baseline but feedback is just the error type (e.g. 'Wrong Answer')")
def _error_type_only(problem, sibling_response, feedback, student_response, sibling_code):
    short_feedback = feedback.split("\n")[0] if feedback else ""
    solution_section = f"\nCorrect solution:\n\n{sibling_response}\n\n" if sibling_response else ""
    feedback_section = f"\nYour previous attempt resulted in: {short_feedback}\n\n" if short_feedback else ""
    content = f"{problem}{solution_section}{feedback_section}Correctly solve the original question.\n"
    return [{"role": "user", "content": content}]


@variant("concise_expert", "Concise expert tone: terse problem restatement + bug diagnosis")
def _concise_expert(problem, sibling_response, feedback, student_response, sibling_code):
    parts = [problem]
    if feedback:
        error_line = feedback.split("\n")[0]
        parts.append(f"\nBug diagnosis: {error_line}.")
    if sibling_code:
        parts.append(f"\nReference implementation:\n```python\n{sibling_code}\n```")
    parts.append("\nWrite the correct solution.")
    return [{"role": "user", "content": "\n".join(parts)}]


@variant("socratic", "Socratic: asks model to identify and fix the error")
def _socratic(problem, sibling_response, feedback, student_response, sibling_code):
    parts = [problem]
    if feedback:
        parts.append(
            f"\nA previous attempt to solve this problem failed with:\n{feedback}\n\n"
            "What went wrong? Identify the bug, explain why it fails, "
            "and write a correct solution."
        )
    if sibling_response:
        parts.append(f"\nFor reference, here is a working solution:\n{sibling_response}")
    return [{"role": "user", "content": "\n".join(parts)}]


@variant("diff_feedback", "Wrong Answer feedback in diff format: -expected / +got")
def _diff_feedback(problem, sibling_response, feedback, student_response, sibling_code):
    diff_fb = feedback
    if "Expected" in feedback and "Output" in feedback:
        lines = feedback.split("\n")
        diff_parts = []
        i = 0
        while i < len(lines):
            if lines[i].strip() == "Expected" and i + 1 < len(lines):
                diff_parts.append(f"- {lines[i+1].strip()}")
                i += 2
            elif lines[i].strip() == "Output" and i + 1 < len(lines):
                diff_parts.append(f"+ {lines[i+1].strip()}")
                i += 2
            elif lines[i].strip() == "Input" and i + 1 < len(lines):
                diff_parts.append(f"Input: {lines[i+1].strip()}")
                i += 2
            else:
                if lines[i].strip():
                    diff_parts.append(lines[i])
                i += 1
        if diff_parts:
            diff_fb = "\n".join(diff_parts)

    solution_section = f"\nCorrect solution:\n\n{sibling_response}\n\n" if sibling_response else ""
    content = f"{problem}{solution_section}\nTest results:\n{diff_fb}\n\nCorrectly solve the original question.\n"
    return [{"role": "user", "content": content}]


@variant("with_student_code", "Baseline + include the student's failing code explicitly")
def _with_student_code(problem, sibling_response, feedback, student_response, sibling_code):
    student_code = extract_code_from_model(student_response)
    solution_section = f"\nCorrect solution:\n\n{sibling_response}\n\n" if sibling_response else ""
    student_section = f"\nYour previous (incorrect) code:\n\n```python\n{student_code}\n```\n\n" if student_code else ""
    feedback_section = f"\nFeedback:\n{feedback}\n\n" if feedback else ""
    content = f"{problem}{student_section}{feedback_section}{solution_section}Correctly solve the original question.\n"
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Round 2: Multi-turn variants (exploring the winning direction)
# ---------------------------------------------------------------------------

@variant("mt_feedback_only", "Multi-turn with feedback only (no sibling solution)")
def _mt_feedback_only(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    correction = ""
    if feedback:
        correction += f"Your solution failed:\n\n{feedback}\n\n"
    correction += "Fix the issue and solve the problem correctly."
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": student_response},
        {"role": "user", "content": correction},
    ]


@variant("mt_solution_only", "Multi-turn with sibling solution only (no feedback)")
def _mt_solution_only(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    correction = "Your solution is incorrect."
    if sibling_response:
        correction += f"\n\nHere is a correct solution for reference:\n\n{sibling_response}"
    correction += "\n\nSolve the problem correctly."
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": student_response},
        {"role": "user", "content": correction},
    ]


@variant("mt_bare", "Multi-turn bare: just 'try again' after seeing the student attempt")
def _mt_bare(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": student_response},
        {"role": "user", "content": "That solution is incorrect. Try again and provide a correct solution."},
    ]


@variant("mt_terse", "Multi-turn terse: one-line error + 'fix it'")
def _mt_terse(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    error_line = feedback.split("\n")[0] if feedback else "incorrect"
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": student_response},
        {"role": "user", "content": f"Error: {error_line}. Fix it."},
    ]


@variant("mt_code_ref", "Multi-turn with only extracted sibling code (no reasoning)")
def _mt_code_ref(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    correction = ""
    if feedback:
        correction += f"Your solution failed: {feedback}\n\n"
    if sibling_code:
        correction += f"Reference implementation:\n```python\n{sibling_code}\n```\n\n"
    correction += "Write the correct solution."
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": student_response},
        {"role": "user", "content": correction},
    ]


@variant("mt_sys_expert", "Multi-turn with system prompt as coding expert")
def _mt_sys_expert(problem, sibling_response, feedback, student_response, sibling_code):
    system = (
        "You are a coding expert. When shown a failed attempt, identify the bug "
        "and write a correct solution that passes all tests."
    )
    prompt_text = CODE_PROMPT.format(problem=problem)
    correction = ""
    if feedback:
        correction += f"Your solution failed:\n\n{feedback}\n\n"
    if sibling_response:
        correction += f"Correct solution for reference:\n\n{sibling_response}\n\n"
    correction += "Solve the problem correctly."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": student_response},
        {"role": "user", "content": correction},
    ]


@variant("mt_xml_feedback", "Multi-turn with XML-structured correction turn")
def _mt_xml_feedback(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    parts = []
    if feedback:
        parts.append(f"<error>{feedback}</error>")
    if sibling_code:
        parts.append(f"<reference_solution>\n{sibling_code}\n</reference_solution>")
    parts.append("<instruction>Write the correct solution.</instruction>")
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": student_response},
        {"role": "user", "content": "\n".join(parts)},
    ]


# ---------------------------------------------------------------------------
# Round 3: Code-focused variants (strip reasoning, focus on bug site)
# ---------------------------------------------------------------------------

@variant("code_repair", "Multi-turn: assistant=code_only (no reasoning), correction=feedback")
def _code_repair(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    student_code = extract_code_from_model(student_response)
    code_msg = f"```python\n{student_code}\n```" if student_code else student_response
    correction = ""
    if feedback:
        correction += f"Your code failed:\n\n{feedback}\n\n"
    correction += "Write a corrected solution."
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": code_msg},
        {"role": "user", "content": correction},
    ]


@variant("code_repair_ref", "Multi-turn: assistant=code_only, correction=feedback+sibling_code")
def _code_repair_ref(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    student_code = extract_code_from_model(student_response)
    code_msg = f"```python\n{student_code}\n```" if student_code else student_response
    correction = ""
    if feedback:
        correction += f"Your code failed:\n\n{feedback}\n\n"
    if sibling_code:
        correction += f"Here is a correct implementation:\n```python\n{sibling_code}\n```\n\n"
    correction += "Write the corrected solution."
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": code_msg},
        {"role": "user", "content": correction},
    ]


@variant("code_diff", "Multi-turn: assistant=failing_code, correction=shows passing code")
def _code_diff(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    student_code = extract_code_from_model(student_response)
    code_msg = f"```python\n{student_code}\n```" if student_code else student_response
    correction = "Your code is incorrect."
    if feedback:
        correction += f"\n\n{feedback}"
    if sibling_code:
        correction += f"\n\nThe correct solution is:\n```python\n{sibling_code}\n```"
    correction += "\n\nStudy the correct solution and write the fixed version."
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": code_msg},
        {"role": "user", "content": correction},
    ]


@variant("mt_assert", "Multi-turn with assertion-style feedback")
def _mt_assert(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    assert_fb = feedback
    if feedback:
        lines = feedback.split("\n")
        parts = [lines[0]]
        inp = out = exp = None
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped == "Input" and i + 1 < len(lines):
                inp = lines[i + 1].strip()
                i += 2
            elif stripped == "Output" and i + 1 < len(lines):
                out = lines[i + 1].strip()
                i += 2
            elif stripped == "Expected" and i + 1 < len(lines):
                exp = lines[i + 1].strip()
                i += 2
            else:
                i += 1
        if inp and exp and out:
            assert_fb = f"f({inp}) returned {out}, expected {exp}"
        elif inp and exp:
            assert_fb = f"f({inp}) should return {exp}"

    correction = f"Test failure: {assert_fb}\n\nFix the code."
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": student_response},
        {"role": "user", "content": correction},
    ]


@variant("mt_hint", "Multi-turn with structural hint derived from comparing fail/pass code")
def _mt_hint(problem, sibling_response, feedback, student_response, sibling_code):
    prompt_text = CODE_PROMPT.format(problem=problem)
    student_code = extract_code_from_model(student_response)

    hint = ""
    if student_code and sibling_code:
        fail_lines = student_code.splitlines()
        pass_lines = sibling_code.splitlines()
        diffs = []
        for i, (fl, pl) in enumerate(zip(fail_lines, pass_lines)):
            if fl.strip() != pl.strip():
                diffs.append(f"  Line {i+1}: yours has `{fl.strip()}`, correct has `{pl.strip()}`")
        if diffs:
            hint = "Differences between your code and a correct solution:\n" + "\n".join(diffs[:5])
        else:
            if len(fail_lines) != len(pass_lines):
                hint = f"Your code has {len(fail_lines)} lines, the correct solution has {len(pass_lines)} lines."

    correction = ""
    if feedback:
        correction += f"{feedback}\n\n"
    if hint:
        correction += f"{hint}\n\n"
    correction += "Fix the bug."
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": student_response},
        {"role": "user", "content": correction},
    ]


# ---------------------------------------------------------------------------
# Logprob backend abstraction
# ---------------------------------------------------------------------------

class LogprobBackend(ABC):
    """Computes logprobs for a student response conditioned on teacher messages."""

    @abstractmethod
    async def get_logprobs(
        self,
        messages: list[dict[str, str]],
        student_response: str,
    ) -> tuple[list[float], int]:
        """Return (per_token_logprobs_for_student_response, prompt_token_count)."""
        ...


class TinkerBackend(LogprobBackend):
    """Uses Tinker's compute_logprobs_async via the renderer to tokenize."""

    def __init__(self, sampling_client, renderer, tokenizer):
        import tinker as _tinker
        self._tinker = _tinker
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.tokenizer = tokenizer

    async def get_logprobs(self, messages, student_response):
        prompt = self.renderer.build_generation_prompt(messages)
        prompt_ids = prompt.to_ints()
        student_ids = self.tokenizer.encode(student_response, add_special_tokens=False)
        full_ids = prompt_ids + student_ids
        model_input = self._tinker.ModelInput.from_ints(full_ids)
        all_lps = await self.sampling_client.compute_logprobs_async(model_input)
        n = len(student_ids)
        raw = all_lps[-n:] if all_lps else [None] * n
        return [lp if lp is not None else 0.0 for lp in raw], len(prompt_ids)


class VLLMBackend(LogprobBackend):
    """Uses vLLM's OpenAI-compatible Chat Completions API with prompt_logprobs.

    Strategy: send teacher messages + the student response as a final assistant
    message.  Request prompt_logprobs=1 and max_tokens=1 (minimum allowed).
    The response.prompt_logprobs covers all prompt tokens; we slice the last
    N entries corresponding to the student response tokens.
    """

    def __init__(self, client: AsyncOpenAI, model: str, tokenizer, chat_template_kwargs: dict | None = None):
        self.client = client
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}

    async def get_logprobs(self, messages, student_response):
        full_messages = list(messages) + [
            {"role": "assistant", "content": student_response},
        ]
        extra_body: dict[str, Any] = {"prompt_logprobs": 1}
        if self.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = self.chat_template_kwargs

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=1,
            temperature=0.0,
            extra_body=extra_body,
        )

        prompt_lps_raw = getattr(response, "prompt_logprobs", None)
        if prompt_lps_raw is None:
            raise RuntimeError(
                "vLLM did not return prompt_logprobs. "
                "Ensure you are running vLLM >= 0.5.5 and using a non-streaming request."
            )

        student_token_ids = self.tokenizer.encode(student_response, add_special_tokens=False)
        n_student = len(student_token_ids)
        n_prompt_total = len(prompt_lps_raw)
        n_teacher_prompt = n_prompt_total - n_student

        student_lp_entries = prompt_lps_raw[n_teacher_prompt:]
        logprobs: list[float] = []
        for entry in student_lp_entries:
            if entry is None:
                logprobs.append(0.0)
            elif isinstance(entry, dict):
                vals = list(entry.values())
                logprobs.append(vals[0]["logprob"] if vals else 0.0)
            elif isinstance(entry, list) and entry:
                logprobs.append(entry[0].get("logprob", 0.0) if isinstance(entry[0], dict) else 0.0)
            else:
                logprobs.append(0.0)

        return logprobs, max(n_teacher_prompt, 0)


# ---------------------------------------------------------------------------
# Logprob coloring (from play_w_code_env.py)
# ---------------------------------------------------------------------------

def _delta_to_rgb(delta: float) -> tuple[int, int, int]:
    clamp = max(-4.0, min(4.0, delta))
    t = clamp / 4.0
    if t >= 0:
        r, g, b = int(180 * (1 - t)), int(180 + 75 * t), int(180 * (1 - t))
    else:
        r, g, b = int(180 + 75 * (-t)), int(180 * (1 + t)), int(180 * (1 + t))
    return r, g, b


def display_conversation(messages: list[dict[str, str]], label: str, *, verbose: bool = False) -> None:
    """Print a teacher prompt's messages as role-colored panels."""
    ROLE_STYLES = {"system": "magenta", "user": "cyan", "assistant": "green"}
    console.print(Rule(f"[bold]{label}[/bold]", style="dim"))
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        style = ROLE_STYLES.get(role, "white")
        if verbose:
            body = content
        else:
            max_len = 1500
            body = content if len(content) <= max_len else content[:max_len] + f"\n\n... ({len(content) - max_len} chars truncated)"
        console.print(Panel(
            Text(body, overflow="fold"),
            title=f"[bold {style}]{role}[/bold {style}]",
            border_style=style,
            expand=True,
            padding=(0, 1),
        ))


def display_logprob_diff(
    tokenizer, tokens, student_logprobs, teacher_logprobs, label,
):
    console.print(Rule(label, style="bold cyan"))
    n = min(len(tokens), len(student_logprobs), len(teacher_logprobs))
    deltas = []
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


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_eval_cache(path: Path) -> dict[int, dict]:
    results: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            results[entry["task_idx"]] = entry
    return results


def load_results_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_results_cache(path: Path, data: dict[str, Any]):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _cache_key(task_idx: int, sample_idx: int, variant_name: str) -> str:
    return f"{task_idx}:{sample_idx}:{variant_name}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def find_code_block_mask(response: str, tokenizer: Any) -> list[bool]:
    """Return a boolean mask over tokens: True for tokens inside ```...``` code blocks."""
    tokens = tokenizer.encode(response, add_special_tokens=False)
    n = len(tokens)
    mask = [False] * n

    in_code = False
    char_pos = 0
    token_char_ranges: list[tuple[int, int]] = []
    for i in range(n):
        tok_str = tokenizer.decode([tokens[i]])
        start = char_pos
        char_pos += len(tok_str)
        token_char_ranges.append((start, char_pos))

    fence_positions: list[tuple[int, bool]] = []
    search_start = 0
    while True:
        idx = response.find("```", search_start)
        if idx == -1:
            break
        is_open = not in_code
        fence_positions.append((idx, is_open))
        in_code = is_open
        search_start = idx + 3

    code_ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(fence_positions):
        if fence_positions[i][1]:  # opening fence
            open_pos = fence_positions[i][0]
            nl = response.find("\n", open_pos + 3)
            content_start = nl + 1 if nl != -1 else open_pos + 3
            if i + 1 < len(fence_positions) and not fence_positions[i + 1][1]:
                content_end = fence_positions[i + 1][0]
                code_ranges.append((content_start, content_end))
                i += 2
            else:
                code_ranges.append((content_start, len(response)))
                i += 1
        else:
            i += 1

    for ti, (ts, te) in enumerate(token_char_ranges):
        for cs, ce in code_ranges:
            if ts >= cs and te <= ce:
                mask[ti] = True
                break

    return mask


def compute_metrics(
    student_lps: list[float],
    teacher_lps: list[float],
    code_mask: list[bool] | None = None,
) -> VariantResult:
    n = min(len(student_lps), len(teacher_lps))
    per_token_kl = [student_lps[i] - teacher_lps[i] for i in range(n)]
    mean_rkl = sum(per_token_kl) / n if n else 0.0
    agree_frac = sum(1 for kl in per_token_kl if kl < 0) / n if n else 0.0
    magnitude = sum(abs(kl) for kl in per_token_kl) / n if n else 0.0

    code_rkl = 0.0
    code_mag = 0.0
    code_n = 0
    if code_mask and len(code_mask) >= n:
        code_kls = [per_token_kl[i] for i in range(n) if code_mask[i]]
        code_n = len(code_kls)
        if code_n:
            code_rkl = sum(code_kls) / code_n
            code_mag = sum(abs(k) for k in code_kls) / code_n

    return VariantResult(
        variant_name="",
        teacher_prompt_tokens=0,
        student_tokens=n,
        mean_reverse_kl=mean_rkl,
        teacher_agreement_frac=agree_frac,
        signal_magnitude=magnitude,
        code_reverse_kl=code_rkl,
        code_magnitude=code_mag,
        code_tokens=code_n,
        per_token_kl=per_token_kl,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_results: list[SampleResult], output_path: Path):
    variant_names = list(all_results[0].variants.keys()) if all_results else []

    agg: dict[str, dict[str, list[float]]] = {
        vn: {"rkl": [], "mag": [], "c_rkl": [], "c_mag": [], "prompt_tok": []}
        for vn in variant_names
    }
    for sr in all_results:
        for vn, vr in sr.variants.items():
            agg[vn]["rkl"].append(vr.mean_reverse_kl)
            agg[vn]["mag"].append(vr.signal_magnitude)
            agg[vn]["c_rkl"].append(vr.code_reverse_kl)
            agg[vn]["c_mag"].append(vr.code_magnitude)
            agg[vn]["prompt_tok"].append(vr.teacher_prompt_tokens)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    lines = [
        "# Teacher Prompt Experiment Results",
        "",
        f"**Problems analyzed**: {len(all_results)}",
        f"**Variants tested**: {len(variant_names)}",
        "",
        "## Summary (ranked by Code Magnitude)",
        "",
        "| Variant | Rev KL | Magnitude | Code KL | Code Mag | Prompt Tok |",
        "|---------|-------:|----------:|--------:|---------:|-----------:|",
    ]

    ranked = sorted(variant_names, key=lambda vn: _mean(agg[vn]["c_mag"]), reverse=True)
    for vn in ranked:
        d = agg[vn]
        lines.append(
            f"| {vn} | {_mean(d['rkl']):+.4f} | {_mean(d['mag']):.4f} "
            f"| {_mean(d['c_rkl']):+.4f} | {_mean(d['c_mag']):.4f} | {_mean(d['prompt_tok']):.0f} |"
        )

    lines += [
        "",
        "**Rev KL** = mean(student_lp - teacher_lp) over all response tokens. Positive = teacher less confident (suppresses student tokens).",
        "",
        "**Magnitude** = mean |student_lp - teacher_lp| over all response tokens (gradient strength).",
        "",
        "**Code KL / Code Mag** = same metrics restricted to tokens inside ```code blocks``` only (excludes reasoning/explanation).",
        "",
        "## Variant Descriptions",
        "",
    ]
    for vn in ranked:
        desc = _VARIANTS[vn][0] if vn in _VARIANTS else "?"
        lines.append(f"- **{vn}**: {desc}")

    lines += ["", "## Per-Problem Breakdown", ""]
    for sr in all_results:
        lines.append(f"### Task {sr.task_idx}, sample {sr.sample_idx}")
        fb = sr.feedback
        lines.append(f"- Feedback: `{fb[:120]}...`" if len(fb) > 120 else f"- Feedback: `{fb}`")
        lines.append(f"- Student tokens: {len(sr.student_token_ids)}")
        first_vr = next(iter(sr.variants.values()), None)
        if first_vr:
            lines.append(f"- Code tokens: {first_vr.code_tokens}")
        lines.append("")
        lines.append("| Variant | Rev KL | Magnitude | Code KL | Code Mag | Prompt Tok |")
        lines.append("|---------|-------:|----------:|--------:|---------:|-----------:|")
        for vn in ranked:
            if vn in sr.variants:
                vr = sr.variants[vn]
                lines.append(
                    f"| {vn} | {vr.mean_reverse_kl:+.4f} | {vr.signal_magnitude:.4f} "
                    f"| {vr.code_reverse_kl:+.4f} | {vr.code_magnitude:.4f} | {vr.teacher_prompt_tokens} |"
                )
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    console.print(f"\n[bold green]Report written to {output_path}[/bold green]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _infer_renderer(model_name: str) -> str:
    try:
        return get_recommended_renderer_name(model_name)
    except (KeyError, ValueError):
        lower = model_name.lower()
        if "qwen3" in lower:
            return "qwen3_instruct" if "instruct" in lower else "qwen3"
        if "llama" in lower:
            return "llama3"
        if "deepseek" in lower:
            return "deepseekv3"
        return "role_colon"


async def run_experiment(args: argparse.Namespace):
    renderer_name = args.renderer or _infer_renderer(args.model)
    console.print(f"[bold]Model:[/bold] {args.model}")
    console.print(f"[bold]Renderer:[/bold] {renderer_name}")
    console.print(f"[bold]Backend:[/bold] {'vLLM @ ' + args.vllm_url if args.vllm_url else 'Tinker'}")

    cache_path = Path(args.cache)
    if not cache_path.exists():
        console.print(f"[red]Cache file not found: {cache_path}[/red]")
        return

    eval_cache = load_eval_cache(cache_path)
    console.print(f"[bold]Eval cache:[/bold] {len(eval_cache)} tasks from {cache_path.name}")

    mixed_tasks = {
        idx: entry for idx, entry in eval_cache.items()
        if 0 < entry["n_passed"] < entry["n"]
    }
    console.print(f"[bold]Mixed groups:[/bold] {len(mixed_tasks)} (have both pass and fail)")

    if not mixed_tasks:
        console.print("[red]No mixed groups found -- nothing to experiment with.[/red]")
        return

    with console.status(f"[bold]Loading tasks ({args.split} split, {args.dataset_name})...[/bold]"):
        tasks = load_tasks(n=None, split=args.split, seed=args.seed, dataset_name=args.dataset_name)
    console.print(f"[bold]Tasks loaded:[/bold] {len(tasks)}")

    if args.variants:
        variant_names = [v.strip() for v in args.variants.split(",")]
        missing = [v for v in variant_names if v not in _VARIANTS]
        if missing:
            console.print(f"[red]Unknown variants: {missing}. Use --list-variants.[/red]")
            return
    else:
        variant_names = list(_VARIANTS.keys())

    console.print(f"[bold]Variants:[/bold] {len(variant_names)}: {', '.join(variant_names)}")

    # Init tokenizer and renderer (needed for both backends)
    tokenizer = tokenizer_utils.get_tokenizer(args.model)
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    # Init backend
    if args.vllm_url:
        api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        oai_client = AsyncOpenAI(base_url=args.vllm_url, api_key=api_key)
        chat_kwargs = {}
        if args.no_think:
            chat_kwargs["enable_thinking"] = False
        backend = VLLMBackend(oai_client, args.model, tokenizer, chat_kwargs or None)
        console.print(f"[bold]vLLM URL:[/bold] {args.vllm_url}")
    else:
        import tinker
        console.print(f"\n[bold]Initialising Tinker...[/bold]")
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=args.model)
        backend = TinkerBackend(sampling_client, renderer, tokenizer)

    # Results cache
    results_cache_path = cache_path.with_suffix(".teacher_experiment.json")
    results_cache = load_results_cache(results_cache_path)

    task_indices = sorted(mixed_tasks.keys())
    if args.skip_problems:
        task_indices = task_indices[args.skip_problems:]
    if args.max_problems:
        task_indices = task_indices[:args.max_problems]

    all_sample_results: list[SampleResult] = []
    total_api_calls = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task("Processing problems", total=len(task_indices))

        for task_idx in task_indices:
            entry = mixed_tasks[task_idx]
            if task_idx >= len(tasks):
                progress.console.print(f"[yellow]Skipping task_idx={task_idx} (out of range)[/yellow]")
                progress.advance(ptask)
                continue

            task = tasks[task_idx]

            sibling_response: str | None = None
            sibling_code: str | None = None
            for s in entry["samples"]:
                if s["correct"]:
                    sibling_response = s["response"]
                    sibling_code = extract_code_from_model(sibling_response)
                    break

            failed_samples = [
                (i, s) for i, s in enumerate(entry["samples"])
                if not s["correct"] and s.get("feedback")
            ]
            if args.max_failed:
                failed_samples = failed_samples[:args.max_failed]

            for sample_idx, sample in failed_samples:
                student_response = sample["response"]
                feedback = sample.get("feedback", "")
                student_token_ids = tokenizer.encode(student_response, add_special_tokens=False)
                code_mask = find_code_block_mask(student_response, tokenizer)

                # Student baseline logprobs
                stu_ck = _cache_key(task_idx, sample_idx, "__student__")
                if stu_ck in results_cache:
                    student_lps = results_cache[stu_ck]["logprobs"]
                else:
                    prompt_text = CODE_PROMPT.format(problem=task.problem)
                    student_msgs: list[dict[str, str]] = [{"role": "user", "content": prompt_text}]
                    student_lps, _ = await backend.get_logprobs(student_msgs, student_response)
                    results_cache[stu_ck] = {"logprobs": student_lps}
                    total_api_calls += 1

                sr = SampleResult(
                    task_idx=task_idx,
                    sample_idx=sample_idx,
                    student_response=student_response,
                    feedback=feedback,
                    sibling_code=sibling_code,
                    student_logprobs=student_lps,
                    student_token_ids=student_token_ids,
                )

                for vn in variant_names:
                    ck = _cache_key(task_idx, sample_idx, vn)
                    if ck in results_cache:
                        teacher_lps = results_cache[ck]["logprobs"]
                        prompt_tokens = results_cache[ck]["prompt_tokens"]
                    else:
                        _, variant_fn = _VARIANTS[vn]
                        messages = variant_fn(
                            task.problem, sibling_response, feedback,
                            student_response, sibling_code,
                        )
                        teacher_lps, prompt_tokens = await backend.get_logprobs(
                            messages, student_response,
                        )
                        results_cache[ck] = {
                            "logprobs": teacher_lps,
                            "prompt_tokens": prompt_tokens,
                        }
                        total_api_calls += 1

                    vr = compute_metrics(student_lps, teacher_lps, code_mask)
                    vr.variant_name = vn
                    vr.teacher_prompt_tokens = prompt_tokens
                    sr.variants[vn] = vr

                all_sample_results.append(sr)

                if not args.quiet:
                    console.print()
                    console.print(Panel(
                        Text(task.problem[:500] + ("..." if len(task.problem) > 500 else ""), overflow="fold"),
                        title=f"[bold]Task {task_idx} / sample {sample_idx}[/bold]",
                        subtitle=f"[dim]{feedback[:120]}[/dim]",
                        border_style="cyan",
                        expand=True,
                        padding=(0, 2),
                    ))

                    for vn in variant_names:
                        vr = sr.variants[vn]
                        desc = _VARIANTS[vn][0]

                        if args.show_prompts:
                            _, variant_fn = _VARIANTS[vn]
                            msgs = variant_fn(
                                task.problem, sibling_response, feedback,
                                student_response, sibling_code,
                            )
                            display_conversation(msgs, f"{vn}: teacher prompt ({vr.teacher_prompt_tokens} tok)", verbose=args.verbose)

                        display_logprob_diff(
                            tokenizer,
                            student_token_ids,
                            student_lps,
                            results_cache[_cache_key(task_idx, sample_idx, vn)]["logprobs"],
                            f"{vn}: {desc} (prompt: {vr.teacher_prompt_tokens} tok)",
                        )

            save_results_cache(results_cache_path, results_cache)
            progress.advance(ptask)

    save_results_cache(results_cache_path, results_cache)
    console.print(f"\n[dim]Total API calls: {total_api_calls}  |  Results cached to {results_cache_path.name}[/dim]")

    # Summary table
    if all_sample_results:
        console.print()
        console.print(Rule("Variant Comparison Summary", style="bold cyan"))
        table = Table(show_header=True, header_style="bold")
        table.add_column("Variant", style="bold")
        table.add_column("Description", max_width=35)
        table.add_column("Rev KL", justify="right")
        table.add_column("Magnitude", justify="right")
        table.add_column("Code KL", justify="right")
        table.add_column("Code Mag", justify="right")
        table.add_column("Prompt", justify="right")

        agg: dict[str, dict[str, list[float]]] = {}
        for vn in variant_names:
            agg[vn] = {"rkl": [], "mag": [], "c_rkl": [], "c_mag": [], "ptok": []}
        for sr in all_sample_results:
            for vn, vr in sr.variants.items():
                agg[vn]["rkl"].append(vr.mean_reverse_kl)
                agg[vn]["mag"].append(vr.signal_magnitude)
                agg[vn]["c_rkl"].append(vr.code_reverse_kl)
                agg[vn]["c_mag"].append(vr.code_magnitude)
                agg[vn]["ptok"].append(vr.teacher_prompt_tokens)

        def _m(xs):
            return sum(xs) / len(xs) if xs else 0.0

        ranked = sorted(variant_names, key=lambda vn: _m(agg[vn]["c_mag"]), reverse=True)
        for vn in ranked:
            d = agg[vn]
            desc = _VARIANTS[vn][0][:35]
            table.add_row(
                vn, desc,
                f"{_m(d['rkl']):+.4f}",
                f"{_m(d['mag']):.4f}",
                f"{_m(d['c_rkl']):+.4f}",
                f"{_m(d['c_mag']):.4f}",
                f"{_m(d['ptok']):.0f}",
            )
        console.print(table)

    if all_sample_results:
        report_path = cache_path.with_suffix(".teacher_experiment_report.md")
        generate_report(all_sample_results, report_path)


def main():
    parser = argparse.ArgumentParser(description="Experiment with SDPO teacher prompt designs")
    parser.add_argument("--cache", default=None, help="Path to eval_pass_at_k JSONL cache")
    parser.add_argument("--model", default=None, help="Model name (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)")
    parser.add_argument("--renderer", default=None, help="Override renderer (default: auto-detect from model)")
    parser.add_argument("--vllm-url", default=None, help="vLLM OpenAI-compat base URL (e.g. http://localhost:8003/v1)")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking (sets enable_thinking=false in chat template)")
    parser.add_argument("--dataset-name", default="lcbv5", choices=["lcbv5", "lcbv6", "codeforces"])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-problems", type=int, default=None, help="Max mixed-group problems to process")
    parser.add_argument("--skip-problems", type=int, default=0, help="Skip first N mixed-group problems")
    parser.add_argument("--max-failed", type=int, default=1, help="Max failed samples per group")
    parser.add_argument("--variants", default=None, help="Comma-separated variant names to run (default: all)")
    parser.add_argument("--list-variants", action="store_true", help="List available variants and exit")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-sample colorized output")
    parser.add_argument("--show-prompts", action="store_true", help="Print each variant's full conversation before its colorized output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Don't truncate conversation output")
    args = parser.parse_args()

    if args.list_variants:
        console.print("[bold]Available teacher prompt variants:[/bold]\n")
        for name, (desc, _) in _VARIANTS.items():
            console.print(f"  [bold cyan]{name:25s}[/bold cyan] {desc}")
        return

    if not args.cache or not args.model:
        parser.error("--cache and --model are required (unless --list-variants)")

    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
