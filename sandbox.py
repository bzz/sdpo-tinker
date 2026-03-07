"""Local and bubblewrap sandbox backends for code grading.

These backends run the same TEST_CODE / TEST_UTIL harness that SandboxFusion
uses, just locally via subprocess.  The upstream tinker_cookbook strings are
imported directly rather than vendored, so they stay in sync automatically.

Backends
--------
sandboxfusion
    Default.  Delegates entirely to tinker_cookbook's sandbox_check_correctness.
    Requires a running SandboxFusion container (SANDBOX_URL env var).

local
    Runs the test harness as a plain subprocess in a temporary directory.
    No network or filesystem isolation.  Safe to use only with trusted code.

bwrap
    Same as local but wrapped with bubblewrap (bwrap >= 0.4 must be in PATH).
    Uses --unshare-all for unprivileged Linux namespacing: no network, no
    persistent writes to the host.  Does not require Docker.
    Install: apt install bubblewrap  /  dnf install bubblewrap
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from tinker_cookbook.recipes.code_rl.code_grading import (
    postprocess_lcb_sample,
    sandbox_check_correctness,
)
from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_CODE, TEST_UTIL
from tinker_cookbook.sandbox import SandboxBackend


async def _run_in_tmpdir(
    test_cases: dict[str, str],
    generation: str,
    timeout: int,
    total_timeout: int,
    cmd_fn,  # (tmpdir: str) -> list[str]
) -> tuple[bool, dict[str, Any]]:
    """Write test files to a temp dir, run cmd_fn(tmpdir), return grading result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "test_cases.txt").write_text(json.dumps(test_cases))
        (p / "code.py").write_text(generation)
        (p / "testing_util.py").write_text(TEST_UTIL)
        (p / "run.py").write_text(TEST_CODE % {"timeout": timeout})

        proc = await asyncio.create_subprocess_exec(
            *cmd_fn(tmpdir),
            cwd=tmpdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=total_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return False, {"error": "total timeout exceeded"}

        return proc.returncode == 0, {
            "run_result": {
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "return_code": proc.returncode,
            }
        }


def _bwrap_cmd(tmpdir: str) -> list[str]:
    return [
        "bwrap", "--unshare-all", "--die-with-parent",
        "--ro-bind", "/", "/",
        "--bind", tmpdir, tmpdir,
        "--chdir", tmpdir,
        "--proc", "/proc",
        "--dev", "/dev",
        sys.executable, "run.py",
    ]


async def grade_code(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int = 6,
    backend: str = "sandboxfusion",
) -> tuple[bool, dict[str, Any]]:
    """Grade generated code against test cases using the chosen backend.

    Args:
        sample:     List of test cases in LiveCodeBench format (same as
                    sandbox_check_correctness).
        generation: Model-generated Python code to evaluate.
        timeout:    Per-test timeout in seconds (passed to the test harness).
        backend:    One of "sandboxfusion" (default), "local", or "bwrap".
                    Pass any other tinker_cookbook SandboxBackend value (e.g.
                    "modal") to have it forwarded to sandbox_check_correctness.

    Returns:
        (all_passed, details) tuple identical in shape to sandbox_check_correctness.
    """
    assert len(sample) >= 1, "sample must contain at least one test case"

    try:
        if backend == "sandboxfusion":
            return await sandbox_check_correctness(sample, generation, timeout=timeout)

        test_cases = postprocess_lcb_sample(sample)
        test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
        total_timeout = (timeout + 1) * test_cnt + 5

        if backend == "local":
            return await _run_in_tmpdir(
                test_cases, generation, timeout, total_timeout,
                cmd_fn=lambda _: [sys.executable, "run.py"],
            )

        if backend == "bwrap":
            if not shutil.which("bwrap"):
                raise RuntimeError(
                    "bwrap not found in PATH. "
                    "Install bubblewrap (apt install bubblewrap / dnf install bubblewrap)."
                )
            return await _run_in_tmpdir(
                test_cases, generation, timeout, total_timeout,
                cmd_fn=_bwrap_cmd,
            )

        # Forward other backends (e.g. "modal") to the upstream implementation.
        return await sandbox_check_correctness(
            sample, generation, timeout=timeout, backend=SandboxBackend(backend)
        )

    except Exception as e:
        return False, {"error": str(e)}
