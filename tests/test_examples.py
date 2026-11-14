"""Run every example script in chapters/*/examples/ and fail if any of them crashes.

Examples marked with `requires: network` in their docstring run only when
RUN_NETWORK_EXAMPLES=1 is set.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(ROOT.glob("chapters/*/examples/*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: str(p.relative_to(ROOT / "chapters")))
def test_example_runs(script: Path) -> None:
    if "requires: network" in script.read_text() and os.environ.get("RUN_NETWORK_EXAMPLES") != "1":
        pytest.skip("needs network: set RUN_NETWORK_EXAMPLES=1 to run")

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )
    assert result.returncode == 0, f"{script.name} failed:\n{result.stderr}"
    assert result.stdout.strip(), f"{script.name} printed nothing"


def test_examples_found() -> None:
    assert len(EXAMPLES) >= 10
