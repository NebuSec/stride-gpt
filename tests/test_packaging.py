"""Tests for what the published wheel declares, read from pyproject.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _dependencies() -> dict[str, Requirement]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    reqs = (Requirement(spec) for spec in data["project"]["dependencies"])
    return {req.name: req for req in reqs}


def test_llm_calls_depend_on_the_openai_sdk_alone():
    dependencies = _dependencies()
    assert "openai" in dependencies
    assert "litellm" not in dependencies
