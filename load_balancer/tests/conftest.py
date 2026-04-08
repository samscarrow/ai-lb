"""
Shared pytest configuration for unit tests.

Resets cloud-dependent config attributes so .env values (PLANNER_BACKEND,
CLOUD_BACKENDS) do not leak into unit tests that use FakeHTTPClient /
FakeRedis instead of real backends.  Tests that need cloud config set it
explicitly within their own body.
"""
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "load_balancer" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from load_balancer import config


@pytest.fixture(autouse=True)
def _reset_cloud_config(monkeypatch):
    """Clear cloud backend config before every unit test."""
    monkeypatch.setattr(config, "PLANNER_BACKEND", "")
    monkeypatch.setattr(config, "CLOUD_BACKENDS", {})
