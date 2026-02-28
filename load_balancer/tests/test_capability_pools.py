"""Tests for Phase 1: capability-typed backend pools.

Covers:
  - Config parser (_parse_backend_capabilities) with pipes, semicolons, cloud prefix stripping
  - Capability matching logic (_backend_has_capabilities)
  - Node filtering with fallback (_filter_nodes_by_capability)
  - Header integration in _parse_execution_mode
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure package path for local src
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_balancer.config import _parse_backend_capabilities
from load_balancer import config as lb_config
from load_balancer.main import (
    _backend_has_capabilities,
    _filter_nodes_by_capability,
)


# ---------------------------------------------------------------------------
# _parse_backend_capabilities
# ---------------------------------------------------------------------------

class TestParseBackendCapabilities:
    """Unit tests for the config parser."""

    def test_basic_semicolon_separator(self):
        raw = "claude=reasoning,code;gemini=research,multimodal"
        result = _parse_backend_capabilities(raw)
        assert result == {
            "claude": frozenset({"reasoning", "code"}),
            "gemini": frozenset({"research", "multimodal"}),
        }

    def test_pipe_separator(self):
        raw = "claude=reasoning,code|gemini=research,multimodal"
        result = _parse_backend_capabilities(raw)
        assert result == {
            "claude": frozenset({"reasoning", "code"}),
            "gemini": frozenset({"research", "multimodal"}),
        }

    def test_mixed_pipes_and_semicolons(self):
        """Pipes and semicolons in the same string both work as entry separators."""
        raw = "claude=reasoning;gemini=research|local=fast"
        result = _parse_backend_capabilities(raw)
        assert len(result) == 3
        assert result["claude"] == frozenset({"reasoning"})
        assert result["gemini"] == frozenset({"research"})
        assert result["local"] == frozenset({"fast"})

    def test_cloud_prefix_stripping(self):
        """'cloud:claude' key should be stored as 'claude'."""
        raw = "cloud:claude=reasoning,code"
        result = _parse_backend_capabilities(raw)
        assert "claude" in result
        assert "cloud:claude" not in result
        assert result["claude"] == frozenset({"reasoning", "code"})

    def test_cloud_prefix_stripping_with_multiple(self):
        raw = "cloud:openai=reasoning;cloud:gemini=multimodal;localhost:11434=fast"
        result = _parse_backend_capabilities(raw)
        assert set(result.keys()) == {"openai", "gemini", "localhost:11434"}

    def test_local_host_port_key(self):
        """host:port backends are stored verbatim (no prefix stripping)."""
        raw = "localhost:11434=fast,private"
        result = _parse_backend_capabilities(raw)
        assert "localhost:11434" in result
        assert result["localhost:11434"] == frozenset({"fast", "private"})

    def test_case_sensitive_capabilities(self):
        """Capability strings preserve case — no lowercasing."""
        raw = "backend=FastInference,GPU"
        result = _parse_backend_capabilities(raw)
        assert result["backend"] == frozenset({"FastInference", "GPU"})
        assert "fastinference" not in result["backend"]

    def test_empty_string(self):
        assert _parse_backend_capabilities("") == {}

    def test_whitespace_only(self):
        assert _parse_backend_capabilities("   ") == {}

    def test_malformed_no_equals(self):
        """Entries without '=' are silently skipped."""
        raw = "claude-reasoning;gemini=research"
        result = _parse_backend_capabilities(raw)
        assert list(result.keys()) == ["gemini"]

    def test_malformed_empty_caps(self):
        """Entry with empty capability list after '=' is skipped (empty frozenset)."""
        raw = "claude=;gemini=research"
        result = _parse_backend_capabilities(raw)
        # "claude=" yields an empty frozenset which is falsy → skipped
        assert "claude" not in result
        assert result["gemini"] == frozenset({"research"})

    def test_whitespace_around_entries(self):
        raw = "  claude = reasoning , code ;  gemini = research  "
        result = _parse_backend_capabilities(raw)
        assert result["claude"] == frozenset({"reasoning", "code"})
        assert result["gemini"] == frozenset({"research"})

    def test_config_roundtrip(self):
        """Parse then serialize back to a canonical form."""
        raw = "claude=reasoning,code;gemini=research,multimodal;localhost:11434=fast,private"
        parsed = _parse_backend_capabilities(raw)
        # Serialize: sort entries and capabilities for determinism
        parts = []
        for name in sorted(parsed):
            caps = ",".join(sorted(parsed[name]))
            parts.append(f"{name}={caps}")
        serialized = ";".join(parts)
        reparsed = _parse_backend_capabilities(serialized)
        assert reparsed == parsed


# ---------------------------------------------------------------------------
# _backend_has_capabilities
# ---------------------------------------------------------------------------

class TestBackendHasCapabilities:
    """Unit tests for capability matching on individual backends."""

    def setup_method(self):
        self._orig = lb_config.BACKEND_CAPABILITIES

    def teardown_method(self):
        lb_config.BACKEND_CAPABILITIES = self._orig

    def _set_caps(self, mapping):
        lb_config.BACKEND_CAPABILITIES = mapping

    def test_superset_match(self):
        """Backend with more caps than required → passes."""
        self._set_caps({"claude": frozenset({"reasoning", "code", "math"})})
        assert _backend_has_capabilities("cloud:claude", frozenset({"reasoning"})) is True

    def test_exact_match(self):
        self._set_caps({"claude": frozenset({"reasoning", "code"})})
        assert _backend_has_capabilities("cloud:claude", frozenset({"reasoning", "code"})) is True

    def test_subset_fails(self):
        """Backend has fewer caps than required → fails."""
        self._set_caps({"claude": frozenset({"reasoning"})})
        assert _backend_has_capabilities("cloud:claude", frozenset({"reasoning", "code"})) is False

    def test_empty_required_matches_all(self):
        """Empty required set always matches (no constraints)."""
        self._set_caps({"claude": frozenset({"reasoning"})})
        assert _backend_has_capabilities("cloud:claude", frozenset()) is True

    def test_unknown_backend(self):
        """Backend not in capability map → empty set → fails unless required is empty."""
        self._set_caps({})
        assert _backend_has_capabilities("cloud:unknown", frozenset({"code"})) is False
        assert _backend_has_capabilities("cloud:unknown", frozenset()) is True

    def test_local_backend_lookup(self):
        """Local backend (host:port) is looked up without prefix stripping."""
        self._set_caps({"localhost:11434": frozenset({"fast", "private"})})
        assert _backend_has_capabilities("localhost:11434", frozenset({"fast"})) is True

    def test_cloud_prefix_stripped_for_lookup(self):
        """cloud:X nodes strip the prefix to match config keys."""
        self._set_caps({"openai": frozenset({"reasoning"})})
        assert _backend_has_capabilities("cloud:openai", frozenset({"reasoning"})) is True

    def test_case_sensitive_matching(self):
        """Capability matching respects case."""
        self._set_caps({"backend": frozenset({"FastInference"})})
        assert _backend_has_capabilities("backend", frozenset({"FastInference"})) is True
        assert _backend_has_capabilities("backend", frozenset({"fastinference"})) is False


# ---------------------------------------------------------------------------
# _filter_nodes_by_capability
# ---------------------------------------------------------------------------

class TestFilterNodesByCapability:
    """Unit tests for the node filter with fallback behavior."""

    def setup_method(self):
        self._orig = lb_config.BACKEND_CAPABILITIES

    def teardown_method(self):
        lb_config.BACKEND_CAPABILITIES = self._orig

    def _set_caps(self, mapping):
        lb_config.BACKEND_CAPABILITIES = mapping

    def test_no_required_returns_all(self):
        """None required → pass-through, no filtering."""
        nodes = ["cloud:claude", "cloud:gemini", "localhost:11434"]
        assert _filter_nodes_by_capability(nodes, None) == nodes

    def test_empty_required_returns_all(self):
        """Empty frozenset → pass-through."""
        nodes = ["cloud:claude", "cloud:gemini"]
        assert _filter_nodes_by_capability(nodes, frozenset()) == nodes

    def test_filter_keeps_matching_excludes_nonmatching(self):
        self._set_caps({
            "claude": frozenset({"reasoning", "code"}),
            "gemini": frozenset({"research", "multimodal"}),
            "localhost:11434": frozenset({"fast", "private"}),
        })
        nodes = ["cloud:claude", "cloud:gemini", "localhost:11434"]
        result = _filter_nodes_by_capability(nodes, frozenset({"reasoning"}))
        assert result == ["cloud:claude"]

    def test_filter_multiple_matches(self):
        """Multiple backends match → all returned."""
        self._set_caps({
            "claude": frozenset({"reasoning", "code"}),
            "openai": frozenset({"reasoning", "math"}),
            "gemini": frozenset({"multimodal"}),
        })
        nodes = ["cloud:claude", "cloud:openai", "cloud:gemini"]
        result = _filter_nodes_by_capability(nodes, frozenset({"reasoning"}))
        assert set(result) == {"cloud:claude", "cloud:openai"}

    def test_fallback_when_no_match(self):
        """No backend matches → return full unfiltered list (advisory, not hard-fail)."""
        self._set_caps({
            "claude": frozenset({"reasoning"}),
        })
        nodes = ["cloud:claude", "cloud:gemini", "localhost:11434"]
        result = _filter_nodes_by_capability(nodes, frozenset({"nonexistent_cap"}))
        assert result == nodes

    def test_empty_node_list(self):
        """Empty node list stays empty regardless of capabilities."""
        result = _filter_nodes_by_capability([], frozenset({"reasoning"}))
        assert result == []

    def test_unknown_backend_in_config_skippable(self):
        """Backends in config but not in node list don't affect filtering."""
        self._set_caps({
            "claude": frozenset({"reasoning"}),
            "unknown_service": frozenset({"reasoning", "magic"}),
        })
        nodes = ["cloud:claude"]
        result = _filter_nodes_by_capability(nodes, frozenset({"reasoning"}))
        assert result == ["cloud:claude"]


# ---------------------------------------------------------------------------
# _parse_execution_mode header integration
# ---------------------------------------------------------------------------

class TestParseExecutionModeCapability:
    """Verify x-require-capability header is parsed into ExtendedExecutionConfig."""

    def test_capability_header_parsed(self):
        """x-require-capability populates required_capabilities."""
        from starlette.testclient import TestClient
        from starlette.requests import Request
        from starlette.datastructures import Headers
        from load_balancer.main import _parse_execution_mode

        # Build a minimal ASGI scope to construct a Request
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"x-execution-mode", b"race"),
                (b"x-require-capability", b"research,multimodal"),
            ],
        }
        req = Request(scope)
        ext = _parse_execution_mode(req)
        assert ext is not None
        assert ext.required_capabilities == frozenset({"research", "multimodal"})

    def test_capability_header_case_preserved(self):
        """Capability strings from header are case-sensitive — no lowercasing."""
        from starlette.requests import Request
        from load_balancer.main import _parse_execution_mode

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"x-execution-mode", b"race"),
                (b"x-require-capability", b"FastInference,GPU"),
            ],
        }
        req = Request(scope)
        ext = _parse_execution_mode(req)
        assert ext is not None
        assert ext.required_capabilities == frozenset({"FastInference", "GPU"})
        assert "fastinference" not in ext.required_capabilities

    def test_no_capability_header(self):
        """Without the header, required_capabilities is None."""
        from starlette.requests import Request
        from load_balancer.main import _parse_execution_mode

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"x-execution-mode", b"race"),
            ],
        }
        req = Request(scope)
        ext = _parse_execution_mode(req)
        assert ext is not None
        assert ext.required_capabilities is None
