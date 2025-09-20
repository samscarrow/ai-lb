"""Helpers for sanitising client payloads before routing upstream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import config


def _normalize_model(model: Any) -> Optional[str]:
    """Return a clean model string or None when missing/blank."""
    if model is None:
        return None
    if isinstance(model, str):
        value = model.strip()
        return value or None
    value = str(model).strip()
    return value or None


@dataclass
class SanitizedRequest:
    payload: Dict[str, Any]
    model_name: Optional[str]
    missing_fields: list[str] = field(default_factory=list)
    default_applied: Optional[str] = None


def sanitize_chat_request(raw: Dict[str, Any]) -> SanitizedRequest:
    """Ensure chat requests have a model and usable conversation payload."""
    payload = dict(raw) if raw is not None else {}
    missing: list[str] = []
    default_applied: Optional[str] = None

    model = _normalize_model(payload.get("model"))
    if not model:
        fallback = config.DEFAULT_CHAT_MODEL or config.LM_MODEL_SENTINEL
        if fallback:
            payload["model"] = fallback
            model = fallback
            default_applied = fallback
        else:
            missing.append("model")
    else:
        payload["model"] = model

    messages = payload.get("messages")
    prompt = payload.get("prompt")
    single_input = payload.get("input")
    if not (messages or prompt or single_input):
        missing.append("messages")

    return SanitizedRequest(payload=payload, model_name=model, missing_fields=missing, default_applied=default_applied)


def sanitize_embeddings_request(raw: Dict[str, Any]) -> SanitizedRequest:
    """Ensure embeddings requests have both a model and input payload."""
    payload = dict(raw) if raw is not None else {}
    missing: list[str] = []
    default_applied: Optional[str] = None

    model = _normalize_model(payload.get("model"))
    if not model:
        fallback = config.DEFAULT_EMBEDDINGS_MODEL
        if fallback:
            payload["model"] = fallback
            model = fallback
            default_applied = fallback
        else:
            missing.append("model")
    else:
        payload["model"] = model

    if "input" not in payload or payload.get("input") in (None, ""):
        missing.append("input")

    return SanitizedRequest(payload=payload, model_name=model, missing_fields=missing, default_applied=default_applied)
