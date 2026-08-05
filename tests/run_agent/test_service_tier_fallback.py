"""Regression tests for provider-safe fast-mode fallback transitions."""

from types import SimpleNamespace

from agent.chat_completion_helpers import _refresh_fast_mode_request_overrides
from hermes_cli.models import resolve_service_tier_overrides


def _agent(*, provider: str, model: str, tier: str | None = "priority"):
    return SimpleNamespace(
        provider=provider,
        model=model,
        service_tier=tier,
        request_overrides={"service_tier": "priority", "temperature": 0.2},
    )


def test_service_tier_resolver_allows_codex_luna_and_rejects_openrouter():
    assert resolve_service_tier_overrides("gpt-5.6-luna", "openai-codex") == {
        "service_tier": "priority"
    }
    assert resolve_service_tier_overrides("gpt-5.6-luna", "openrouter") is None
    assert resolve_service_tier_overrides(
        "deepseek/deepseek-v4-flash-0731", "openrouter"
    ) is None


def test_fast_override_is_stripped_on_fallback_and_restored_on_primary():
    agent = _agent(provider="openai-codex", model="gpt-5.6-luna")

    _refresh_fast_mode_request_overrides(agent)
    assert agent.request_overrides == {
        "service_tier": "priority",
        "temperature": 0.2,
    }

    agent.provider = "openrouter"
    agent.model = "deepseek/deepseek-v4-flash-0731"
    _refresh_fast_mode_request_overrides(agent)
    assert agent.request_overrides == {"temperature": 0.2}

    agent.provider = "openai-codex"
    agent.model = "gpt-5.6-luna"
    _refresh_fast_mode_request_overrides(agent)
    assert agent.request_overrides == {
        "service_tier": "priority",
        "temperature": 0.2,
    }


def test_normal_mode_never_invents_fast_override():
    agent = _agent(provider="openai-codex", model="gpt-5.6-luna", tier=None)
    _refresh_fast_mode_request_overrides(agent)
    assert agent.request_overrides == {"temperature": 0.2}
