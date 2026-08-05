"""Regression tests for agentic cron service-tier propagation and guards."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _run_agentic_job(tmp_path, monkeypatch, *, config, runtime, job=None):
    from cron.scheduler import run_job

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(config, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)

    fake_db = MagicMock()
    fake_db.get_compression_tip.return_value = None
    fake_db.get_session.return_value = {"model_config": "{}"}
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "ok", "completed": True}
    job = job or {"id": "tier-job", "name": "tier", "prompt": "hello"}

    with (
        patch("cron.scheduler._hermes_home", home),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=fake_db),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime),
        patch("run_agent.AIAgent", return_value=agent) as agent_cls,
    ):
        result = run_job(job)

    assert result[0] is True
    assert result[2] == "ok"
    return agent_cls.call_args.kwargs


def test_agentic_cron_inherits_fast_service_tier_for_luna(tmp_path, monkeypatch):
    kwargs = _run_agentic_job(
        tmp_path,
        monkeypatch,
        config="model:\n  default: gpt-5.6-luna\nagent:\n  service_tier: fast\n",
        runtime={
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "provider": "openai",
            "api_mode": "chat_completions",
        },
    )

    assert kwargs["service_tier"] == "priority"
    assert kwargs["request_overrides"] == {"service_tier": "priority"}


@pytest.mark.parametrize(
    ("model", "provider"),
    [("deepseek-chat", "deepseek"), ("gpt-5.6-luna", "openrouter")],
)
def test_agentic_cron_strips_service_tier_for_incompatible_runtime(
    tmp_path, monkeypatch, model, provider
):
    kwargs = _run_agentic_job(
        tmp_path,
        monkeypatch,
        config=f"model:\n  default: {model}\nagent:\n  service_tier: fast\n",
        runtime={
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": provider,
            "api_mode": "chat_completions",
        },
    )

    assert kwargs["service_tier"] is None
    assert "service_tier" not in kwargs["request_overrides"]


def test_no_agent_cron_path_does_not_construct_agent(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    script = home / "scripts" / "watch.sh"
    script.write_text("#!/bin/sh\nprintf 'watchdog ok\\n'\n", encoding="utf-8")
    job = {
        "id": "no-agent-tier-job",
        "name": "watchdog",
        "prompt": None,
        "script": "watch.sh",
        "no_agent": True,
    }

    with patch("run_agent.AIAgent", side_effect=AssertionError("no_agent built an agent")):
        from cron.scheduler import run_job

        success, _doc, final_response, error = run_job(job)

    assert success is True
    assert error is None
    assert final_response == "watchdog ok"