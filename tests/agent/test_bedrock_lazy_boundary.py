"""Importing routing helpers must not install a provider the user did not select."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run_isolated(code, tmp_path):
    env = dict(os.environ, HERMES_HOME=str(tmp_path), PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_non_aws_import_never_installs_packages(tmp_path):
    run_isolated('''
from unittest.mock import patch
with patch('tools.lazy_deps.ensure') as install:
    import agent.bedrock_adapter as adapter
    assert adapter.configure_bedrock_openai_client_kwargs({'base_url':'https://api.openai.com/v1'}) == {'base_url':'https://api.openai.com/v1'}
    install.assert_not_called()
''', tmp_path)


def test_existing_sdk_is_used_without_install(tmp_path):
    run_isolated('''
import sys, types
from unittest.mock import patch
sdk = types.ModuleType('boto3'); sdk.__version__ = '1.34.59'
sys.modules['boto3'] = sdk
with patch('tools.lazy_deps.ensure') as install:
    from agent.bedrock_adapter import _require_boto3
    assert _require_boto3() is sdk
    install.assert_not_called()
''', tmp_path)


def test_actual_bedrock_use_installs_missing_sdk_only_once(tmp_path):
    run_isolated('''
import builtins, sys, types
from unittest.mock import patch
original = builtins.__import__
sdk = types.ModuleType('boto3'); sdk.__version__ = '1.34.59'
installed = False
def missing(name, *args, **kwargs):
    if name == 'boto3' and not installed: raise ModuleNotFoundError('boto3')
    return original(name, *args, **kwargs)
def ensure(feature, **kwargs):
    global installed
    assert feature == 'provider.bedrock' and kwargs == {'prompt':False}
    installed = True; sys.modules['boto3'] = sdk
with patch('tools.lazy_deps.ensure', side_effect=ensure) as install, patch('builtins.__import__', side_effect=missing):
    from agent.bedrock_adapter import _require_boto3
    install.assert_not_called()
    assert _require_boto3() is sdk
    assert _require_boto3() is sdk
    install.assert_called_once()
''', tmp_path)


def test_provider_install_failure_is_visible_on_use(tmp_path):
    run_isolated('''
import builtins
from unittest.mock import patch
original = builtins.__import__
def missing(name, *args, **kwargs):
    if name == 'boto3': raise ModuleNotFoundError('boto3')
    return original(name, *args, **kwargs)
with patch('tools.lazy_deps.ensure', side_effect=RuntimeError('provider unavailable')) as install, patch('builtins.__import__', side_effect=missing):
    from agent.bedrock_adapter import _require_boto3
    install.assert_not_called()
    try: _require_boto3()
    except RuntimeError as exc: assert str(exc) == 'provider unavailable'
    else: raise AssertionError('must expose install failure')
''', tmp_path)
