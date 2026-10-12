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
with patch('pm.ensure_import') as install:
    import agent.bedrock_adapter as adapter
    assert adapter.configure_bedrock_openai_client_kwargs({'base_url':'https://api.openai.com/v1'}) == {'base_url':'https://api.openai.com/v1'}
    install.assert_not_called()
''', tmp_path)



def test_selected_bedrock_provider_uses_managed_sdk_on_demand(tmp_path):
    run_isolated("""
import sys, types
from unittest.mock import patch
sdk = types.ModuleType('boto3'); sdk.__version__ = '1.34.59'
def ensure(extra):
    assert extra == 'bedrock'
    sys.modules['boto3'] = sdk
with patch('pm.ensure_import', side_effect=ensure) as install:
    from agent.bedrock_adapter import _require_boto3
    install.assert_not_called()
    assert _require_boto3() is sdk
    install.assert_called_once_with('bedrock')
""", tmp_path)
