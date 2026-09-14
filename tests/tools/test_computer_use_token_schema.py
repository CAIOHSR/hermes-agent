"""Modern Cua schemas retain element tokens without legacy capability tags."""
from unittest.mock import MagicMock
import pytest
from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.cua_backend_session import _CuaDriverSession

@pytest.mark.parametrize("schema_support", [True, False])
def test_snapshot_token_follows_advertised_input_schema(schema_support):
    backend = CuaDriverBackend()
    session = MagicMock(spec=_CuaDriverSession)
    session.supports_capability.return_value = False
    session.supports_input_property.side_effect = lambda tool, prop: schema_support and prop == "element_token"
    session.call_tool.return_value = {"data": "ok", "images": [], "image_mime_types": [], "structuredContent": None, "isError": False}
    backend._session = session
    backend._active_pid = 111
    backend._active_window_id = 222
    backend._snapshot_tokens = {5: "fresh-snapshot:5"}
    backend.click(element=5, button="left")
    name, args = session.call_tool.call_args.args
    assert name == "click"
    assert args["element_index"] == 5
    assert args.get("element_token") == ("fresh-snapshot:5" if schema_support else None)
