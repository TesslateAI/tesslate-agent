import pytest
from tesslate_agent.agent.tools.shell_ops.execute import is_command_safe

@pytest.mark.parametrize("command, expected_safe", [
    ("ls -la", True),
    ("echo 'hello'", True),
    ("rm -rf /", False),
    ("chmod 777 sensitive_file.key", False),
    (":(){ :|:& };:", False),
])
def test_is_command_safe(command, expected_safe):
    safe, msg = is_command_safe(command)
    assert safe == expected_safe
    if not safe:
        assert "dangerous" in msg.lower()