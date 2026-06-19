import sys
from unittest.mock import MagicMock, patch

# --- WINDOWS COMPATIBILITY FIX ---
# The agent imports `fcntl` for memory file locking, which does not exist on Windows.
# We mock it globally in the test environment so the test suite can run cross-platform.
if sys.platform == "win32":
    sys.modules['fcntl'] = MagicMock()

import pytest
import asyncio
from pathlib import Path
from tesslate_agent.cli.runner import run_agent

@pytest.mark.anyio
async def test_graceful_shutdown_on_signal():
    # 1. Patch the final write function to prove the finally block executes
    with patch('tesslate_agent.cli.runner._write_trajectory') as mock_write:
        
        # 2. Simulate the user pressing Ctrl+C (KeyboardInterrupt) during execution
        with patch('tesslate_agent.cli.runner._drive_agent', side_effect=KeyboardInterrupt):
            
            # Start the task
            runner_task = asyncio.create_task(
                run_agent("test task", "openai/gpt-4o", Path("."), Path("output.json"))
            )
            
            await runner_task
            
            # 3. ASSERT: Verify the trajectory was written despite the interruption
            assert mock_write.called
            print("\n✅ PASS: Trajectory finalized successfully after simulated Ctrl+C.")