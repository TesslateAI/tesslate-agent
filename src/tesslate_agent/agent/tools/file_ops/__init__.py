"""
File Operations Module

Essential file operation tools for AI agents.

The tools registered here:
    * ``read_file`` / ``write_file``        (read_write.py)
    * ``read_many_files``                   (read_many.py)
    * ``patch_file`` / ``multi_edit``       (edit.py -- multi-strategy matcher)
    * ``apply_patch``                       (apply_patch_tool.py -- structured batch)
    * ``view_image``                        (view_image.py)
    * ``file_undo``                         (undo_tool.py)

``EDIT_HISTORY`` from :mod:`edit_history` is the shared ring buffer
every mutating tool records into so ``file_undo`` can walk it back.
"""

from __future__ import annotations

from .apply_patch_tool import register_apply_patch_tool
from .edit import register_edit_tools
from .edit_history import EDIT_HISTORY, EditHistory, EditHistoryEntry
from .read_many import register_read_many_files_tool
from .read_write import register_read_write_tools
from .undo_tool import register_undo_tool
from .view_image import register_view_image_tool


def register_file_ops_tools(registry) -> None:
    """Register the full file operations tool set on ``registry``."""
    register_read_write_tools(registry)
    register_read_many_files_tool(registry)
    register_edit_tools(registry)
    register_apply_patch_tool(registry)
    register_view_image_tool(registry)
    register_undo_tool(registry)


__all__ = [
    "register_file_ops_tools",
    "register_read_write_tools",
    "register_read_many_files_tool",
    "register_edit_tools",
    "register_apply_patch_tool",
    "register_view_image_tool",
    "register_undo_tool",
    "EDIT_HISTORY",
    "EditHistory",
    "EditHistoryEntry",
]
