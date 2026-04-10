"""
File Read/Write Tools

Tools for reading and writing files via the active orchestrator. Both
tools retry on transient I/O errors (exponential backoff, see
:mod:`tesslate_agent.agent.tools.retry_config`) while failing fast on
non-retryable conditions such as missing files or permission errors.
"""

from __future__ import annotations

import logging
from typing import Any

from tesslate_agent.agent.tools.output_formatter import (
    error_output,
    format_file_size,
    pluralize,
    success_output,
)
from tesslate_agent.agent.tools.registry import Tool, ToolCategory
from tesslate_agent.agent.tools.retry_config import tool_retry
from tesslate_agent.orchestration import get_orchestrator

logger = logging.getLogger(__name__)


@tool_retry
async def read_file_tool(
    params: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """
    Read a file from the active orchestrator.

    Args:
        params: ``{"file_path": str}``
        context: Standard tool context. Honored keys include
            ``user_id``, ``project_id``, ``project_slug``,
            ``container_directory``, ``container_name``, ``volume_id``,
            ``cache_node``.

    Returns:
        Standardized success dict with ``content`` and file metadata,
        or an error dict when the file could not be read.
    """
    file_path = params.get("file_path")
    if not file_path:
        raise ValueError("file_path parameter is required")

    user_id = context.get("user_id")
    project_id_raw = context.get("project_id")
    project_id = str(project_id_raw) if project_id_raw is not None else ""
    project_slug = context.get("project_slug")
    container_directory = context.get("container_directory")
    container_name = context.get("container_name")

    logger.info(
        "[READ-FILE] Reading '%s' - project_slug: %s, subdir: %s",
        file_path,
        project_slug,
        container_directory,
    )

    try:
        orchestrator = get_orchestrator()
        content = await orchestrator.read_file(
            user_id=user_id,
            project_id=project_id,
            container_name=container_name,
            file_path=file_path,
            project_slug=project_slug,
            subdir=container_directory,
            volume_id=context.get("volume_id"),
            cache_node=context.get("cache_node"),
        )

        if content is not None:
            return success_output(
                message=f"Read {format_file_size(len(content))} from '{file_path}'",
                file_path=file_path,
                content=content,
                details={"size_bytes": len(content), "lines": len(content.split("\n"))},
            )

    except (FileNotFoundError, PermissionError, ValueError, TypeError, KeyError):
        raise
    except Exception as exc:
        logger.error("[READ-FILE] Failed to read '%s': %s", file_path, exc)

    return error_output(
        message=f"File '{file_path}' does not exist",
        suggestion="Use a directory listing to browse available files before reading",
        exists=False,
        file_path=file_path,
    )


@tool_retry
async def write_file_tool(
    params: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """
    Write content to a file via the active orchestrator.

    Args:
        params: ``{"file_path": str, "content": str}``
        context: Same keys as :func:`read_file_tool`.

    Returns:
        Standardized success dict with a preview of the written content,
        or an error dict if the write failed.
    """
    file_path = params.get("file_path")
    content = params.get("content")

    if not file_path:
        raise ValueError("file_path parameter is required")
    if content is None:
        raise ValueError("content parameter is required")

    user_id = context.get("user_id")
    project_id_raw = context.get("project_id")
    project_id = str(project_id_raw) if project_id_raw is not None else ""
    project_slug = context.get("project_slug")
    container_directory = context.get("container_directory")
    container_name = context.get("container_name")

    lines = content.split("\n")
    preview_lines = 5

    if len(lines) <= preview_lines * 2:
        preview = content
    else:
        preview = (
            "\n".join(lines[:preview_lines])
            + "\n\n... ("
            + str(len(lines) - preview_lines * 2)
            + " lines omitted) ...\n\n"
            + "\n".join(lines[-preview_lines:])
        )

    logger.info(
        "[WRITE-FILE] Writing '%s' - project_slug: %s, subdir: %s",
        file_path,
        project_slug,
        container_directory,
    )

    try:
        orchestrator = get_orchestrator()
        success = await orchestrator.write_file(
            user_id=user_id,
            project_id=project_id,
            container_name=container_name,
            file_path=file_path,
            content=content,
            project_slug=project_slug,
            subdir=container_directory,
            volume_id=context.get("volume_id"),
            cache_node=context.get("cache_node"),
        )

        if success:
            return success_output(
                message=(
                    f"Wrote {pluralize(len(lines), 'line')} "
                    f"({format_file_size(len(content))}) to '{file_path}'"
                ),
                file_path=file_path,
                preview=preview,
                details={"size_bytes": len(content), "line_count": len(lines)},
            )

    except (PermissionError, ValueError, TypeError, KeyError):
        raise
    except Exception as exc:
        logger.error("[WRITE-FILE] Failed to write '%s': %s", file_path, exc)
        return error_output(
            message=f"Could not write to '{file_path}': {exc}",
            suggestion="Check if the directory exists and you have write permissions",
            file_path=file_path,
            details={"error": str(exc)},
        )

    return error_output(
        message=f"Failed to write to '{file_path}'",
        suggestion="Check permissions and available disk space",
        file_path=file_path,
    )


def register_read_write_tools(registry) -> None:
    """Register ``read_file`` and ``write_file`` tools on ``registry``."""

    registry.register(
        Tool(
            name="read_file",
            description=(
                "Read the contents of a file from the project directory. Always "
                "use this to read actual file content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Path to the file relative to project root "
                            "(e.g. 'src/App.jsx')"
                        ),
                    }
                },
                "required": ["file_path"],
            },
            executor=read_file_tool,
            category=ToolCategory.FILE_OPS,
            examples=[
                '{"tool_name": "read_file", "parameters": {"file_path": "package.json"}}',
                '{"tool_name": "read_file", "parameters": {"file_path": "src/components/Header.jsx"}}',
            ],
        )
    )

    registry.register(
        Tool(
            name="write_file",
            description=(
                "Write complete file content (creates if missing). Use patch_file "
                "or multi_edit for editing existing files to avoid token waste."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file relative to project root",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete content to write to the file",
                    },
                },
                "required": ["file_path", "content"],
            },
            executor=write_file_tool,
            category=ToolCategory.FILE_OPS,
            examples=[
                '{"tool_name": "write_file", "parameters": {"file_path": "src/NewComponent.jsx", "content": "import React from \'react\'..."}}'
            ],
        )
    )

    logger.info("Registered 2 read/write file tools")
