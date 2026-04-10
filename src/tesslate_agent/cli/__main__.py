"""
Tesslate Agent command-line entry point.

Usage:
    tesslate-agent run --task "..." --output trajectory.json
    tesslate-agent tools list
    tesslate-agent --version
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]

_VERSION = "tesslate-agent 0.1.0"

EXIT_ARGPARSE_ERROR = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tesslate-agent",
        description=(
            "Standalone autonomous coding agent with file, shell, "
            "git, memory, planning, web, and subagent tools."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the tesslate-agent version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Run the agent against a local working directory.",
    )
    run_parser.add_argument(
        "--task",
        required=True,
        help="Natural-language task description for the agent.",
    )
    run_parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="LiteLLM model identifier (default: openai/gpt-4o-mini).",
    )
    run_parser.add_argument(
        "--workdir",
        default=".",
        help="Directory the agent should treat as PROJECT_ROOT (default: .).",
    )
    run_parser.add_argument(
        "--output",
        required=True,
        help="Path to write the ATIF v1.4 trajectory JSON to.",
    )
    run_parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Maximum number of agent iterations (default: 30).",
    )
    run_parser.add_argument(
        "--effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Extended-thinking effort for supporting models (default: medium).",
    )
    run_parser.add_argument(
        "--tools",
        default=None,
        help="Optional comma-separated list of tool names to enable.",
    )
    run_parser.add_argument(
        "--timeout-ms",
        type=int,
        default=900_000,
        help="Wall-clock agent loop timeout in milliseconds (default: 900000).",
    )
    run_parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional override for the default system prompt.",
    )
    run_parser.add_argument(
        "--api-base",
        default=None,
        help="LiteLLM proxy base URL (overrides LITELLM_API_BASE).",
    )
    run_parser.add_argument(
        "--api-key",
        default=None,
        help="LiteLLM proxy master key (overrides LITELLM_MASTER_KEY).",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose (INFO-level) logging.",
    )
    run_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet (ERROR-level) logging.",
    )

    # tools
    tools_parser = subparsers.add_parser(
        "tools",
        help="Introspect the built-in tool set.",
    )
    tools_sub = tools_parser.add_subparsers(dest="tools_command")
    tools_sub.add_parser(
        "list",
        help="List every registered tool and its category.",
    )

    return parser


def _configure_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _print_event(event: dict[str, Any]) -> None:
    """Default human-readable event printer for ``run``."""
    etype = event.get("type", "")
    if etype == "stream":
        content = event.get("content", "")
        sys.stdout.write(str(content))
        sys.stdout.flush()
    elif etype == "agent_step":
        data = event.get("data") or {}
        tool_calls = data.get("tool_calls") or []
        if tool_calls:
            names = ", ".join(tc.get("name", "") for tc in tool_calls)
            print(f"\n[step {data.get('iteration', '?')}] tools: {names}")
    elif etype == "tool_result":
        data = event.get("data") or {}
        print(f"[tool] {data.get('name', '')} -> done")
    elif etype == "complete":
        data = event.get("data") or {}
        print(
            f"\n[complete] iterations={data.get('iterations', '?')} "
            f"success={data.get('success', '?')}"
        )
    elif etype == "error":
        print(f"\n[error] {event.get('content', '')}", file=sys.stderr)


def _cmd_run(args: argparse.Namespace) -> int:
    from tesslate_agent.cli.runner import run_agent

    _configure_logging(args.verbose, args.quiet)

    tool_names: list[str] | None = None
    if args.tools:
        tool_names = [name.strip() for name in args.tools.split(",") if name.strip()]
        if not tool_names:
            tool_names = None

    workdir = Path(args.workdir)
    output = Path(args.output)

    return asyncio.run(
        run_agent(
            task=args.task,
            model=args.model,
            workdir=workdir,
            output=output,
            max_iterations=args.max_iterations,
            effort=args.effort,
            tool_names=tool_names,
            timeout_ms=args.timeout_ms,
            system_prompt=args.system_prompt,
            event_printer=_print_event,
            api_base=args.api_base,
            api_key=args.api_key,
        )
    )


def _cmd_tools_list() -> int:
    # Pin an environment so the tool registry can be built without
    # tripping on missing PROJECT_ROOT when the user just wants to
    # enumerate tools.
    os.environ.setdefault("DEPLOYMENT_MODE", "local")
    os.environ.setdefault("PROJECT_ROOT", os.getcwd())

    from tesslate_agent.agent.tools.registry import get_tool_registry

    registry = get_tool_registry()
    for tool in sorted(registry.all_tools(), key=lambda t: t.name):
        print(f"{tool.name}\t{tool.category.value}")
    return 0


def main() -> None:
    """Entry point invoked by the ``tesslate-agent`` console script."""
    parser = _build_parser()

    try:
        args = parser.parse_args()
    except SystemExit as exc:
        # argparse exits with code 2 on parse errors; remap to our
        # documented argparse exit code of 3.
        if exc.code == 0:
            raise
        sys.exit(EXIT_ARGPARSE_ERROR)

    if args.version:
        print(_VERSION)
        sys.exit(0)

    if args.command == "run":
        sys.exit(_cmd_run(args))

    if args.command == "tools":
        if args.tools_command == "list":
            sys.exit(_cmd_tools_list())
        parser.parse_args(["tools", "--help"])
        sys.exit(EXIT_ARGPARSE_ERROR)

    parser.print_help()
    sys.exit(EXIT_ARGPARSE_ERROR)


if __name__ == "__main__":
    main()
