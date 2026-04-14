TOOLS.md — comprehensive tool & system reference (verbatim)



================================================================================



# Tesslate Agent — Tool & System Reference

> Single source of truth for what the Tesslate Agent can do, what the
> model sees, and every quirk that affects tool behavior.
> Generated from the live tool registry and source code.

---

## Table of contents

- [1. The agent at a glance](#1-the-agent-at-a-glance)
- [2. The agent loop](#2-the-agent-loop)
- [3. Edit modes & approval](#3-edit-modes--approval)
- [4. API key scopes](#4-api-key-scopes)
- [5. Tool result envelope](#5-tool-result-envelope)
- [6. The system prompt](#6-the-system-prompt)
- [7. Tool catalog](#7-tool-catalog)
  - [7.1 file_operations](#71-file_operations)
    - [read_file](#read_file)
    - [write_file](#write_file)
    - [read_many_files](#read_many_files)
    - [patch_file](#patch_file)
    - [multi_edit](#multi_edit)
    - [apply_patch](#apply_patch)
    - [view_image](#view_image)
    - [file_undo](#file_undo)
  - [7.2 shell_commands](#72-shell_commands)
    - [bash_exec](#bash_exec)
    - [shell_open](#shell_open)
    - [shell_close](#shell_close)
    - [shell_exec](#shell_exec)
    - [write_stdin](#write_stdin)
    - [list_background_processes](#list_background_processes)
    - [read_background_output](#read_background_output)
    - [python_repl](#python_repl)
  - [7.3 navigation_operations](#73-navigation_operations)
    - [glob](#glob)
    - [grep](#grep)
    - [list_dir](#list_dir)
  - [7.4 git_operations](#74-git_operations)
    - [git_log](#git_log)
    - [git_blame](#git_blame)
    - [git_status](#git_status)
    - [git_diff](#git_diff)
  - [7.5 memory_operations](#75-memory_operations)
    - [memory_read](#memory_read)
    - [memory_write](#memory_write)
  - [7.6 web_operations](#76-web_operations)
    - [web_fetch](#web_fetch)
    - [web_search](#web_search)
  - [7.7 planning_operations](#77-planning_operations)
    - [update_plan](#update_plan)
  - [7.8 delegation_operations](#78-delegation_operations)
    - [task](#task)
    - [wait_agent](#wait_agent)
    - [send_message_to_agent](#send_message_to_agent)
    - [close_agent](#close_agent)
    - [list_agents](#list_agents)
- [8. File editing strategy (deep dive)](#8-file-editing-strategy-deep-dive)
- [9. Shell sessions & PTY model](#9-shell-sessions--pty-model)
- [10. Memory store](#10-memory-store)
- [11. Plan store](#11-plan-store)
- [12. Subagents (delegation)](#12-subagents-delegation)
- [13. Web search & fetch](#13-web-search--fetch)
- [14. Trajectory format (ATIF v1.4)](#14-trajectory-format-atif-v14)
- [15. LLM adapter (LiteLLM)](#15-llm-adapter-litellm)
- [16. Orchestration backends](#16-orchestration-backends)
- [17. CLI flags](#17-cli-flags)
- [18. Quirks, gotchas, and design decisions](#18-quirks-gotchas-and-design-decisions)
- [19. Glossary](#19-glossary)

---

## 1. The agent at a glance

The **Tesslate Agent** is a self-contained, function-calling coding agent
that drives an LLM via LiteLLM's unified OpenAI-style tool contract. It is
distributed as the Python package `tesslate_agent` in the repo
[github.com/TesslateAI/tesslate-agent](https://github.com/TesslateAI/tesslate-agent)
and is also vendored in the Tesslate Studio monorepo at
`packages/tesslate-agent/`.

**What it is**

- A single class, `TesslateAgent`
  (`src/tesslate_agent/agent/tesslate_agent.py`), that runs a streaming
  iteration loop of `model → tool calls → results → model → …` until the
  model emits a turn with no tool calls.
- A fixed **33-tool** built-in registry across **8 categories**:
  `file_operations` (8), `shell_commands` (8), `nav_operations` (3),
  `git_operations` (4), `memory_operations` (2), `web_operations` (2),
  `planning_operations` (1), `delegation_operations` (5).
- One pluggable orchestration backend interface
  (`BaseOrchestrator`, `src/tesslate_agent/orchestration/base.py`). Only
  `LocalOrchestrator` ships in the open-source repo; downstream packages
  can register Docker / Kubernetes / Modal / Daytona backends by calling
  `OrchestratorFactory.register(mode, cls)`.
- A self-describing **ATIF v1.4** trajectory recorder
  (`src/tesslate_agent/agent/trajectory.py`) that is always written to
  disk on exit (success, error, timeout, interrupt).
- A CLI entrypoint (`tesslate-agent run`, `tesslate-agent tools list`) in
  `src/tesslate_agent/cli/__main__.py`.

**What it is not**

- Not coupled to a database, Redis, billing, credit system, lock manager,
  or the Tesslate Studio orchestrator service.
- Not a chat agent — there's no ambient conversation history unless the
  caller injects one through `context["chat_history"]`.
- Not hard-wired to any single LLM vendor. Any model LiteLLM can reach is
  usable, including OpenAI, Anthropic, OpenRouter, Groq, Together,
  DeepSeek, Fireworks, Gemini/Vertex, Mistral, Cohere, Perplexity, and
  Bedrock — or any provider reachable through a LiteLLM proxy.

**Where it runs**

- Locally from the CLI: `tesslate-agent run --task "…" --workdir . --output trajectory.json`
- Embedded in Tesslate Studio's orchestrator worker via the vendored
  package.
- As a benchmark runner (the CLI is tuned for long-horizon autonomous
  runs, not for interactive pair-programming).

---

## 2. The agent loop

Everything happens inside `TesslateAgent.run(user_request, context)`
(`src/tesslate_agent/agent/tesslate_agent.py`). It is an `async`
generator that yields streaming event dicts and never touches the
filesystem directly — every tool call goes through the `ToolRegistry`.

### Iteration model

```
loop (iteration = 1..∞):
    pre-flight compaction if token estimate ≥ context_window × threshold
    response = model.chat_with_tools(messages, tools)   # with retry
    if response.content:     yield {"type": "stream", ...}
    if not response.tool_calls:
        yield agent_step + complete
        return
    append assistant turn with tool_calls to history
    results = execute tool_calls (parallel-safe in gather, rest sequentially)
    for each tool_call:
        yield {"type": "tool_result", ...}
        append {"role": "tool", ...} to history
    yield agent_step
```

### Termination conditions

The loop terminates when **any** of the following fire:

1. The model response has `tool_calls == []` (normal completion).
2. `iteration > max_iterations` and `max_iterations > 0`.
3. The model call raises a non-retryable exception, or exceeds
   `MAX_RETRIES = 2` retries on a retryable exception.
4. The caller cancels the async task (e.g. CLI timeout via
   `asyncio.wait_for`).

### `DEFAULT_MAX_ITERATIONS = 0`

`DEFAULT_MAX_ITERATIONS` is **0**, which means **"no cap"**. The loop
runs until the model stops emitting tool calls, the host kills it, an
exception terminates it, or a provider-side rate limit / token budget
stops it. This is a deliberate design choice for long-horizon
autonomous runs — the agent is expected to bound itself, not to have a
hard-coded ceiling.

The CLI `--max-iterations` flag also defaults to `0`. The caller can
still pass any positive integer to bound the loop explicitly.

Source: `DEFAULT_MAX_ITERATIONS = 0` in
`src/tesslate_agent/agent/tesslate_agent.py`, and the argparse default
in `src/tesslate_agent/cli/__main__.py:70`.

### Streaming event shapes

`TesslateAgent.run` yields event dicts with a `type` discriminator. The
full set:

| type | Payload | When emitted |
|---|---|---|
| `stream` | `{"content": str}` | Model emitted text content on this iteration. Sent once per iteration, before tool execution. |
| `agent_step` | `{"data": {"iteration", "tool_calls", "tool_results", "response_text", "timestamp", "is_complete"}}` | After every iteration, even terminal ones. |
| `tool_result` | `{"data": {"iteration", "index", "total", "name", "parameters", "result", "timestamp"}}` | One per tool call, emitted between `agent_step` turns. Used by live UIs. The trajectory bridge intentionally **ignores** `tool_result` events to avoid double-recording. |
| `context_pressure` | `{"data": {"compacted": True, "iteration": N}}` | Emitted immediately after a successful history compaction. |
| `complete` | `{"data": {"success", "iterations", "final_response", "tool_calls_made", "completion_reason", "session_id", "usage", ...}}` | Terminal event. `completion_reason` is one of `stop`, `no_more_actions`, `max_iterations`, `model_error`, `missing_model_adapter`. |
| `error` | `{"content": str}` | Recoverable or terminal error, emitted before `complete`. |

### Parallel-safe tools

Within a single iteration the registered `tool_calls` are partitioned
into two groups:

- **Parallel** — read-only tools in `PARALLEL_SAFE_TOOLS`. Executed via
  `asyncio.gather`.
- **Sequential** — everything else. Executed one after another to
  preserve filesystem ordering.

The `PARALLEL_SAFE_TOOLS` frozenset from
`src/tesslate_agent/agent/tesslate_agent.py`:

```
read_file, list_files, glob_search, grep, grep_search,
web_fetch, web_search, metadata,
container_status, container_logs, container_health
```

Note the asymmetry: `glob` and the navigation tools in the built-in
registry are named `glob`, `grep`, `list_dir` — **not** `glob_search` or
`grep_search`. Only the tool called `grep` is actually in the
parallel-safe set. `glob` is **not** in the parallel-safe set as
currently implemented, so it runs sequentially. `list_files` is a name
from an earlier tool era and has no matching built-in tool today. This
is a historical quirk; do not rely on the exact names in
`PARALLEL_SAFE_TOOLS` without cross-checking.

### Tool result truncation (fed back to the model)

Every successful tool result is rendered into a `role: tool` message by
`format_tool_result` before being appended to the history. Long outputs
in any of the fields `content`, `stdout`, `output`, or `preview` are
truncated **in the middle** to at most `MAX_TOOL_OUTPUT = 10_000` chars:

```
{first 5000 chars}
... (N chars truncated) ...
{last 5000 chars}
```

This is a separate, harder cap than the per-tool `max_output_tokens`
budgets applied inside shell tools. It exists so a single giant tool
output cannot blow up the next `chat_with_tools` call.

Source: `MAX_TOOL_OUTPUT = 10_000` and `format_tool_result` in
`src/tesslate_agent/agent/tesslate_agent.py`.

### Model call retry

`_call_model_with_retry` retries `chat_with_tools` up to `MAX_RETRIES = 2`
times (i.e. up to **3 total attempts**) with exponential backoff plus
jitter:

```
delay_ms ≈ 200 * 2^attempt * random(0.9, 1.1)
```

A call is considered retryable when its error string contains any of:
`timeout`, `connection`, `temporary`, `transient`, `service unavailable`,
`bad gateway`, `gateway timeout`, `rate limit`, `stream`, `502`, `503`,
`504`, `429`. Source: `RETRYABLE_KEYWORDS` in `tesslate_agent.py`.

### Token estimation & compaction

Token usage is estimated at `~4 chars per token` across every message's
content and tool-call payloads (`_estimate_tokens`). When
`compaction_adapter is not None` and

```
estimated_tokens >= context_window_tokens * compaction_threshold
```

the middle of the message history is summarised by the cheap
`compaction_adapter` before the next model call. Defaults:

- `context_window_tokens = 128_000` (overridable in the constructor)
- `compaction_threshold = 0.8` (clamped to `[0.1, 0.99]`)

Compaction is implemented in `_compact_messages`:

1. Keep the leading system message verbatim (`messages[:1]`).
2. Keep the trailing **6** messages verbatim (`messages[-6:]`).
3. Send the middle to the compaction adapter with a "compact this
   conversation" system prompt. The middle is serialized as JSON and
   truncated to `60_000` chars before the summary call.
4. Replace the middle with a single `role: system` message whose body is
   `"[Conversation summary]\n<summary text>"`.

The standalone CLI runner does **not** currently pass a
`compaction_adapter`, so compaction is inert for CLI-driven runs. Embed
the agent and pass `compaction_adapter=<cheap model>` to enable it.

### Assistant message serialization

`serialize_assistant_message` follows the OpenAI tool-calling contract
strictly:

- When there are no `tool_calls`, the assistant message is
  `{"role": "assistant", "content": <str>}`.
- When there are `tool_calls`, `content` is **`None`** (not omitted, not
  `""`) and every call carries an explicit `"type": "function"`.

This is important when forwarding histories to any model provider that
LiteLLM routes to.

### Fallback stream collector

`_collect_stream` exists as a defensive net: if a model adapter
mis-implements the contract and returns an async iterator for
`stream=False`, the iterator is drained into the non-streaming dict
shape. No known adapter currently needs this, but the code path is there.

---

## 3. Edit modes & approval

Every tool execution is gated by the `edit_mode` value in the
`context` dict. Three modes are supported
(`src/tesslate_agent/agent/tools/registry.py`):

| Mode | Dangerous tool behavior | Default where |
|---|---|---|
| `auto` | Executed unconditionally. | **Standalone CLI default** (`src/tesslate_agent/cli/runner.py:171`). |
| `ask` | Dangerous tools must be approved. | `ToolRegistry.execute` default (`edit_mode = context.get("edit_mode", "ask")`). Tesslate Studio hosts use this mode. |
| `plan` | Dangerous tools blocked with a structured error, unless they are listed in `PLAN_MODE_ALLOWED`. Read-only tools still run. | Explicit opt-in by the caller. |

### `DANGEROUS_TOOLS`

Verbatim from `registry.py`:

```python
DANGEROUS_TOOLS = frozenset({
    "write_file",
    "patch_file",
    "multi_edit",
    "apply_patch",
    "bash_exec",
    "shell_exec",
    "shell_open",
    "web_fetch",
    "web_search",
    "send_message",
})
```

Note that `send_message` is not in the 33-tool built-in registry — it
lives in downstream packages. It is listed here so if a host ever
registers it, it is automatically gated.

### `PLAN_MODE_ALLOWED`

Only one dangerous tool is permitted in plan mode:

```python
PLAN_MODE_ALLOWED = frozenset({"bash_exec"})
```

The rationale is that the agent must be able to explore the repo (run
shell commands, inspect git state, execute read-only checks) to build a
plan, but every mutation primitive is suppressed. In plan mode the agent
is expected to explain what it would change, not actually change it.

### Ask-mode flow

In `ask` mode, the first attempt to call a dangerous tool triggers the
`ApprovalManager`
(`src/tesslate_agent/agent/tools/approval_manager.py`):

1. `ToolRegistry.execute` checks
   `approval_mgr.is_tool_approved(session_id, tool_name)`. Session IDs
   come from `context["chat_id"]`, falling back to the literal string
   `"default"`.
2. If not previously approved, `request_approval` is called. The
   approval is resolved **synchronously** in the constructor based on
   the `TESSLATE_AGENT_APPROVAL_POLICY` environment variable:
   - `allow` (default) → every request resolves to `allow_once`.
   - `deny` → every request resolves to `stop`.
3. If the resolved response is anything other than `allow_once` or
   `allow_all`, the tool returns
   `{"approval_required": True, "tool", "parameters", "session_id", "approval_id", "response"}`
   instead of executing. The agent loop forwards this to the model as the
   text `"Awaiting approval for <tool>"`.
4. If the response is `allow_all`, subsequent calls to the same tool in
   the same `session_id` skip the approval gate (stored in
   `ApprovalManager._approved_tools`).

Approval state lives in a **process-local, in-memory** singleton. It is
**not persisted** across CLI invocations or process restarts. The
`respond_to_approval` method is a compatibility shim for older callers
and is a no-op under the synchronous policy model.

### Skip-approval escape hatch

Callers can set `context["skip_approval_check"] = True` to bypass the
approval gate regardless of `edit_mode`. This is what the `task` tool
does when spawning a subagent (`_build_child_context` in
`delegation_ops/task_tool.py`), because the child runs in the same trust
domain as the parent.

---

## 4. API key scopes

The registry honours an optional `context["api_key_scopes"]` list. When
present, every tool call is checked against `TOOL_REQUIRED_SCOPES`. Tools
not listed there are unrestricted.

Verbatim from `registry.py:TOOL_REQUIRED_SCOPES`:

| Tool | Required scope |
|---|---|
| `write_file` | `file.write` |
| `patch_file` | `file.write` |
| `multi_edit` | `file.write` |
| `apply_patch` | `file.write` |
| `delete_file` | `file.delete` |
| `bash_exec` | `terminal.access` |
| `shell_exec` | `terminal.access` |
| `shell_open` | `terminal.access` |
| `shell_close` | `terminal.access` |
| `web_fetch` | `file.read` |
| `web_search` | `file.read` |
| `send_message` | `channel.manage` |
| `container_status` | `container.view` |
| `container_restart` | `container.start_stop` |
| `container_logs` | `container.view` |
| `container_health` | `container.view` |
| `kanban_create` | `kanban.edit` |
| `kanban_move` | `kanban.edit` |
| `kanban_update` | `kanban.edit` |
| `kanban_comment` | `kanban.edit` |

When a required scope is missing from `api_key_scopes`, the registry
returns
`{"success": False, "tool": <name>, "error": "API key scope restriction: …"}`
**without** invoking the tool.

`delete_file`, `send_message`, the container tools, and the kanban
tools are **not** in the 33-tool built-in registry — they ship in the
Tesslate Studio fork. The entries are kept here so downstream packages
don't need to redeclare scopes.

Read-only tools (`read_file`, `read_many_files`, `glob`, `grep`,
`list_dir`, `git_*`, `memory_read`, `memory_write`, `update_plan`,
`view_image`, `file_undo`, delegation tools, `python_repl`, etc.) have
**no** scope requirement.

---

## 5. Tool result envelope

Every tool invocation returns two nested envelopes:

### Outer envelope (from `ToolRegistry.execute`)

```json
{
  "success": true,
  "tool": "read_file",
  "result": { ... per-tool payload ... }
}
```

On error:

```json
{
  "success": false,
  "tool": "read_file",
  "error": "File 'x' does not exist"
}
```

When approval is required (ask mode):

```json
{
  "approval_required": true,
  "tool": "write_file",
  "parameters": { ... },
  "session_id": "default",
  "approval_id": "<uuid>",
  "response": "stop"
}
```

### Inner payload (from the tool executor)

All 33 built-in tools route their output through
`success_output` / `error_output` in
`src/tesslate_agent/agent/tools/output_formatter.py`. Success has the
shape:

```json
{
  "success": true,
  "message": "<user-friendly>",
  "details": { ... optional technical ... },
  "<extra fields merged at top level>": "..."
}
```

Error has the shape:

```json
{
  "success": false,
  "message": "<user-friendly>",
  "suggestion": "<optional actionable>",
  "details": { ... },
  "<extra fields>": "..."
}
```

`ToolRegistry.execute` treats the inner `success` flag as authoritative:
if the inner payload is a dict with `success=False`, the outer envelope
also sets `success=False` (and keeps the payload under `result`).

### Feeding results back to the model

`format_tool_result` in `tesslate_agent.py` renders the envelope into a
`role: tool` message for the next model turn. Its field priority:

1. `result.message` — the human-readable line.
2. `result.content`, `result.stdout`, `result.output`, `result.preview`
   (in this order, whichever are present) — **truncated to 10 KB** via
   middle-elision.
3. `result.files` — rendered as `"Files (N items): [first 20 ...]"`.
4. `result.stderr` — appended as `"stderr: <value>"`.

If the envelope says `approval_required`, the model is told
`"Awaiting approval for <tool>"`. If `success=False`, the model sees
`"Error: <error>\nSuggestion: <suggestion>"`.

The `suggestion`, `details`, and most other keys are **not** forwarded
verbatim — they are only visible to callers that consume the raw
`tool_result` events (live UIs, the trajectory recorder).

---

## 6. The system prompt

The agent does **not** inject tool descriptions into the system prompt.
Tool definitions are passed as OpenAI function-tool schemas via the
`tools=` argument of `ModelAdapter.chat_with_tools`. The model sees
each tool only through those schemas plus any `system_prompt` field on
the `Tool` dataclass (currently empty for all 33 tools).

The standalone CLI uses `DEFAULT_BENCHMARK_SYSTEM_PROMPT` from
`src/tesslate_agent/cli/prompts.py`. Verbatim:

```text
You are Tesslate Agent, a senior engineer operating autonomously inside a
local working directory. You have full access to a filesystem, a shell,
a git repository, and a persistent memory store through your tool set.
There is no human reviewing each step. Your job is to take the task
described in the user turn and drive it to a verifiable, working
solution by yourself.

## Operating principles

- Treat every task as a professional code change. Read the relevant
  files, understand the existing conventions, and match them. Do not
  reformat code you are not touching. Do not introduce unrelated
  refactors. Do not leave unfinished markers, dummy values, or
  commented-out code behind.
- Prefer small, reversible steps. Make one logical change, verify it,
  then move to the next. When a change is larger than a single file,
  plan it out before touching anything.
- Work from evidence, not assumption. Run the existing tests, inspect
  the actual file contents, and check the real behaviour of the system
  before deciding what to change.
- If you genuinely cannot make progress — for example because the task
  is ambiguous, credentials are missing, or a required tool fails —
  stop, explain why, and return the best partial result you have.

## Planning

- For anything non-trivial, call `update_plan` with a short ordered
  list of steps before you start editing code. Update the plan as you
  complete each step. A plan is not a status report; it is a living
  checklist that reflects what still has to happen.
- For smaller tasks you may skip the plan, but you must still think
  through the steps before touching the filesystem.

## Exploration

- Before opening any file, orient yourself. Use `list_dir` to see the
  project layout, `glob` to find files by name pattern, and `grep` to
  find definitions, usages, or configuration keys by content.
- Use `read_many_files` when you need to load several related files in
  one turn. It is much more efficient than reading them one at a time.
- Only read a single file with `read_file` once you know which file you
  want and why.

## Editing

- Use `patch_file` or `multi_edit` for targeted edits. These tools are
  purpose-built for precise substitutions and diff-style changes — use
  them instead of rewriting whole files.
- Only use `write_file` when you are creating a brand-new file or
  genuinely need to replace the entire contents of an existing file.
- Before editing a file you have not read this turn, read it first.
  Never guess at surrounding context.
- After editing, re-read or `grep` the changed region to confirm the
  change landed the way you expected.

## Shell and verification

- Use `bash_exec` for one-off commands: running tests, installing
  dependencies, inspecting processes, checking file metadata. Always
  capture both stdout and stderr and read the output before deciding
  what to do next.
- Use `shell_open` / `shell_exec` / `shell_close` only when you need a
  persistent session — for example, when running a long-lived dev
  server or a REPL that has to keep state between commands.
- After any non-trivial code change, run the project's test suite or
  the narrowest relevant test you can find. If tests do not exist,
  exercise the code path you touched through a direct invocation.
- Do not declare a task complete without at least one verification
  step that exercises the code you changed.

## Memory and git

- Use `memory_read` at the start of a fresh task to pick up any
  persistent notes about the project. Use `memory_write` to record
  durable facts you discovered that would help on the next run.
- Use `git_status`, `git_diff`, `git_log`, and `git_blame` to
  understand what has already happened in the repo and to produce a
  clean diff when you are done.

## Finishing

- When the task is done, return a concise summary covering what you
  changed, how you verified it, and anything you deliberately left out
  of scope. Do not dump large diffs into the final message — the
  reviewer will look at the files directly.
- Never claim a task is done if your verification step failed.
```

### Memory preamble injection

The CLI runner concatenates the persistent project memory file to the
base system prompt at startup:

```python
base_prompt = system_prompt or DEFAULT_BENCHMARK_SYSTEM_PROMPT
memory_prefix = load_memory_prefix(resolved_workdir)
if memory_prefix:
    sys_prompt = f"{base_prompt}\n\n{memory_prefix}"
```

`load_memory_prefix` reads `<workdir>/.tesslate/memory.md` and wraps it
in a `---\n## Persistent Memory\n\n<body>\n\n---` block. If the file is
missing or empty the prefix is an empty string. Global memory
(`~/.tesslate/memory.md`) is **not** auto-injected — the agent must read
it explicitly via `memory_read` with `scope="global"`.

### Per-tool `system_prompt`

`Tool.system_prompt` is a per-tool extra instruction that
`to_prompt_format` would splice into a prompt-style listing. The field
is **unused** in the current agent loop — tools are only ever registered
via OpenAI schemas. All 33 built-in tools ship with an empty
`system_prompt`.

---

## 7. Tool catalog

Every schema in this section is copied verbatim from the live tool
registry dump. Descriptions are the **exact strings** the model sees via
`_tool_to_openai`. Return shapes are read from the source files; each
entry cites the path.

### 7.1 file_operations

Eight tools live under `src/tesslate_agent/agent/tools/file_ops/`.

#### `read_file`

**Category:** `file_operations`

**Description (verbatim):** `Read the contents of a file from the project directory. Always use this to read actual file content.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Path to the file relative to project root (e.g. 'src/App.jsx')"
    }
  },
  "required": ["file_path"]
}
```

**Returns (from `file_ops/read_write.py:read_file_tool`):**

Success:

```json
{
  "success": true,
  "message": "Read 12.3 KB from 'src/App.jsx'",
  "file_path": "src/App.jsx",
  "content": "<full file text>",
  "details": {"size_bytes": 12567, "lines": 342}
}
```

Error (file missing):

```json
{
  "success": false,
  "message": "File 'src/App.jsx' does not exist",
  "suggestion": "Use a directory listing to browse available files before reading",
  "exists": false,
  "file_path": "src/App.jsx"
}
```

**When to use:** Once you know the exact path. Prefer `read_many_files`
for batch reads; prefer `glob` / `grep` to locate unknown files first.

**Quirks & gotchas:**

- Wrapped in `@tool_retry` (tenacity, 3 attempts, 1 s → 10 s exponential
  backoff). Retries on `ConnectionError` / `TimeoutError`; never retries
  on `FileNotFoundError`, `PermissionError`, `NotADirectoryError`,
  `IsADirectoryError`, `ValueError`, `TypeError`, `KeyError`, or
  `AttributeError`.
- Reads bytes as UTF-8; falls back to **latin-1** on `UnicodeDecodeError`
  so the agent never crashes on mixed-encoding files
  (`LocalOrchestrator.read_file`).
- Path resolution is root-safe: any `..` escape or symlink escape is
  refused by `_safe_resolve` in `orchestration/local.py`.
- Runs in parallel within an iteration (`PARALLEL_SAFE_TOOLS`).
- **Does not** honour `.gitignore` — file reads are unconditional.

**Example call:**

```json
{"name": "read_file", "arguments": {"file_path": "package.json"}}
```

**Related:** `read_many_files`, `write_file`, `patch_file`, `view_image`.

---

#### `write_file`

**Category:** `file_operations`

**Description (verbatim):** `Write complete file content (creates if missing). Use patch_file or multi_edit for editing existing files to avoid token waste.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Path to the file relative to project root"
    },
    "content": {
      "type": "string",
      "description": "Complete content to write to the file"
    }
  },
  "required": ["file_path", "content"]
}
```

**Returns (from `file_ops/read_write.py:write_file_tool`):**

```json
{
  "success": true,
  "message": "Wrote 42 lines (1.2 KB) to 'src/NewComponent.jsx'",
  "file_path": "src/NewComponent.jsx",
  "preview": "<head 5 lines>\n\n... (N lines omitted) ...\n\n<tail 5 lines>",
  "details": {"size_bytes": 1234, "line_count": 42}
}
```

The `preview` field shows the full content when ≤ 10 lines, otherwise
the first 5 and last 5 lines joined by `"\n\n... (N lines omitted) ...\n\n"`.

**When to use:** Brand-new files, or rewrites where the entire contents
must change. For edits to existing files, **prefer** `patch_file` or
`multi_edit`.

**Quirks & gotchas:**

- **DANGEROUS** → blocked in plan mode; requires approval in ask mode.
- Requires `file.write` scope if `api_key_scopes` is set.
- Wrapped in `@tool_retry`.
- `LocalOrchestrator.write_file` performs **atomic writes** via
  `tempfile.NamedTemporaryFile` in the same directory + `os.fsync` +
  `os.replace`. Concurrent readers never observe a partial write.
- Existing file mode (0o777 bits) is preserved when overwriting.
- Parent directories are created automatically (`mkdir(parents=True)`).
- Path containment is enforced — escapes are refused with a logged
  warning.
- Does **not** record to `EDIT_HISTORY` automatically (only the
  `patch_file`, `multi_edit`, `apply_patch` tools record history, so
  `file_undo` only walks back surgical edits, not raw writes).

**Example call:**

```json
{"name": "write_file", "arguments": {"file_path": "README.md", "content": "# Project\n"}}
```

**Related:** `patch_file`, `multi_edit`, `apply_patch`, `file_undo`.

---

#### `read_many_files`

**Category:** `file_operations`

**Description (verbatim):** `Batch-read many files at once using glob patterns. Applies per-file and total byte budgets to keep results bounded, and automatically skips binaries, lockfiles, build output, virtualenvs, and other noise unless use_default_excludes is disabled.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "include": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Glob patterns of files to include (required, non-empty)"
    },
    "exclude": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Glob patterns to exclude"
    },
    "recursive": {
      "type": "boolean",
      "description": "Kept for API parity; recursion is controlled via '**' in patterns"
    },
    "use_default_excludes": {
      "type": "boolean",
      "description": "Apply the baseline exclude set (default: true)"
    },
    "file_filtering_options": {
      "type": "object",
      "description": "{respect_gitignore?, respect_tesslate_ignore?}"
    },
    "max_bytes_per_file": {
      "type": "integer",
      "description": "Per-file byte budget (default: 65536)"
    },
    "max_total_bytes": {
      "type": "integer",
      "description": "Total byte budget across all files (default: 1048576)"
    }
  },
  "required": ["include"]
}
```

**Returns (from `file_ops/read_many.py:read_many_files_tool`):**

```json
{
  "success": true,
  "message": "Read 12 files (124.3 KB), skipped 3 files",
  "files": [
    {"path": "src/a.py", "content": "...", "lines": 42, "size": 1234, "truncated": false}
  ],
  "skipped": [{"path": "image.png", "reason": "binary file extension"}],
  "total_bytes": 127334,
  "truncated_overall": false,
  "details": {"include": ["**/*.py"], "exclude": [], "use_default_excludes": true, "max_bytes_per_file": 65536, "max_total_bytes": 1048576}
}
```

**Quirks & gotchas:**

- Default budgets: `65 536 B` per file, `1 048 576 B` (1 MiB) total.
- Reads via `orchestrator.list_tree` + `read_files_batch` (batch size
  **32**) so `.gitignore` and the baseline excluded directories
  (`node_modules`, `.git`, `dist`, `build`, `__pycache__`, `.venv`, etc.)
  are already filtered by the orchestrator.
- `use_default_excludes=True` (the default) additionally filters out
  lockfiles (`package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`),
  `*.min.js`, `*.map`, archives (`.zip`, `.tar*`, `.7z`), `.log`,
  `.lock`, and object binaries. Full list in
  `DEFAULT_EXCLUDE_PATTERNS`.
- Binary file extensions are **always** skipped regardless of
  `use_default_excludes` (see `BINARY_EXTENSIONS`: images, audio, video,
  archives, compiled artifacts, fonts, SQLite DBs, etc.).
- Per-file byte truncation is applied via UTF-8 byte slicing with
  replacement chars — decoded content may end with a replacement char on
  boundary-crossing truncation.
- `recursive`, `file_filtering_options.respect_gitignore`, and
  `file_filtering_options.respect_tesslate_ignore` are **accepted for
  API parity but ignored** today (`.gitignore` is always honoured via
  `list_tree`).
- Returns are sorted lexicographically by path.
- Not wrapped in `@tool_retry` (unlike `read_file` / `write_file`).

**Example call:**

```json
{"name": "read_many_files", "arguments": {"include": ["src/**/*.ts", "src/**/*.tsx"], "exclude": ["**/*.test.ts"]}}
```

**Related:** `read_file`, `glob`, `grep`.

---

#### `patch_file`

**Category:** `file_operations`

**Description (verbatim):** `Apply a surgical edit to an existing file using search/replace. Uses a multi-strategy matcher (exact -> whitespace-flexible -> Levenshtein fuzzy) with optional LLM repair for failed matches.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "file_path": {"type": "string", "description": "Path to the file relative to project root"},
    "old_str": {"type": "string", "description": "Exact text to find (include 3-5 lines of context for uniqueness)."},
    "new_str": {"type": "string", "description": "Replacement text."},
    "expected_occurrence": {"type": "integer", "description": "Number of occurrences you expect to replace. Defaults to 1."},
    "allow_multiple": {"type": "boolean", "description": "When true, replace every match regardless of count."}
  },
  "required": ["file_path", "old_str", "new_str"]
}
```

**Returns (from `file_ops/edit.py:patch_file_tool`):**

```json
{
  "success": true,
  "message": "Patched 'src/App.jsx' via exact match",
  "file_path": "src/App.jsx",
  "diff": "@@ -10,3 +10,3 @@\n-bg-blue-500\n+bg-green-500\n ...",
  "details": {
    "strategy": "exact",
    "match_method": "exact",
    "occurrences": 1,
    "repair_applied": false,
    "size_bytes": 1234
  }
}
```

On failure the error details include `attempted_strategies` and a
targeted `suggestion` (see [Section 8](#8-file-editing-strategy-deep-dive)).

**Quirks & gotchas:**

- **DANGEROUS**, scope `file.write`, plan-mode blocked.
- Three-strategy matcher: exact → flexible whitespace → Levenshtein
  fuzzy (threshold `0.10`, minimum needle length `10`, unique-best
  tiebreaker). See Section 8 for full details.
- Optional LLM repair pass — enabled unless
  `context["disable_llm_repair"]` is truthy. Repair uses the model from
  `TESSLATE_REPAIR_MODEL`, falling back to `COMPACTION_SUMMARY_MODEL` or
  `openai/gpt-4o-mini`.
- Accepts legacy field names `search` / `replace` as synonyms for
  `old_str` / `new_str`.
- Refuses any `new_str` containing an omission placeholder line
  (`...`, `// ...`, `# ...`) as the entire line.
- Before mutating, records the prior file content into `EDIT_HISTORY`
  so `file_undo` can walk back the change.
- Returns a unified-diff preview (up to 10 hunk lines then elided) in
  the `diff` field.
- Wrapped in `@tool_retry`.

**Example call:**

```json
{"name": "patch_file", "arguments": {"file_path": "src/App.jsx", "old_str": "className=\"bg-blue-500\"", "new_str": "className=\"bg-green-500\""}}
```

**Related:** `multi_edit`, `apply_patch`, `file_undo`, `write_file`.

---

#### `multi_edit`

**Category:** `file_operations`

**Description (verbatim):** `Apply multiple search/replace edits to a single file in sequence. Each edit runs through the full multi-strategy matcher and sees the buffer produced by the previous edit.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "file_path": {"type": "string", "description": "Path to the file relative to project root"},
    "edits": {
      "type": "array",
      "description": "List of search/replace operations, applied in order.",
      "items": {
        "type": "object",
        "properties": {
          "old_str": {"type": "string"},
          "new_str": {"type": "string"},
          "expected_occurrence": {"type": "integer"},
          "allow_multiple": {"type": "boolean"}
        },
        "required": ["old_str", "new_str"]
      }
    }
  },
  "required": ["file_path", "edits"]
}
```

**Returns (from `file_ops/edit.py:multi_edit_tool`):**

```json
{
  "success": true,
  "message": "Applied 3 edit(s) to 'src/App.jsx'",
  "file_path": "src/App.jsx",
  "diff": "...",
  "details": {
    "edit_count": 3,
    "applied": [
      {"index": 0, "strategy": "exact", "occurrences": 1, "repair_applied": false}
    ],
    "applied_edits": [...],
    "size_bytes": 1234
  }
}
```

**Quirks & gotchas:**

- **DANGEROUS**, scope `file.write`, plan-mode blocked.
- Each edit is applied against the **buffer produced by the previous
  edit**, not the original file.
- Each successful edit records a separate `EDIT_HISTORY` entry so undo
  walks the edits one at a time.
- If an edit fails mid-batch, the prior edits are left in the buffer,
  the latest `EDIT_HISTORY` entry is popped, and the tool returns an
  error describing which index failed with the list of applied edits so
  far.
- **Important:** a mid-batch failure does **not** roll back the file on
  disk — only the pre-first-edit snapshot is captured. Use `apply_patch`
  if you need all-or-nothing semantics.
- Wrapped in `@tool_retry`.

**Example call:**

```json
{
  "name": "multi_edit",
  "arguments": {
    "file_path": "src/App.jsx",
    "edits": [
      {"old_str": "useState(0)", "new_str": "useState(10)"},
      {"old_str": "bg-blue-500", "new_str": "bg-green-500"}
    ]
  }
}
```

**Related:** `patch_file`, `apply_patch`, `file_undo`.

---

#### `apply_patch`

**Category:** `file_operations`

**Description (verbatim):** `Apply a batch of file changes atomically. Supports create / update / delete / move operations. Every change is validated in-memory first; if any validation fails nothing is written. Update operations use the same multi-strategy matcher as patch_file.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "cwd": {
      "type": "string",
      "description": "Project-relative base directory for all paths. Use '' for the project root."
    },
    "changes": {
      "type": "array",
      "description": "List of structured change operations.",
      "items": {
        "type": "object",
        "properties": {
          "op": {"type": "string", "enum": ["create", "delete", "move", "update"]},
          "path": {"type": "string"},
          "from": {"type": "string"},
          "content": {"type": "string"},
          "old_str": {"type": "string"},
          "new_str": {"type": "string"},
          "expected_occurrence": {"type": "integer"},
          "allow_multiple": {"type": "boolean"}
        },
        "required": ["op"]
      }
    }
  },
  "required": ["changes"]
}
```

**Returns (from `file_ops/apply_patch_tool.py`):**

```json
{
  "success": true,
  "message": "apply_patch applied 3 change(s)",
  "details": {
    "applied": [
      {"index": 0, "op": "create", "path": "new.md", "status": "ok"},
      {"index": 1, "op": "update", "path": "old.md", "status": "ok", "strategy": "exact", "repair_applied": false}
    ]
  }
}
```

On validation failure:

```json
{
  "success": false,
  "message": "apply_patch validation failed: 2 error(s), nothing was written",
  "suggestion": "Fix the per-change errors and retry. No filesystem mutations were applied.",
  "details": {"errors": [{"index": 0, "op": "update", "path": "x", "error": "...", "attempted_strategies": [...]}]}
}
```

**Operations:**

| op | Required fields | Behavior |
|---|---|---|
| `create` | `path`, `content` | Fails if destination already exists. |
| `update` | `path`, `old_str`, `new_str` (+ optional `expected_occurrence`, `allow_multiple`) | Same matcher as `patch_file`; LLM repair **always** enabled. |
| `delete` | `path` | Fails if target does not exist. |
| `move` | `from`, `path` (as `to`) | Fails if source missing or destination exists. |

**Quirks & gotchas:**

- **DANGEROUS**, scope `file.write`, plan-mode blocked.
- **Two-phase commit**: Phase 1 validates every change in memory. Phase
  2 applies them sequentially against the orchestrator. If phase 1 finds
  **any** error, **nothing** is written.
- Phase 2 is **not** atomic: if a write fails mid-batch, any previously
  applied changes stay applied. The error suggests using `file_undo` to
  walk back the partial mutations.
- `cwd` is a path relative to the project root (`""` = root). All paths
  inside changes are resolved relative to `cwd`.
- Path escape attempts (`..` beyond root) are refused.
- A single batch cannot touch the same path twice — duplicates are
  reported as validation errors.
- Every applied change records to `EDIT_HISTORY` so `file_undo` can
  walk each step.
- `delete` prefers `orchestrator.delete_file`; if the orchestrator lacks
  that method it falls back to shelling out `rm -f <path>`.
- `move` is implemented as write-then-delete (content is copied to the
  destination, then the source is deleted). **Not** a rename — file
  metadata is reset on the destination.
- Wrapped in `@tool_retry`.

**Example call:**

```json
{
  "name": "apply_patch",
  "arguments": {
    "cwd": "src",
    "changes": [
      {"op": "create", "path": "new.ts", "content": "export const X = 1;\n"},
      {"op": "update", "path": "App.tsx", "old_str": "bg-blue-500", "new_str": "bg-green-500"},
      {"op": "delete", "path": "old.ts"}
    ]
  }
}
```

**Related:** `patch_file`, `multi_edit`, `file_undo`.

---

#### `view_image`

**Category:** `file_operations`

**Description (verbatim):** `Load an image file from the project and attach it to the next model turn. Supports PNG, JPG/JPEG, GIF, and WEBP up to 20 MiB.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "path": {"type": "string", "description": "Path to the image file, relative to the project root."},
    "detail": {
      "type": "string",
      "enum": ["auto", "high", "low"],
      "description": "Requested rendering detail hint for the model."
    }
  },
  "required": ["path"]
}
```

**Returns (from `file_ops/view_image.py`):**

```json
{
  "success": true,
  "message": "Loaded image 'design/mockup.png' (123456 bytes, image/png)",
  "file_path": "design/mockup.png",
  "content_parts": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "<base64>"}},
    {"type": "text", "text": "Image loaded: design/mockup.png (123456B)"}
  ],
  "details": {"size_bytes": 123456, "media_type": "image/png", "detail": "auto"}
}
```

**Quirks & gotchas:**

- Supported extensions: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`
  (strict — `.svg`, `.bmp`, `.tiff`, `.ico` etc. are rejected).
- Hard limit: **20 MiB** (`MAX_IMAGE_BYTES = 20 * 1024 * 1024`).
- **Vision gating:** if `context["model_supports_vision"]` is explicitly
  `False`, the tool short-circuits with an error. Any other value
  (including `None` — the default) allows the call through. The CLI
  never sets this flag, so the agent can always call `view_image`
  regardless of whether the current model actually understands images.
- On `LocalOrchestrator` the file is read directly off disk (exact
  bytes); on container backends it shells out `base64 -w0 <path>` via
  `execute_command` and decodes the result.
- `content_parts` is returned for callers that want to attach the image
  to the next turn. The **built-in agent loop does not forward
  `content_parts` back to the model** — the trajectory recorder and
  live UI see it, but the next `chat_with_tools` call only sees the
  truncated text `message`. Wiring `view_image` through to an actual
  multimodal model requires the embedding host to intercept
  `tool_result` events and splice `content_parts` into the next user
  turn.
- Not wrapped in `@tool_retry`.

**Example call:**

```json
{"name": "view_image", "arguments": {"path": "assets/logo.jpg", "detail": "high"}}
```

**Related:** `read_file`.

---

#### `file_undo`

**Category:** `file_operations`

**Description (verbatim):** `Revert the most recent mutation (write/edit/patch/delete/move) applied to a file by the agent in this run. Only undoes a single step -- call again to walk further back.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Path of the file to revert (relative to project root)."
    }
  },
  "required": ["file_path"]
}
```

**Returns (from `file_ops/undo_tool.py`):**

```json
{
  "success": true,
  "message": "Reverted 'src/App.jsx' to previous state",
  "file_path": "src/App.jsx",
  "details": {"op": "edit", "timestamp": 1718..., "restored_bytes": 1234, "action": "restore"}
}
```

If the recorded `prev_content` is `None` (the file was created in this
run), the action is `"delete"` and the message says
`"Reverted 'X' (deleted -- file did not previously exist)"`.

**Quirks & gotchas:**

- Walks **one step** at a time. Call repeatedly to undo further.
- Reads `EDIT_HISTORY` — a **process-local FIFO ring buffer** with
  capacity `100` (`DEFAULT_CAPACITY` in
  `file_ops/edit_history.py`). Mutations beyond the capacity are
  evicted and cannot be undone.
- Only `patch_file`, `multi_edit`, and `apply_patch` record to
  `EDIT_HISTORY`. **`write_file` does not record**, so you cannot undo a
  raw write.
- History is **ephemeral** — cleared between runs and not persisted.
- Undoing a `delete` recreates the file with the pre-delete content.
- The entry is **popped** from the buffer when you undo — re-running
  the undo walks to the next most recent entry on that path.
- Not DANGEROUS (no scope required, plan-mode allowed).

**Example call:**

```json
{"name": "file_undo", "arguments": {"file_path": "src/App.jsx"}}
```

**Related:** `patch_file`, `multi_edit`, `apply_patch`.

---

### 7.2 shell_commands

Eight tools live under `src/tesslate_agent/agent/tools/shell_ops/`. All
shell tools run commands through the `PTY_SESSIONS` registry
(`orchestration/local.py:PtySessionRegistry`), which spawns
`ptyprocess.PtyProcess` subprocesses in their own process group with a
`(24, 120)` terminal size.

#### `bash_exec`

**Category:** `shell_commands`

**Description (verbatim):** `Execute a bash/sh command under a PTY session and return its output. Supports soft yielding via yield_time_ms, idle detection via idle_timeout_ms, background spawning via is_background=True, and output truncation via max_output_tokens.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "command": {"type": "string", "description": "Command to execute (e.g., 'npm install', 'ls -la', 'cat package.json')."},
    "cwd": {"type": "string", "description": "Working directory relative to the project root. Defaults to the project root itself."},
    "timeout": {"type": "integer", "description": "Hard timeout in seconds — the process group is killed when it elapses (default: 120).", "default": 120},
    "timeout_ms": {"type": "integer", "description": "Alternative hard timeout expressed in milliseconds. When provided, overrides ``timeout``."},
    "yield_time_ms": {"type": "integer", "description": "Soft yield window in milliseconds. If the command is still running after this window elapses, bash_exec returns a partial snapshot with status=running and the session_id so the agent can poll or send stdin. 0 disables soft yield. Default: 10000.", "default": 10000},
    "max_output_tokens": {"type": "integer", "description": "Approximate output budget in model tokens (4 bytes/token). Output beyond this is truncated with a [truncated] marker. Default: 16384.", "default": 16384},
    "env": {"type": "object", "description": "Optional environment variable overrides applied on top of the current process environment.", "additionalProperties": {"type": "string"}},
    "is_background": {"type": "boolean", "description": "When true, spawn the command as a detached PTY session and return immediately with the session_id. Use list_background_processes and read_background_output to inspect it later.", "default": false},
    "idle_timeout_ms": {"type": "integer", "description": "Idle output timeout in milliseconds. When >0 and no new output arrives for this long, bash_exec yields a partial snapshot. 0 disables the idle timeout. Default: 0.", "default": 0}
  },
  "required": ["command"]
}
```

**Returns (from `shell_ops/bash.py`):**

Completed foreground run:

```json
{
  "success": true,
  "message": "Executed 'npm install'",
  "output": "<ANSI-stripped stdout/stderr combined>",
  "details": {
    "command": "npm install",
    "exit_code": 0,
    "output": "...",
    "status": "exited",
    "truncated": false,
    "session_id": "<uuid4 hex>",
    "tier": "local"
  }
}
```

Yielded (still running) run:

```json
{
  "success": true,
  "message": "Yielded after 10000ms; session <id> still running",
  "output": "<partial>",
  "details": {"command": "...", "exit_code": null, "status": "running", "truncated": false, "session_id": "<id>", "tier": "local"}
}
```

Background spawn:

```json
{
  "success": true,
  "message": "Started background PTY session <id>",
  "session_id": "<id>",
  "details": {"command": "...", "pid": 12345, "status": "running", "tier": "local", "is_background": true}
}
```

Timeout:

```json
{
  "success": false,
  "message": "Command timed out after 120s: <cmd>",
  "suggestion": "Increase the timeout, use is_background=True, or split the command into smaller steps",
  "details": {"command": "...", "timeout": 120, "output": "<partial>", "exit_code": 124, "session_id": "<id>", "tier": "local"}
}
```

**Quirks & gotchas:**

- **DANGEROUS**, scope `terminal.access`. **Unusually, it is also in
  `PLAN_MODE_ALLOWED`** — this is the only dangerous tool that runs in
  plan mode. The rationale is read-only exploration.
- Every invocation spawns a **fresh** PTY session. If you want a
  persistent shell that keeps state between commands, use
  `shell_open` / `shell_exec` instead.
- ANSI escape codes and control characters are stripped from the
  returned output via `strip_ansi_codes`. Newlines and tabs are kept.
- Output is truncated **tail-preserving**: when the accumulated byte
  count exceeds `max_output_tokens * 4`, the tool keeps the last
  `budget` bytes and prepends `"\n[truncated]\n"`.
- `timeout_ms` **overrides** `timeout` when both are present. `timeout`
  units are seconds; `timeout_ms` units are milliseconds.
- `yield_time_ms` is a **soft** yield — the tool returns a partial
  snapshot with `status="running"` and the session ID so the agent can
  poll via `read_background_output` or send input via `write_stdin`.
  Setting `yield_time_ms=0` disables soft yield and forces the call to
  run to completion or hit the hard timeout.
- `idle_timeout_ms > 0` yields when no new bytes have arrived for that
  long, independent of the soft yield.
- `cwd` is resolved relative to `context["cwd"]` → `$PROJECT_ROOT` →
  `os.getcwd()`. Absolute `cwd` values are used verbatim. There is no
  explicit root-containment check on the resolved cwd, unlike file ops.
- `env` is merged on top of the current process environment (not a
  replacement).
- When the budget (`max_output_tokens * 4`) is hit mid-run, the tool
  SIGTERMs the process group, returns the collected output with
  `status="truncated"`, and closes the session.
- On hard timeout, the process group is SIGTERMed, given 2 seconds to
  exit, then SIGKILLed if still alive.
- `run_id` for background-process scoping is resolved from
  `context["run_id"]` → `chat_id` → `task_id` → `message_id`.
- Not wrapped in `@tool_retry`.

**Example call:**

```json
{"name": "bash_exec", "arguments": {"command": "pytest -x", "yield_time_ms": 5000, "idle_timeout_ms": 2000}}
```

**Related:** `shell_open`, `shell_exec`, `write_stdin`,
`list_background_processes`, `read_background_output`.

---

#### `shell_open`

**Category:** `shell_commands`

**Description (verbatim):** `Open an interactive shell session in the current project directory. Returns session_id for subsequent operations. MUST be called before shell_exec. The shell remains open until explicitly closed with shell_close.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "command": {
      "type": "string",
      "description": "Shell command to run (default: /bin/sh). The shell starts in the project directory with all your source files."
    }
  },
  "required": []
}
```

**Returns (from `shell_ops/session.py`):**

```json
{
  "success": true,
  "message": "Opened shell session <uuid4 hex>",
  "session_id": "<id>",
  "details": {"command": "/bin/sh", "tier": "local"}
}
```

**Quirks & gotchas:**

- **DANGEROUS**, scope `terminal.access`, blocked in plan mode.
- Defaults to `/bin/sh`. Any command string is accepted; if it contains
  a space it is `split()`ed and passed as argv; otherwise run as a bare
  binary (no `sh -c` wrapping).
- Session runs in the project root (`context["cwd"]` →
  `$PROJECT_ROOT` → `os.getcwd()`).
- The underlying PTY session is tracked in `PTY_SESSIONS` and
  background-scoped by `run_id`.
- Fails cleanly with a structured error if the shell binary is missing
  or `/dev/ptmx` is unavailable.
- Sessions **do not auto-expire** in local mode — they leak until
  `shell_close` is called or the process exits. An `atexit` hook in the
  subagent registry is unrelated.

**Example call:**

```json
{"name": "shell_open", "arguments": {}}
```

**Related:** `shell_exec`, `shell_close`, `write_stdin`.

---

#### `shell_close`

**Category:** `shell_commands`

**Description (verbatim):** `Close an active shell session. Always close sessions when done to free resources.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "session_id": {"type": "string", "description": "Shell session ID to close"}
  },
  "required": ["session_id"]
}
```

**Returns:**

```json
{"success": true, "message": "Closed shell session <id>", "session_id": "<id>", "details": {"tier": "local"}}
```

**Quirks & gotchas:**

- **DANGEROUS**, scope `terminal.access`, blocked in plan mode.
- Sends SIGTERM to the process group, waits 1 second, then SIGKILLs if
  still alive, then cleans up the drain task.
- Returns a structured error if the session ID is unknown.
- Idempotent at the registry layer (the internal `close` is a no-op for
  missing sessions), but the tool surface rejects unknown IDs so the
  model notices mistakes.

**Example call:**

```json
{"name": "shell_close", "arguments": {"session_id": "abc123"}}
```

**Related:** `shell_open`.

---

#### `shell_exec`

**Category:** `shell_commands`

**Description (verbatim):** `Execute a command in an open shell session and wait for output. REQUIRES session_id from shell_open first. DO NOT use 'exit' or close the shell - it stays open for multiple commands.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "session_id": {"type": "string", "description": "Shell session ID obtained from shell_open"},
    "command": {"type": "string", "description": "Command to execute (automatically adds \\n). DO NOT include 'exit' - the shell stays open."},
    "wait_seconds": {"type": "number", "description": "Seconds to wait before reading output (default: 2)"},
    "max_output_tokens": {"type": "integer", "description": "Approximate output budget in model tokens (4 bytes/token). Default: 16384.", "default": 16384}
  },
  "required": ["session_id", "command"]
}
```

**Returns (from `shell_ops/execute.py`):**

```json
{
  "success": true,
  "message": "Executed 'ls -la' in session <id>",
  "output": "<ANSI-stripped>",
  "session_id": "<id>",
  "details": {"bytes": 1234, "status": "running", "exit_code": null, "tier": "local"}
}
```

**Quirks & gotchas:**

- **DANGEROUS**, scope `terminal.access`, blocked in plan mode.
- Wrapped in `@tool_retry` — retries on `ConnectionError`, `TimeoutError`,
  generic `OSError`.
- A trailing `\n` is auto-appended if missing.
- `wait_seconds` is capped to a minimum of 1 ms and converted to a
  drain window in milliseconds.
- Output accumulation is **non-blocking for exit** — `wait_for_exit=False`
  — because the shell session is expected to stay open.
- Output is ANSI-stripped and truncated to
  `max_output_tokens * 4` bytes.
- Errors like "Unknown shell session" are returned as structured
  payloads, not exceptions.

**Example call:**

```json
{"name": "shell_exec", "arguments": {"session_id": "abc123", "command": "npm install"}}
```

**Related:** `shell_open`, `shell_close`.

---

#### `write_stdin`

**Category:** `shell_commands`

**Description (verbatim):** `Write characters into a running PTY session previously created by bash_exec (either is_background=True or a yielded long-running command). Include a trailing newline to simulate pressing Enter. Returns whatever output arrives within yield_time_ms.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "session_id": {"type": "string", "description": "Identifier returned by bash_exec for the running PTY session."},
    "chars": {"type": "string", "description": "Characters to send to the session's stdin. Include a trailing '\\n' to simulate pressing Enter."},
    "yield_time_ms": {"type": "integer", "description": "Drain window in milliseconds after the write (default: 250).", "default": 250},
    "max_output_tokens": {"type": "integer", "description": "Approximate token budget for the drained output (default: 4096).", "default": 4096}
  },
  "required": ["session_id", "chars"]
}
```

**Returns (from `shell_ops/write_stdin.py`):**

```json
{
  "success": true,
  "message": "Wrote 5 chars to session <id>",
  "session_id": "<id>",
  "new_output": "<drained output>",
  "status": "running",
  "truncated": false,
  "details": {"session_id": "<id>", "new_output": "...", "status": "running", "truncated": false, "return_code": 0, "tier": "local"}
}
```

**Quirks & gotchas:**

- Not in `DANGEROUS_TOOLS` and **not** in `TOOL_REQUIRED_SCOPES` — no
  scope required and no approval gate. The session must already exist;
  callers cannot spawn a new PTY through `write_stdin`.
- `chars` is written verbatim — the caller is responsible for trailing
  newlines. Useful for sending interactive answers (`y\n`), signals
  (send `\x03` for Ctrl-C), or paste-style input.
- Drain window after the write is bounded by `yield_time_ms` (default
  **250 ms**) — very short compared to `bash_exec`.
- Output is truncated to `max_output_tokens * 4` bytes (default budget
  **4096 × 4 = 16 KiB**) and ANSI-stripped.
- On `KeyError` (session removed during the call) or
  `OSError` / `BrokenPipeError` (e.g. the process died mid-write), the
  tool returns a structured error, not an exception.
- Not wrapped in `@tool_retry`.

**Example call:**

```json
{"name": "write_stdin", "arguments": {"session_id": "abc123", "chars": "y\n", "yield_time_ms": 500}}
```

**Related:** `bash_exec` (`is_background=True`), `read_background_output`.

---

#### `list_background_processes`

**Category:** `shell_commands`

**Description (verbatim):** `List background PTY sessions spawned by bash_exec in this invocation. Returns session_id, command, pid, started_at, status, and exit_code for each live or recently-exited session.`

**Parameters:**

```json
{"type": "object", "properties": {}, "required": []}
```

**Returns (from `shell_ops/background.py`):**

```json
{
  "success": true,
  "message": "2 background session(s)",
  "sessions": [
    {"session_id": "<id>", "command": "npm run dev", "pid": 12345, "started_at": "2025-01-01T12:00:00+00:00", "status": "running", "exit_code": null}
  ],
  "details": {"count": 2, "run_id": "<id or None>", "tier": "local"}
}
```

**Quirks & gotchas:**

- Not DANGEROUS, no scope required.
- **Scoped by `run_id`**: only sessions created under the same
  invocation ID are visible. When no `run_id` can be resolved from the
  context, every session in the registry is returned as a fallback.
- The registry is process-global, so multiple concurrent agent runs in
  the same process share it but are filtered by `run_id`.
- `started_at` is rendered as an ISO-8601 UTC timestamp.

**Example call:**

```json
{"name": "list_background_processes", "arguments": {}}
```

**Related:** `read_background_output`, `bash_exec` (`is_background=True`).

---

#### `read_background_output`

**Category:** `shell_commands`

**Description (verbatim):** `Read the tail of a background PTY session's accumulated output (up to the last `lines` lines). Use list_background_processes to discover valid session IDs. Non-destructive — multiple reads return the same tail.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "session_id": {"type": "string", "description": "Background PTY session ID."},
    "lines": {"type": "integer", "description": "Number of tail lines to return (default: 200).", "default": 200},
    "delay_ms": {"type": "integer", "description": "Optional wait in milliseconds before reading, to let the process make progress.", "default": 0}
  },
  "required": ["session_id"]
}
```

**Returns:**

```json
{
  "success": true,
  "message": "Read 42 line(s) from session <id>",
  "session_id": "<id>",
  "output": "<tail joined with \\n>",
  "status": "running",
  "truncated": true,
  "details": {"session_id": "<id>", "status": "running", "exit_code": null, "total_lines": 1500, "returned_lines": 42, "truncated": true, "tier": "local"}
}
```

**Quirks & gotchas:**

- Reads up to **`64 KiB`** of accumulated PTY history
  (`_MAX_HISTORY_BYTES`). Longer histories are tail-sliced before the
  line-based trim happens.
- **Non-destructive** — peeks at the `history` bytearray in the session
  entry. Subsequent reads see the same tail plus anything new.
- Output is ANSI-stripped before splitting on `\n`; a trailing empty
  line is dropped.
- Enforces cross-invocation isolation: sessions owned by a different
  `run_id` return an "access denied" error.
- `delay_ms` is an optional `asyncio.sleep` before reading, to let a
  slow-producing process make progress.
- Raises `ValueError` if `lines <= 0` or `delay_ms < 0`.

**Example call:**

```json
{"name": "read_background_output", "arguments": {"session_id": "abc123", "lines": 50, "delay_ms": 500}}
```

**Related:** `list_background_processes`, `bash_exec`.

---

#### `python_repl`

**Category:** `shell_commands`

**Description (verbatim):** `Execute Python code in a persistent REPL session. Locals survive across calls within the same session_id, so you can define a variable or import a module in one call and use it in the next. Expressions return a `value` (repr of the result); statements return stdout/stderr. Pass reset=true to drop the session. timeout_ms is a hard deadline — on timeout the session is marked bad and must be reset.`

**Parameters:**

```json
{
  "type": "object",
  "properties": {
    "code": {"type": "string", "description": "Python source code (expression or statements)."},
    "session_id": {"type": "string", "description": "Persistent session identifier. Omit to auto-generate a fresh one (the id is returned in the response)."},
    "reset": {"type": "boolean", "description": "Drop any existing session for session_id before executing.", "default": false},
    "timeout_ms": {"type": "integer", "description": "Hard deadline in milliseconds. On timeout the session is marked bad.", "default": 30000},
    "max_output_tokens": {"type": "integer", "description": "Approximate token budget for stdout/stderr/value.", "default": 4096}
  },
  "required": ["code"]
}
```

**Returns (from `shell_ops/python_repl.py`):**

```json
{
  "success": true,
  "message": "python_repl executed in session <id>",
  "session_id": "<id>",
  "stdout": "<captured>",
  "stderr": "<captured>",
  "value": "42",
  "timed_out": false,
  "details": {"session_id": "<id>", "stdout": "...", "stderr": "...", "value": "42", "timed_out": false, "tier": "local"}
}
```

**Quirks & gotchas:**

- Not in `DANGEROUS_TOOLS` and **not** scope-gated — even though it
  executes arbitrary Python in the host process.
- Each session owns a `code.InteractiveInterpreter` with its own
  `locals_dict`. The REPL mirrors the stdlib REPL behavior: the last
  evaluated expression is bound to `_`.
- Expression vs statement detection uses `ast.parse(mode="eval")` first,
  falling back to `mode="exec"` on `SyntaxError`.
- Execution is dispatched to a **daemon thread**. On `timeout_ms` the
  thread keeps running (Python cannot be safely interrupted), the
  session is marked `bad` with the timeout reason, and subsequent calls
  return an error until `reset=True` is passed.
- Concurrent calls against the same session serialize on a per-session
  `threading.Lock`. If the lock is already held the call returns with
  stderr `"another python_repl call is already running against this session"`.
- Output is truncated to `max_output_tokens * 4` bytes with a tail-keep
  strategy.
- `session_id` is auto-generated (uuid4 hex) when omitted. The new ID is
  returned in the response so the agent can reuse it.
- The default token budget is `4096` (half of `bash_exec`).
- Not wrapped in `@tool_retry`.

**Example call:**

```json
{"name": "python_repl", "arguments": {"code": "import json; json.dumps({'x': 1})"}}
```

**Related:** `bash_exec`, `shell_exec`.

---
