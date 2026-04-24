# tesslate-agent

Standalone autonomous agent with 30+ built-in tools for code editing, shell
execution, navigation, git, memory, structured planning, and subagent
delegation. Single-process, batteries-included, zero runtime dependencies on
the OpenSail orchestrator.

## Install

```bash
uv tool install git+https://github.com/Tesslate-AI/tesslate-agent
```

Or from PyPI once published:

```bash
uv tool install tesslate-agent
```

## Usage

```bash
# Run a task against the current directory
tesslate-agent run \
  --task "add a README.md with a one-paragraph project description" \
  --model openai/gpt-4o-mini \
  --workdir . \
  --output ./trajectory.json
```

The agent talks to any LLM reachable via
[LiteLLM](https://docs.litellm.ai/). Export the provider API key for whichever
model you pick:

| Provider | Env var |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GEMINI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Groq | `GROQ_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Together | `TOGETHER_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Cohere | `COHERE_API_KEY` |

## LiteLLM proxy

If you run a LiteLLM proxy, point the agent at it instead of exporting a
per-provider key:

```bash
export LITELLM_API_BASE="https://litellm.example.com"
export LITELLM_MASTER_KEY="sk-..."
tesslate-agent run --task "..." --model gpt-4o-mini
```

## Project root

The agent reads/writes files under the directory given by `--workdir` (or the
`PROJECT_ROOT` env var). Paths are resolved with strict containment checks —
the agent cannot escape the project root via symlinks or `..`.

## License

Apache-2.0. Copyright 2026 Tesslate AI.
