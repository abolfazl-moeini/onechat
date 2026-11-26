# onechat

One chat session surviving AI backend switches (CLI agents + OpenAI-compatible APIs).

## Features

- **Single persistent conversation session**: Seamless handoffs across different AI backends without losing context.
- **Support for multi-agent CLIs**: Claude Code, OpenAI Codex, Google Gemini CLI, OpenCode, Qwen Code, and more.
- **Failover & cooldown circuit**: Graceful fallback when an agent hits quota, rate limits, or errors.
- **Zero secrets stored**: Credentials are read directly from environment variables.

## Installation

```bash
uv tool install onechat
# or
pip install onechat
```

## Quick Start

```bash
# Verify your CLI tools and environment
onechat doctor

# Start interactive chat REPL
onechat chat

# Run a single turn
onechat run "Refactor this function to be async"
```

## Python SDK

```python
from onechat import OneChatClient

client = OneChatClient()
session = client.session(workdir=".")
reply = session.ask_sync("Hello from onechat!")
print(reply.text)
```
