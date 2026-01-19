# Huberman RLM

Q&A over Huberman Lab podcast transcripts using [DSPY + RLM](https://dspy.ai/).

## What's RLM?

RLM (Recursive Language Model) lets the model write Python to explore input context in a python REPL instead of stuffing everything into one limited context. You give it a big dict of transcripts, ask a question, and watch it write code to search, filter, and analyze until it finds what it needs. The key is that it can use sub LMs to explore context without blowing up the main LM's context window. 

The code runs in a sandboxed Deno/Pyodide interpreter. When the model needs to do semantic analysis on a chunk of text, it calls `llm_query()` which hits a smaller/faster sub-model.

## What's in here

- `huberman_cli.py` - interactive CLI with conversation history
- `huberman_qa.py` - simple one-shot example
- `data/transcripts/` - 180 Huberman Lab transcripts

The CLI shows you each step as RLM thinks through the problem - the reasoning, the code it writes, and the output.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Deno required for RLM sandbox
curl -fsSL https://deno.land/install.sh | sh

cp .env.example .env
# add your API key
```

## Run

```bash
python huberman_cli.py
```

Type questions, get answers. Follow-ups work because it keeps conversation history. Type `help` for commands.

## Config

`.env`:
```
GEMINI_API_KEY=your-key
MAIN_MODEL=gemini/gemini-3-pro-preview
SUB_MODEL=gemini/gemini-3-flash-preview
MAX_ITERATIONS=20
MAX_LLM_CALLS=25
```

Uses litellm format for model names so you can swap in Anthropic, OpenAI, etc.
