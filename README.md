# Huberman RLM

Q&A over Huberman Lab podcast transcripts using [DSPy's RLM](https://dspy.ai/) (Recursive Language Model).

RLM lets the model programmatically explore large document collections by writing Python code, rather than stuffing everything into context.

## Setup

```bash
# Clone and enter
git clone https://github.com/youruser/huberman-rlm.git
cd huberman-rlm

# Create venv
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install dspy rich python-dotenv

# Install Deno (required for RLM's sandboxed interpreter)
curl -fsSL https://deno.land/install.sh | sh
export PATH="$HOME/.deno/bin:$PATH"

# Configure
cp .env.example .env
# Edit .env with your API key
```

## Add Transcripts

Place transcript files in `data/transcripts/`:

```
data/transcripts/
├── 20240101_uuid_Episode_Title.vtt.txt
├── 20240102_uuid_Another_Episode.vtt.txt
└── ...
```

Expected filename format: `YYYYMMDD_uuid_Title.vtt.txt`

## Usage

**Interactive CLI:**
```bash
python huberman_cli.py
```

```
╭──────────────────────────────────────────────────────╮
│  Huberman Lab Podcast Q&A                            │
│  Powered by DSPy RLM                                 │
│  • 180 transcripts loaded                            │
│  • Models: gemini/gemini-3-pro-preview               │
╰──────────────────────────────────────────────────────╯

? What does Huberman recommend for improving sleep?

──────────────────── Step 1 ────────────────────────────
╭─ Reasoning ──────────────────────────────────────────╮
│ I need to search for sleep-related transcripts...    │
╰──────────────────────────────────────────────────────╯
╭─ Code ───────────────────────────────────────────────╮
│ sleep_eps = [t for t in transcripts if 'sleep' in t] │
╰──────────────────────────────────────────────────────╯
```

**One-shot example:**
```bash
python huberman_qa.py
```

## Configuration

Edit `.env`:

```bash
# API key for your provider
GEMINI_API_KEY=your-key
# or ANTHROPIC_API_KEY=your-key

# Models (litellm format)
MAIN_MODEL=gemini/gemini-3-pro-preview
SUB_MODEL=gemini/gemini-3-flash-preview

# RLM settings
MAX_ITERATIONS=20
MAX_LLM_CALLS=25
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `help` | Show commands |
| `reset` | Clear conversation history |
| `history` | Show conversation history |
| `clear` | Clear screen |
| `quit` | Exit |

## How RLM Works

Instead of feeding all transcripts directly to the model, RLM:

1. Gives the model access to transcripts as a Python dict
2. Model writes code to explore: search, filter, extract
3. Code runs in a sandboxed Deno/Pyodide interpreter
4. Model sees output, writes more code
5. For semantic analysis, model calls `llm_query()` with a sub-LLM
6. Repeats until `SUBMIT()` is called with final answer

This allows handling arbitrarily large document collections.

## Files

```
├── huberman_cli.py   # Interactive CLI
├── huberman_qa.py    # One-shot example
├── config.py         # Shared config from .env
├── .env.example      # Config template
└── docs/
    └── dspy-rlm.md   # RLM reference documentation
```
