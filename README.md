# Huberman RLM

Q&A over Huberman Lab podcast transcripts using [DSPy RLM](https://dspy.ai/).

RLM lets the model write Python to explore documents instead of stuffing everything into context.

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

## Config

`.env`:
```
GEMINI_API_KEY=your-key
MAIN_MODEL=gemini/gemini-3-pro-preview
SUB_MODEL=gemini/gemini-3-flash-preview
```
