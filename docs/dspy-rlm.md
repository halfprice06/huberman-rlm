# DSPy RLM (Recursive Language Model) Module

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)

## Overview

RLM is an experimental inference strategy in DSPy 3.1.2+ where LLMs treat long contexts as an **external environment** rather than feeding them directly to the model. The LLM writes Python code to programmatically explore, decompose, and recursively call sub-LLMs over snippets.

## Import

```python
import dspy
from dspy.predict import RLM
# or simply: dspy.RLM
```

## Core Architecture

### How It Works

1. LLM receives metadata about input variables (type, length, preview)
2. LLM writes Python code to explore the data
3. Code executes in a sandboxed Deno/Pyodide WASM interpreter
4. LLM sees output, then writes more code based on what it learned
5. Process repeats until LLM calls `SUBMIT()` with final answer

### Available Tools in Sandbox

| Tool | Description |
|------|-------------|
| `llm_query(prompt)` | Query a sub-LLM (~500K char capacity) for semantic analysis |
| `llm_query_batched(prompts)` | Query multiple prompts concurrently (faster for batch) |
| `print()` | **Always use** to see results |
| `SUBMIT(field1, field2, ...)` | Submit final output when done |
| Standard libs | `re`, `json`, `collections`, `math`, etc. |

### Key Principles (from system prompt)

1. **EXPLORE FIRST** - Look at data before processing. Print samples, check types/lengths
2. **ITERATE** - Write small code snippets, observe outputs, then decide next steps
3. **VERIFY BEFORE SUBMITTING** - If results seem wrong, reconsider approach
4. **USE llm_query FOR SEMANTICS** - String matching finds WHERE; llm_query understands WHAT
5. **MINIMIZE RETYPING** - Use variables instead of copying values manually
6. **SUBMIT ONLY AFTER SEEING OUTPUTS** - SUBMIT ends immediately; inspect first

## Basic Usage

```python
import dspy

dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4-20250514"))

rlm = dspy.RLM(
    signature="context, query -> answer",
    max_iterations=20,
    max_llm_calls=50,
    verbose=True,
)

result = rlm(context="...long text...", query="What is X?")
print(result.answer)
print(result.trajectory)  # List of dicts with reasoning/code/output
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str` or `Signature` | Required | Defines inputs/outputs (e.g., "context, query -> answer") |
| `max_iterations` | `int` | 20 | Maximum REPL interaction iterations |
| `max_llm_calls` | `int` | 50 | Maximum sub-LLM calls per execution |
| `max_output_chars` | `int` | 100,000 | Maximum characters from REPL output |
| `verbose` | `bool` | False | Log detailed execution info |
| `tools` | `dict[str, Callable]` | None | Additional tool functions callable from interpreter |
| `sub_lm` | `dspy.LM` | None | Different LM for llm_query (e.g., cheaper model) |
| `interpreter` | `CodeInterpreter` | None | Custom interpreter implementation |

## Custom Tools

```python
def search_database(query: str) -> str:
    """Search the vector database for relevant documents."""
    # Your implementation
    return "search results..."

rlm = dspy.RLM(
    "context, query -> answer",
    tools={"search_database": search_database}
)
```

Reserved tool names (cannot be overridden): `llm_query`, `llm_query_batched`, `SUBMIT`, `print`

## Return Value

The `forward()` method returns a `Prediction` with:
- Output fields from signature (e.g., `result.answer`)
- `result.trajectory` - List of dicts with `reasoning`, `code`, `output` for each step
- `result.final_reasoning` - Last reasoning before submission

## Async Support

```python
result = await rlm.aforward(context="...", query="...")
```

## Prerequisites

**Deno must be installed** for the sandboxed Python interpreter:

```bash
curl -fsSL https://deno.land/install.sh | sh
# or on macOS:
brew install deno
```

## Key Files in DSPy

- `dspy/predict/rlm.py` - Main RLM module
- `dspy/primitives/python_interpreter.py` - Deno/Pyodide sandbox
- `dspy/primitives/code_interpreter.py` - Abstract interpreter protocol
- `dspy/primitives/repl_types.py` - REPLVariable, REPLEntry, REPLHistory types

## SUBMIT() Syntax for Multiple Outputs

When your signature has multiple output fields (e.g., `-> answer, sources`), use keyword arguments:

```python
# Correct - use keyword arguments for multiple outputs
SUBMIT(answer=my_answer, sources=my_sources)

# Single output field works with positional
SUBMIT(my_answer)
```

The error messages can be confusing - if you see conflicting errors about SUBMIT arguments, try keyword arguments.

## Thread Safety

RLM instances are **not thread-safe** when using a custom interpreter. Create separate RLM instances for concurrent use, or use the default PythonInterpreter which creates a fresh instance per `forward()` call.

## Example: Complex Document Analysis

```python
import dspy

dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4-20250514"))

# Use a cheaper model for sub-queries
sub_lm = dspy.LM("anthropic/claude-haiku-4-20250514")

rlm = dspy.RLM(
    signature="document, questions -> answers: list[str]",
    max_iterations=30,
    max_llm_calls=100,
    sub_lm=sub_lm,  # Cheaper model for llm_query calls
    verbose=True,
)

result = rlm(
    document=very_long_document,
    questions=["What is the main thesis?", "List all citations", "Summarize section 3"]
)

for q, a in zip(questions, result.answers):
    print(f"Q: {q}\nA: {a}\n")
```

## Environment Setup for This Repo

```bash
# Create venv with uv
uv venv
source .venv/bin/activate

# Install latest dspy
uv pip install dspy --upgrade

# Install Deno (required for RLM sandbox)
curl -fsSL https://deno.land/install.sh | sh
export PATH="$HOME/.deno/bin:$PATH"
```
