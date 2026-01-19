#!/usr/bin/env python3
"""Interactive CLI for Huberman podcast Q&A using DSPy RLM."""

import logging
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.table import Table
from rich import box

import dspy
from config import MAIN_MODEL, SUB_MODEL, MAX_ITERATIONS, MAX_LLM_CALLS, TRANSCRIPTS_DIR

console = Console()


class RLMProgressHandler(logging.Handler):
    """Logging handler that displays RLM iteration progress."""

    def __init__(self, console: Console):
        super().__init__()
        self.console = console

    def emit(self, record):
        msg = record.getMessage()
        if "RLM iteration" not in msg:
            return

        parts = msg.split("\n", 1)
        header = parts[0]

        iter_num = header.split("iteration")[1].strip().split("/")[0].strip()
        self.console.print()
        self.console.rule(f"[bold cyan]Step {iter_num}[/bold cyan]", style="cyan")

        if len(parts) <= 1 or "Reasoning:" not in parts[1]:
            return

        content = parts[1]
        reasoning_start = content.find("Reasoning:") + len("Reasoning:")
        code_start = content.find("Code:")

        if code_start > 0:
            reasoning = content[reasoning_start:code_start].strip()
            code = content[code_start + len("Code:"):].strip()
        else:
            reasoning = content[reasoning_start:].strip()
            code = ""

        if reasoning:
            display = reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
            self.console.print(Panel(
                Text(display, style="italic"),
                title="[yellow]Reasoning[/yellow]",
                border_style="yellow",
                padding=(0, 1),
            ))

        if code:
            code = code.strip()
            for prefix in ("```python", "```"):
                if code.startswith(prefix):
                    code = code[len(prefix):]
                    break
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()

            if len(code) > 800:
                code = code[:800] + "\n# ... (truncated)"

            self.console.print(Panel(
                Syntax(code, "python", theme="monokai", line_numbers=False),
                title="[green]Code[/green]",
                border_style="green",
                padding=(0, 1),
            ))


def setup_logging():
    handler = RLMProgressHandler(console)
    handler.setLevel(logging.INFO)

    rlm_logger = logging.getLogger("dspy.predict.rlm")
    rlm_logger.setLevel(logging.INFO)
    rlm_logger.addHandler(handler)

    for name in ("httpx", "anthropic", "google"):
        logging.getLogger(name).setLevel(logging.WARNING)


def load_transcripts() -> dict[str, str]:
    transcripts = {}
    for filepath in TRANSCRIPTS_DIR.glob("*.txt"):
        parts = filepath.stem.split("_", 2)
        title = parts[2].replace("_", " ").replace(".vtt", "") if len(parts) >= 3 else filepath.stem
        transcripts[title] = filepath.read_text()
    return transcripts


def format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "No previous conversation."
    lines = ["Previous conversation:"]
    for i, (q, a) in enumerate(history, 1):
        lines.extend([f"\nQ{i}: {q}", f"A{i}: {a}"])
    return "\n".join(lines)


def display_answer(result):
    console.print()
    console.rule("[bold green]Answer[/bold green]", style="green")
    console.print()

    answer = getattr(result, 'answer', str(result))
    console.print(Panel(
        Markdown(answer),
        title="[bold white]Response[/bold white]",
        border_style="green",
        padding=(1, 2),
    ))

    sources = getattr(result, 'sources', None)
    if sources:
        console.print()
        text = "\n".join(f"• {s}" for s in sources) if isinstance(sources, list) else str(sources)
        console.print(Panel(text, title="[bold blue]Sources[/bold blue]", border_style="blue", padding=(0, 1)))


def print_welcome(num_transcripts: int):
    console.print()
    console.print(Panel(
        f"""[bold cyan]Huberman Lab Podcast Q&A[/bold cyan]

Powered by DSPy RLM

[dim]• {num_transcripts} transcripts loaded
• Models: {MAIN_MODEL} / {SUB_MODEL}
• Conversational: follow-ups use context
• Type 'help' for commands[/dim]""",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def print_help():
    table = Table(box=box.ROUNDED, border_style="cyan")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    table.add_row("quit, exit, q", "Exit")
    table.add_row("help", "Show commands")
    table.add_row("reset", "Clear screen and history")
    table.add_row("history", "Show conversation history")
    console.print(table)
    console.print()


def main():
    console.print("[dim]Loading transcripts...[/dim]")
    transcripts = load_transcripts()

    if not transcripts:
        console.print("[red]No transcripts found. Run: unzip transcripts.zip[/red]")
        sys.exit(1)

    console.print("[dim]Configuring models...[/dim]")
    dspy.configure(lm=dspy.LM(MAIN_MODEL))

    rlm = dspy.RLM(
        signature="transcripts, conversation_history, question -> answer, sources",
        max_iterations=MAX_ITERATIONS,
        max_llm_calls=MAX_LLM_CALLS,
        sub_lm=dspy.LM(SUB_MODEL),
        verbose=True,
    )

    history: list[tuple[str, str]] = []
    setup_logging()
    print_welcome(len(transcripts))

    while True:
        try:
            console.print("[bold cyan]?[/bold cyan] ", end="")
            question = input().strip()

            if not question:
                continue

            cmd = question.lower()
            if cmd in ('quit', 'exit', 'q'):
                break
            if cmd == 'help':
                print_help()
                continue
            if cmd == 'reset':
                history.clear()
                console.clear()
                print_welcome(len(transcripts))
                continue
            if cmd == 'history':
                if not history:
                    console.print("[dim]No history yet.[/dim]")
                else:
                    for i, (q, a) in enumerate(history, 1):
                        console.print(f"\n[cyan]Q{i}:[/cyan] {q}")
                        console.print(f"[green]A{i}:[/green] {a[:200]}{'...' if len(a) > 200 else ''}")
                console.print()
                continue

            console.print()
            if history:
                console.print(f"[dim]Context: {len(history)} previous turn(s)[/dim]")
            console.rule(f"[bold]{question[:60]}{'...' if len(question) > 60 else ''}[/bold]")

            try:
                result = rlm(
                    transcripts=transcripts,
                    conversation_history=format_history(history),
                    question=question,
                )
                display_answer(result)
                history.append((question, getattr(result, 'answer', str(result))))
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

            console.print()

        except (KeyboardInterrupt, EOFError):
            break

    console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
