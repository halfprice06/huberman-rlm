#!/usr/bin/env python3
"""Simple one-shot Q&A example using DSPy RLM over Huberman transcripts."""

import dspy
from config import MAIN_MODEL, SUB_MODEL, MAX_ITERATIONS, MAX_LLM_CALLS, TRANSCRIPTS_DIR


def load_transcripts() -> dict[str, str]:
    transcripts = {}
    for filepath in TRANSCRIPTS_DIR.glob("*.txt"):
        parts = filepath.stem.split("_", 2)
        title = parts[2].replace("_", " ").replace(".vtt", "") if len(parts) >= 3 else filepath.stem
        transcripts[title] = filepath.read_text()
    return transcripts


def main():
    transcripts = load_transcripts()
    print(f"Loaded {len(transcripts)} transcripts")

    dspy.configure(lm=dspy.LM(MAIN_MODEL))

    rlm = dspy.RLM(
        signature="transcripts, question -> answer, sources",
        max_iterations=MAX_ITERATIONS,
        max_llm_calls=MAX_LLM_CALLS,
        sub_lm=dspy.LM(SUB_MODEL),
        verbose=True,
    )

    question = "What does Andrew Huberman recommend for improving focus?"
    print(f"\nQ: {question}\n")

    result = rlm(transcripts=transcripts, question=question)

    print(f"\nAnswer:\n{result.answer}")
    print(f"\nSources: {result.sources}")


if __name__ == "__main__":
    main()
