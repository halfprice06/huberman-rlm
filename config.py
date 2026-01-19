"""Shared configuration loaded from .env"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MAIN_MODEL = os.getenv("MAIN_MODEL", "gemini/gemini-3-pro-preview")
SUB_MODEL = os.getenv("SUB_MODEL", "gemini/gemini-3-flash-preview")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "20"))
MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALLS", "25"))
TRANSCRIPTS_DIR = Path("data/transcripts")
