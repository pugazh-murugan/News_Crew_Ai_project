"""Shared setup for every topic example: load .env, check keys, build the LLM.

Each topic file does:  from common import make_llm
Run any of them from the project root, e.g.:  python topics/01_multistep_workflows.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from crewai import LLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"

# Real OS environment variables win over .env values.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def require_env(*names: str) -> None:
    """Exit with a clear message if any variable is missing or blank."""
    missing = [n for n in names if not os.getenv(n, "").strip()]
    if missing:
        sys.exit(f"Missing variable(s): {', '.join(missing)}. "
                 "Add them to .env (see .env.example) or set them in your environment.")


def make_llm() -> LLM:
    """LLM used by all examples. Needs OPENAI_API_KEY and MODEL."""
    require_env("OPENAI_API_KEY", "MODEL")
    return LLM(model=os.environ["MODEL"], api_key=os.environ["OPENAI_API_KEY"])
