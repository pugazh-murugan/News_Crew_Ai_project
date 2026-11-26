"""Topic 9: Short Term vs Long Term Memory.

Concept
  Short-term : context for the CURRENT run only (the current conversation /
               task chain). Gone when the process ends.
  Long-term  : persisted knowledge that survives across runs (user prefs,
               past results, lessons learned).

How it maps in CrewAI 1.x
  Older versions had separate ShortTermMemory / LongTermMemory classes. The
  current release uses a single `Memory` class; the difference is WHERE it is
  stored:
    * in-process / throw-away storage  -> behaves like short-term
    * a persistent storage path        -> behaves like long-term

This script demonstrates both using Memory directly (no crew needed):

  python topics/09_short_vs_long_term_memory.py remember   # save long-term facts
  python topics/09_short_vs_long_term_memory.py recall     # new process: still there
  python topics/09_short_vs_long_term_memory.py short      # temp memory vanishes
"""
import shutil
import sys
import tempfile

from crewai.memory.unified_memory import Memory

from common import PROJECT_ROOT, make_llm

llm = make_llm()

LONG_TERM_DIR = str(PROJECT_ROOT / ".memory" / "long_term")


def long_term() -> Memory:
    return Memory(llm=llm, storage=LONG_TERM_DIR)  # persisted on disk


def show(matches) -> None:
    if not matches:
        print("  (nothing found)")
    for m in matches:
        print(f"  - {m.record.content}  (score={m.score:.2f})")


def demo_remember() -> None:
    mem = long_term()
    mem.remember("The user's name is Priya.", importance=0.9)
    mem.remember("Priya prefers short answers and Python examples.", importance=0.8)
    print(f"Saved to long-term memory at {LONG_TERM_DIR}")
    print("Now run again with:  recall   (even after closing the terminal)")


def demo_recall() -> None:
    mem = long_term()
    print("Long-term recall for 'user preferences':")
    show(mem.recall("What does the user prefer?", limit=3))


def demo_short() -> None:
    tmp = tempfile.mkdtemp(prefix="short_term_")
    try:
        mem = Memory(llm=llm, storage=tmp)  # throw-away store
        mem.remember("Today's meeting is at 4 PM.", importance=0.5)
        print("Same process, short-term recall:")
        show(mem.recall("When is the meeting?", limit=3))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # memory is gone after the run
    print("Temp store deleted -> a new process would recall nothing.")


if __name__ == "__main__":
    modes = {"remember": demo_remember, "recall": demo_recall, "short": demo_short}
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in modes:
        sys.exit("Usage: python topics/09_short_vs_long_term_memory.py [remember|recall|short]")
    modes[mode]()
