"""Two-agent CrewAI crew: a Researcher finds today's top 3 tech stories,
a Writer turns them into a 5-line newsletter.

Run:  python news_crew.py
Needs: OPENAI_API_KEY, MODEL (e.g. gpt-4o-mini), SERPER_API_KEY
Put them in a .env file next to this script (see .env.example) or set them
as OS environment variables. Real OS variables win over .env values.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# ---- 1. Load keys from .env / environment (never hardcode) ----
# Resolve .env relative to this file so it works from any working directory.
# override=False -> variables already set in the OS environment take priority.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

REQUIRED = ["OPENAI_API_KEY", "MODEL", "SERPER_API_KEY"]
missing = [name for name in REQUIRED if not os.getenv(name, "").strip()]
if missing:
    sys.exit(f"Missing variable(s): {', '.join(missing)}. "
             "Add them to .env (see .env.example) or set them in your environment.")

llm = LLM(model=os.environ["MODEL"], api_key=os.environ["OPENAI_API_KEY"])
today = datetime.now().strftime("%A, %d %B %Y")  # changes every day

# ---- 2. Agents ----
# Only the Researcher gets a search tool. A Writer with search would
# go and research again instead of just writing.
search = SerperDevTool()

researcher = Agent(
    role="Tech News Researcher",
    goal="Find today's top 3 tech news stories using web search.",
    backstory="You are a sharp tech journalist who scans the web daily "
              "and only reports stories you can back with a source URL.",
    tools=[search],
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Newsletter Writer",
    goal="Turn the researched stories into a crisp 5-line newsletter.",
    backstory="You are a concise newsletter editor. You write only from "
              "the research you are given and never invent facts.",
    # no tools on purpose
    llm=llm,
    verbose=True,
)

# ---- 3. Tasks ----
find = Task(
    description=(
        f"Today is {today}. Search the web and find the top 3 tech news "
        "stories published today or in the last 24 hours."
    ),
    expected_output=(
        "A numbered list of exactly 3 items. Each item has:\n"
        "Title: <headline>\n"
        "Summary: <one sentence>\n"
        "Source: <URL>\n"
        "No intro, no extra commentary."
    ),
    agent=researcher,
)

write = Task(
    description=(
        "Using ONLY the stories provided by the researcher, write a short "
        "tech newsletter. Do not add facts that are not in the research."
    ),
    expected_output=(
        "EXACTLY 5 lines of plain text, no markdown, no preamble, no extra lines:\n"
        "Line 1: Tech in 5 lines\n"
        "Line 2: 1. <first story, one line>\n"
        "Line 3: 2. <second story, one line>\n"
        "Line 4: 3. <third story, one line>\n"
        "Line 5: Why it matters: <one line>"
    ),
    agent=writer,
    # In a sequential crew, the 2nd task does NOT automatically see the
    # 1st task's output. context=[find] hands the Researcher's result over.
    context=[find],
)

# ---- 4. Crew ----
crew = Crew(
    agents=[researcher, writer],
    tasks=[find, write],
    process=Process.sequential,  # find first, then write
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n===== FINAL NEWSLETTER =====")
    print(result)

    # Self-check: count non-empty lines
    lines = [ln for ln in str(result).splitlines() if ln.strip()]
    print(f"\n[check] line count = {len(lines)} (expected 5)")
