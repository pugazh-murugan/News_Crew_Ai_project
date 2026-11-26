"""Topic 5: Integrating Custom Python Tools.

Two ways to turn your own Python into an agent tool:
  1. `@tool("name")` decorator on a function  (quickest)
  2. Subclass `BaseTool` with a pydantic args_schema (typed, more control)

The docstring / description is what the LLM reads to decide when to call the
tool, so write it clearly.

Run:  python topics/05_custom_python_tools.py
"""
from datetime import datetime
from typing import Type

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field

from common import make_llm

llm = make_llm()


# ---- Way 1: decorator ----
@tool("Word Counter")
def word_counter(text: str) -> str:
    """Count the number of words in the given text."""
    return f"{len(text.split())} words"


# ---- Way 2: BaseTool subclass ----
class DaysUntilInput(BaseModel):
    """Input schema: the LLM must supply these fields."""
    date: str = Field(..., description="Target date in YYYY-MM-DD format.")


class DaysUntilTool(BaseTool):
    name: str = "Days Until Date"
    description: str = "Returns how many days remain from today until a date (YYYY-MM-DD)."
    args_schema: Type[BaseModel] = DaysUntilInput

    def _run(self, date: str) -> str:
        try:
            target = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return "Error: date must be in YYYY-MM-DD format."
        return f"{(target - datetime.now().date()).days} days"


helper = Agent(
    role="Utility Assistant",
    goal="Answer questions by calling the right custom tool.",
    backstory="You never guess numbers; you use your tools.",
    tools=[word_counter, DaysUntilTool()],
    llm=llm,
    verbose=True,
)

task = Task(
    description=(
        "1) Count the words in: 'CrewAI makes multi-agent workflows simple'. "
        "2) Tell me how many days are left until 2027-01-01."
    ),
    expected_output="Two short lines: the word count, then the days remaining.",
    agent=helper,
)

crew = Crew(agents=[helper], tasks=[task], verbose=True)

if __name__ == "__main__":
    print(crew.kickoff())
