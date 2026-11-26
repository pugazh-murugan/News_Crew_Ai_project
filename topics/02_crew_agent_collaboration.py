"""Topic 2: Crew and Agent Collaboration.

Agents collaborate in two ways:
  1. Task context   - one agent's output feeds another's task.
  2. Delegation     - `allow_delegation=True` lets an agent ask a teammate
                      for help or hand off part of its work mid-task.

Here a Lead agent can delegate to a Fact-checker and a Copywriter.

Run:  python topics/02_crew_agent_collaboration.py
"""
from crewai import Agent, Task, Crew, Process

from common import make_llm

llm = make_llm()

lead = Agent(
    role="Product Lead",
    goal="Produce an accurate, catchy product announcement.",
    backstory="You coordinate specialists and only ship work that is checked.",
    allow_delegation=True,  # may delegate to other agents in the crew
    llm=llm,
    verbose=True,
)
fact_checker = Agent(
    role="Fact Checker",
    goal="Verify claims and flag anything unsupported.",
    backstory="You are skeptical and only accept claims backed by the brief.",
    allow_delegation=False,
    llm=llm,
    verbose=True,
)
copywriter = Agent(
    role="Copywriter",
    goal="Write punchy marketing copy.",
    backstory="You turn dry facts into short, energetic copy.",
    allow_delegation=False,
    llm=llm,
    verbose=True,
)

BRIEF = (
    "Product: Acme Buds. 30-hour battery, Bluetooth 5.3, water resistant (IPX4), "
    "price 59 USD, launches next month."
)

announce = Task(
    description=(
        f"Write a 3-sentence product announcement from this brief: {BRIEF}\n"
        "Ask the Copywriter to draft the copy and the Fact Checker to verify "
        "every claim before you give the final answer."
    ),
    expected_output="A verified 3-sentence announcement using only facts from the brief.",
    agent=lead,
)

crew = Crew(
    agents=[lead, fact_checker, copywriter],
    tasks=[announce],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    print(crew.kickoff())
