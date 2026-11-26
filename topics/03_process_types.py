"""Topic 3: Process types.

Process.sequential    - tasks run one after another in the order listed.
Process.hierarchical  - a manager agent plans, delegates tasks to the workers
                        and validates results. Needs `manager_llm` (or a
                        custom `manager_agent`).

Pick with:  python topics/03_process_types.py [sequential|hierarchical]
"""
import sys

from crewai import Agent, Task, Crew, Process

from common import make_llm

llm = make_llm()

researcher = Agent(
    role="Researcher",
    goal="Collect key facts about a topic from your own knowledge.",
    backstory="You give brief, accurate bullet points.",
    llm=llm,
)
writer = Agent(
    role="Writer",
    goal="Write a short summary from provided facts.",
    backstory="You write tight, plain-English summaries.",
    llm=llm,
)

research = Task(
    description="List 3 key facts about the Python programming language.",
    expected_output="3 short bullet points.",
    agent=researcher,
)
summary = Task(
    description="Write a 2-sentence summary using the facts provided.",
    expected_output="Exactly 2 sentences.",
    agent=writer,
    context=[research],
)


def build_crew(kind: str) -> Crew:
    if kind == "hierarchical":
        # The manager decides who does what. In hierarchical mode tasks do NOT
        # need a fixed `agent`; the manager assigns them.
        return Crew(
            agents=[researcher, writer],
            tasks=[
                Task(description=research.description,
                     expected_output=research.expected_output),
                Task(description=summary.description,
                     expected_output=summary.expected_output),
            ],
            process=Process.hierarchical,
            manager_llm=llm,
            verbose=True,
        )
    return Crew(
        agents=[researcher, writer],
        tasks=[research, summary],
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "sequential"
    if kind not in ("sequential", "hierarchical"):
        sys.exit("Usage: python topics/03_process_types.py [sequential|hierarchical]")
    print(f"Running with process = {kind}")
    print(build_crew(kind).kickoff())
