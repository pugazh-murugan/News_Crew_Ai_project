"""Topic 1: Creating Multi-step Workflows.

A workflow = several Tasks run in order, where each step builds on the last.
Key ideas shown here:
  * one Task per step, each with a clear expected_output
  * `context=[...]` passes earlier task output to a later task
  * `output_file` saves a step's result to disk
  * `{placeholders}` in descriptions are filled from kickoff(inputs=...)

Run:  python topics/01_multistep_workflows.py
"""
from crewai import Agent, Task, Crew, Process

from common import make_llm

llm = make_llm()

planner = Agent(
    role="Content Planner",
    goal="Plan a short blog post about {topic}.",
    backstory="You outline posts so writers know exactly what to cover.",
    llm=llm,
)
writer = Agent(
    role="Blog Writer",
    goal="Write a clear blog post from an outline.",
    backstory="You write only from the outline you are given.",
    llm=llm,
)
editor = Agent(
    role="Editor",
    goal="Polish a draft for clarity and correct grammar.",
    backstory="You tighten sentences and fix mistakes without changing the meaning.",
    llm=llm,
)

# Step 1 -> Step 2 -> Step 3
plan = Task(
    description="Create a 4-point outline for a blog post about {topic}.",
    expected_output="A numbered list of 4 outline points, one line each.",
    agent=planner,
)
draft = Task(
    description="Write a ~150 word blog post that follows the outline.",
    expected_output="A blog post of about 150 words.",
    agent=writer,
    context=[plan],  # explicit hand-over of step 1's output
)
final = Task(
    description="Edit the draft for clarity and grammar.",
    expected_output="The final edited blog post, plain text.",
    agent=editor,
    context=[draft],
    output_file="topics/output/01_blog_post.txt",  # step result saved to disk
)

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, draft, final],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff(inputs={"topic": "why developers should learn CrewAI"})
    print("\n===== FINAL =====")
    print(result)

    # Every task keeps its own output, so you can inspect each step.
    for t in crew.tasks:
        print(f"\n[step] {t.agent.role}: {len(str(t.output))} chars")
