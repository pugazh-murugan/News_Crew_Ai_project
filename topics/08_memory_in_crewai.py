"""Topic 8: Memory in CrewAI.

Without memory every kickoff() starts from a blank slate. With
`Crew(memory=True)` the crew stores what happened (task results, facts) in a
vector store and agents automatically recall relevant items on later tasks
and later runs.

In CrewAI 1.x memory is one unified `Memory` object that:
  * uses an LLM to infer scope / categories / importance when saving
  * embeds the content (OpenAI embeddings by default)
  * recalls by relevance, recency and importance

Run it TWICE to see the effect:
  python topics/08_memory_in_crewai.py "My name is Priya and I prefer short answers."
  python topics/08_memory_in_crewai.py "What is my name and what style do I prefer?"
"""
import sys

from crewai import Agent, Task, Crew

from common import make_llm

llm = make_llm()

assistant = Agent(
    role="Personal Assistant",
    goal="Help the user and remember what they tell you.",
    backstory="You use your memory of earlier conversations to personalize answers.",
    llm=llm,
    verbose=True,
)

message = sys.argv[1] if len(sys.argv) > 1 else "Hi! My name is Priya and I prefer short answers."

task = Task(
    description=f"The user says: {message}\nRespond helpfully, using anything you remember about them.",
    expected_output="A short, personalized reply.",
    agent=assistant,
)

crew = Crew(
    agents=[assistant],
    tasks=[task],
    memory=True,  # turn on the unified memory system
    verbose=True,
)

if __name__ == "__main__":
    print(crew.kickoff())
