"""Topic 4: Tools in CrewAI.

Tools are what let an agent act: search, read files, call APIs.
Rules of thumb:
  * Give tools to the agent that needs them (agent-level `tools=[...]`).
  * A tool can also be attached to a single Task (`Task(tools=[...])`),
    which overrides the agent's tools for that task only.
  * Fewer, well-described tools make agents more reliable.

This example uses two ready-made tools from crewai_tools:
  FileReadTool       - read a local file
  DirectoryReadTool  - list files in a folder

Run:  python topics/04_tools_in_crewai.py
"""
from crewai import Agent, Task, Crew
from crewai_tools import DirectoryReadTool, FileReadTool

from common import DATA_DIR, make_llm

llm = make_llm()

list_files = DirectoryReadTool(directory=str(DATA_DIR))
read_file = FileReadTool()

analyst = Agent(
    role="Data Folder Analyst",
    goal="Find out what files exist and what they contain.",
    backstory="You inspect folders first, then read only the files you need.",
    tools=[list_files, read_file],  # agent-level tools
    llm=llm,
    verbose=True,
)

task = Task(
    description=(
        f"List the files in {DATA_DIR}, then read company_faq.txt "
        "and report the warranty length."
    ),
    expected_output="The list of files, then one sentence stating the warranty length.",
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[task], verbose=True)

if __name__ == "__main__":
    print(crew.kickoff())
