"""Topic 6: Web Search Tools.

SerperDevTool    - Google search results via serper.dev   (needs SERPER_API_KEY)
ScrapeWebsiteTool - fetch and read the text of one specific URL

Typical pattern: search to find URLs, then scrape the best one for detail.

Run:  python topics/06_web_search_tools.py
"""
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

from common import make_llm, require_env

require_env("SERPER_API_KEY")
llm = make_llm()

search = SerperDevTool(n_results=5)  # how many results to return per query
scrape = ScrapeWebsiteTool()         # reads whichever URL the agent passes in

researcher = Agent(
    role="Web Researcher",
    goal="Answer questions using fresh web results and cite sources.",
    backstory="You search first, open the most relevant page, and always give URLs.",
    tools=[search, scrape],
    llm=llm,
    verbose=True,
)

task = Task(
    description=(
        "Find what the latest stable version of Python is and when it was "
        "released. Search the web, open one authoritative page to confirm."
    ),
    expected_output="Version number, release date, and the source URL.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task], verbose=True)

if __name__ == "__main__":
    print(crew.kickoff())
