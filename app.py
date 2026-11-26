import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.tools import tool
from crewai_tools import SerperDevTool

# read keys from .env (real environment variables still win)
load_dotenv(Path(__file__).parent / ".env", override=False)

ANSWER_FILE = "answers.txt"
FAQ_FILE = Path(__file__).parent / "topics" / "data" / "company_faq.txt"
MODEL = os.getenv("MODEL", "gpt-4o-mini")


# tool for the entry agent - just appends whatever it gets to answers.txt
@tool("save_to_file")
def save_to_file(content: str) -> str:
    """Saves the given text at the end of answers.txt with a timestamp."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ANSWER_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{now}]\n")
        f.write(content.strip() + "\n")
        f.write("-" * 60 + "\n\n")
    return "Saved to " + ANSWER_FILE


def read_answers():
    # returns what is saved in answers.txt so far, empty string if no file yet
    if not os.path.exists(ANSWER_FILE):
        return ""
    with open(ANSWER_FILE, "r", encoding="utf-8") as f:
        return f.read()


def build_crew(use_memory=False):
    search_tool = SerperDevTool()

    # company faq as knowledge, only the assistant gets it
    faq = StringKnowledgeSource(content=FAQ_FILE.read_text(encoding="utf-8"))

    # agent 1 - answers from the company faq and its own knowledge, no tools
    assistant = Agent(
        role="Assistant",
        goal="Answer the customer's query directly from the company FAQ "
             "and your own knowledge",
        backstory="You are a friendly customer support assistant. "
                  "You answer clearly and simply without looking anything up. "
                  "If the company FAQ has the answer, use it.",
        knowledge_sources=[faq],
        llm=MODEL,
        verbose=True,
    )

    # agent 2 - the only one with the search tool
    web_assistant = Agent(
        role="Web Search Assistant",
        goal="Search the web for the customer's query and answer using what you find",
        backstory="You are a support researcher. You always search the web first "
                  "and then write an answer based only on the search results.",
        tools=[search_tool],
        llm=MODEL,
        verbose=True,
    )

    # agent 3 - saves everything to the file
    entry_agent = Agent(
        role="Entry Agent",
        goal="Save the query and both answers in answers.txt and return both answers",
        backstory="You keep the records for the support team. You save "
                  "exactly what the other agents wrote, you don't change it.",
        tools=[save_to_file],
        llm=MODEL,
        verbose=True,
    )

    task1 = Task(
        description="Answer this customer query from your own knowledge: {query}",
        expected_output="A clear and helpful answer to the query.",
        agent=assistant,
    )

    task2 = Task(
        description="Search the web for this customer query and write an answer "
                    "from the results: {query}",
        expected_output="A clear answer based on the web search results.",
        agent=web_assistant,
        context=[task1],
    )

    # context gives the entry agent both earlier answers
    task3 = Task(
        description=(
            "The customer query was: {query}\n\n"
            "Use the two answers from the previous tasks. Call the save_to_file tool "
            "once with the text in exactly this format:\n\n"
            "Query: <the query>\n"
            "Answer 1 (Assistant): <answer from the Assistant>\n"
            "Answer 2 (Web Search): <answer from the Web Search Assistant>\n\n"
            "After saving, give back both answers in your final output."
        ),
        expected_output="Confirmation that the file was saved, plus both answers.",
        agent=entry_agent,
        context=[task1, task2],
    )

    return Crew(
        agents=[assistant, web_assistant, entry_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        memory=use_memory,  # remembers earlier queries between runs
        verbose=True,
    )


def main():
    st.set_page_config(page_title="Support Crew")
    st.title("Multi-Agent Customer Support")
    st.write("Ask your question. One agent answers from the company FAQ and its own "
             "knowledge, another searches the web, and a third saves everything to a file.")

    # keys only from environment / .env
    missing = [k for k in ("OPENAI_API_KEY", "SERPER_API_KEY") if not os.getenv(k)]
    if missing:
        st.error("Missing environment variable(s): " + ", ".join(missing))
        st.stop()

    use_memory = st.sidebar.checkbox("Remember earlier queries (memory)")

    query = st.text_area("Your query or task", placeholder="How do I reset my password?")

    if st.button("Submit"):
        if not query.strip():
            st.warning("Please type a query first.")
            return

        try:
            with st.spinner("Agents are working..."):
                crew = build_crew(use_memory)
                result = crew.kickoff(inputs={"query": query.strip()})

            # tasks_output is in the same order as the tasks
            answer1 = result.tasks_output[0].raw
            answer2 = result.tasks_output[1].raw
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            return

        st.subheader("Assistant Answer")
        st.write(answer1)

        st.subheader("Web Search Answer")
        st.write(answer2)

        st.success(f"Query and both answers saved to {ANSWER_FILE}")

    # saved history, read back from the file
    history = read_answers()
    if history:
        with st.expander("Saved answers (answers.txt)"):
            st.text(history)


if __name__ == "__main__":
    main()
