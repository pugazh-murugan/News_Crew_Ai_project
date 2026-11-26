"""Topic 10: Knowledge Source & RAG.

Knowledge = documents you attach to an agent/crew. CrewAI chunks them,
embeds them into a vector store, and at run time retrieves only the chunks
relevant to the question (Retrieval-Augmented Generation) - so the LLM answers
from YOUR data instead of guessing.

Knowledge sources available: StringKnowledgeSource, TextFileKnowledgeSource,
PDFKnowledgeSource, CSVKnowledgeSource, JSONKnowledgeSource,
ExcelKnowledgeSource, CrewDoclingSource (URLs, docx, md...).

Note: TextFileKnowledgeSource file paths are relative to a `knowledge/`
folder in the project root, so this example copies the FAQ there first.

Run:  python topics/10_knowledge_source_rag.py
"""
import shutil

from crewai import Agent, Task, Crew
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

from common import DATA_DIR, PROJECT_ROOT, make_llm

llm = make_llm()

# --- Source 1: a file (must live under <project root>/knowledge/) ---
knowledge_dir = PROJECT_ROOT / "knowledge"
knowledge_dir.mkdir(exist_ok=True)
shutil.copy(DATA_DIR / "company_faq.txt", knowledge_dir / "company_faq.txt")
faq_source = TextFileKnowledgeSource(file_paths=["company_faq.txt"])

# --- Source 2: plain text defined in code ---
extra_source = StringKnowledgeSource(
    content="Acme's CEO is Jane Doe. The company was founded in 2015 in Pune."
)

support_agent = Agent(
    role="Acme Support Agent",
    goal="Answer customer questions using ONLY the company knowledge base.",
    backstory="If the answer is not in your knowledge, you say you don't know.",
    knowledge_sources=[faq_source, extra_source],  # agent-level knowledge
    llm=llm,
    verbose=True,
)

task = Task(
    description=(
        "Answer these customer questions:\n"
        "1) How many days do I have to return an item?\n"
        "2) Does the standard warranty cover water damage?\n"
        "3) Who founded Acme's competitor Globex?"
    ),
    expected_output="Three numbered answers. Say 'I don't know' when the knowledge has no answer.",
    agent=support_agent,
)

crew = Crew(agents=[support_agent], tasks=[task], verbose=True)

if __name__ == "__main__":
    print(crew.kickoff())
