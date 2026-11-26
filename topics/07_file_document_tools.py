"""Topic 7: File & Document Processing Tools.

FileReadTool    - read any text file
FileWriterTool  - write/save text to a file
TXTSearchTool   - semantic (RAG) search inside a text file
PDFSearchTool   - same idea for PDFs (pass pdf='path.pdf'); also see
                  DOCXSearchTool, CSVSearchTool, JSONSearchTool

Search-style tools embed the document, so they need OPENAI_API_KEY too.

Run:  python topics/07_file_document_tools.py
"""
from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool, FileWriterTool, TXTSearchTool

from common import DATA_DIR, PROJECT_ROOT, make_llm

llm = make_llm()

FAQ = str(DATA_DIR / "company_faq.txt")
OUT_DIR = PROJECT_ROOT / "topics" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

read_file = FileReadTool(file_path=FAQ)          # default file to read
search_faq = TXTSearchTool(txt=FAQ)              # semantic search on one file
write_file = FileWriterTool(directory=str(OUT_DIR), overwrite=True)

processor = Agent(
    role="Document Processor",
    goal="Extract answers from documents and save reports.",
    backstory="You read documents carefully and save clean summaries to files.",
    tools=[read_file, search_faq, write_file],
    llm=llm,
    verbose=True,
)

task = Task(
    description=(
        "Using the FAQ document, summarize the return policy, warranty and "
        "shipping rules as 3 bullet points, then save them to a file named "
        "07_faq_summary.txt."
    ),
    expected_output="Confirmation that 07_faq_summary.txt was written, plus the 3 bullets.",
    agent=processor,
)

crew = Crew(agents=[processor], tasks=[task], verbose=True)

if __name__ == "__main__":
    print(crew.kickoff())
    print(f"\nCheck: {OUT_DIR / '07_faq_summary.txt'}")
