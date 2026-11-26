# News Crew AI

A small [CrewAI](https://docs.crewai.com/) project with two parts:

1. **`news_crew.py`** - a two-agent crew. A *Researcher* searches the web for today's top 3 tech stories, and a *Writer* turns them into a 5-line newsletter.
2. **`app.py`** - a Streamlit customer-support app with a three-agent crew (see below).
3. **`topics/`** - ten short, runnable examples, one per CrewAI topic (workflows, collaboration, process types, tools, memory, knowledge/RAG, ...).

## Project structure

```
News_Crew_Ai_project/
├── news_crew.py                    # Researcher + Writer newsletter crew (CLI)
├── app.py                          # Streamlit customer-support app (3 agents)
├── answers.txt                     # Created by app.py: saved queries and answers
├── requirements.txt                # crewai, crewai-tools, python-dotenv, streamlit
├── .env.example                    # Template for your API keys (copy to .env)
├── .env                            # Your real keys (gitignored, never commit)
├── .gitignore / .gitattributes
└── topics/
    ├── common.py                   # Shared helper: loads .env, builds the LLM
    ├── data/company_faq.txt        # Sample document used by examples 4, 7, 10
    ├── 01_multistep_workflows.py
    ├── 02_crew_agent_collaboration.py
    ├── 03_process_types.py
    ├── 04_tools_in_crewai.py
    ├── 05_custom_python_tools.py
    ├── 06_web_search_tools.py
    ├── 07_file_document_tools.py
    ├── 08_memory_in_crewai.py
    ├── 09_short_vs_long_term_memory.py
    └── 10_knowledge_source_rag.py
```

Folders created when you run the examples (all gitignored): `topics/output/`, `.memory/`, `knowledge/`, `db/`.

## Setup

Requires Python 3.10+ (developed on 3.12).

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Git Bash: source .venv/Scripts/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure keys
copy .env.example .env              # then edit .env
```

### API keys

| Variable | Used for | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | LLM calls and embeddings (memory, knowledge, RAG tools) | platform.openai.com |
| `MODEL` | LLM model name, e.g. `gpt-4o-mini` | - |
| `SERPER_API_KEY` | Google search via `SerperDevTool` | serper.dev |

Keys are read from `.env` (next to `news_crew.py`) **or** from OS environment variables. If both exist, the OS environment variable wins. If you set a variable system-wide on Windows (e.g. with `setx`), restart VS Code/terminal so it is picked up. Never hardcode keys in code.

## Run the newsletter crew

```powershell
python news_crew.py
```

Prints the final 5-line newsletter and a self-check of the line count. Needs all three variables above.

## Run the support app

```powershell
streamlit run app.py
```

Type a customer query and press Submit. Three agents run in order:

1. **Assistant** answers from the company FAQ (`topics/data/company_faq.txt`, loaded as a knowledge source) and its own knowledge.
2. **Web Search Assistant** searches the web with `SerperDevTool` and answers from the results.
3. **Entry Agent** saves the query and both answers to `answers.txt` with a custom `save_to_file` tool.

The page shows both answers and lists the saved history. The sidebar checkbox **Remember earlier queries (memory)** turns on `Crew(memory=True)`. Needs `OPENAI_API_KEY` and `SERPER_API_KEY`; `MODEL` is optional (defaults to `gpt-4o-mini`).

Topics used: multi-step workflow with `context=` (1), sequential process (3), custom Python tool (5), web search tool (6), reading the saved file (7), memory (8), knowledge source (10).

## Topic examples

Run from the project root, e.g. `python topics/05_custom_python_tools.py`.

| # | File | Topic | Extra requirements |
|---|---|---|---|
| 1 | `01_multistep_workflows.py` | Multi-step workflows: chained tasks, `context=`, inputs, `output_file` (writes `topics/output/01_blog_post.txt`) | - |
| 2 | `02_crew_agent_collaboration.py` | Agents collaborating via delegation (`allow_delegation`) | - |
| 3 | `03_process_types.py` | Sequential vs hierarchical process: `... 03_process_types.py [sequential\|hierarchical]` | - |
| 4 | `04_tools_in_crewai.py` | Using built-in tools (`DirectoryReadTool`, `FileReadTool`) | - |
| 5 | `05_custom_python_tools.py` | Custom tools via `@tool` and `BaseTool` | - |
| 6 | `06_web_search_tools.py` | `SerperDevTool` + `ScrapeWebsiteTool` | `SERPER_API_KEY` |
| 7 | `07_file_document_tools.py` | File/document tools: read, semantic search, write (writes `topics/output/07_faq_summary.txt`) | - |
| 8 | `08_memory_in_crewai.py` | `Crew(memory=True)`; pass a message as argument and run twice | - |
| 9 | `09_short_vs_long_term_memory.py` | Short vs long-term memory: `... [remember\|recall\|short]` | - |
| 10 | `10_knowledge_source_rag.py` | Knowledge sources and RAG over a text file and a string | - |

Notes:

- **Memory (8, 9):** this project uses CrewAI 1.x, which has a single unified `Memory` class instead of separate short/long-term classes. Example 9 shows the difference by storage: a throwaway temp folder (short-term) vs a persistent `.memory/long_term` folder (long-term). Delete `.memory/` to reset.
- **Knowledge (10):** `TextFileKnowledgeSource` reads from a `knowledge/` folder in the project root, so the script copies the sample FAQ there first.
- **Cost:** every example calls the OpenAI API (and examples 7-10 also create embeddings). They are small, but not free.

## Troubleshooting

- **`Missing variable(s): ...`** - the named variable is not in `.env` or your environment (or is blank).
- **Old key still used** - a system-wide environment variable overrides `.env`. Remove it, or change `override=False` to `override=True` in `load_dotenv(...)`.
- **Import errors** - make sure the virtual environment is active and `pip install -r requirements.txt` succeeded.
