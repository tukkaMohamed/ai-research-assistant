# AI Research Assistant

An AI-powered research assistant that helps users understand and analyze research papers using Retrieval-Augmented Generation (RAG), semantic search, structured extraction, and an intelligent task router.

## Project Overview

Research papers can be long and difficult to analyze manually, especially when researchers need to compare multiple studies, identify methodologies, datasets, results, limitations, or research gaps.

The AI Research Assistant provides an interactive way to upload research paper PDFs and analyze their content through natural-language questions and research-oriented tasks.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded papers before generating an answer with the LLM. It also includes an Agent-based router that automatically determines the appropriate task for the user's request.

The application is designed to work with research papers uploaded by the user. The included papers serve as a demonstration and evaluation corpus.

## Features

- Upload research paper PDFs
- Extract text from PDF documents while preserving page information
- Split papers into overlapping text chunks
- Generate semantic embeddings for document chunks
- Store embeddings using FAISS
- Perform semantic similarity search
- Answer questions using Retrieval-Augmented Generation (RAG)
- Provide source paper and page references
- Summarize research papers
- Extract structured research information
- Extract:
  - Research task
  - Methodology
  - Models
  - Datasets
  - Evaluation metrics
  - Key results
  - Limitations
- Compare multiple research papers
- Identify common research gaps
- Generate literature-review style analysis
- Automatically route user requests to the appropriate task using an Agent
- Validate structured LLM outputs using Pydantic
- Evaluate retrieval performance using a predefined evaluation set

## Technologies Used

- Python
- Streamlit
- Google Gemini
- PyMuPDF
- Sentence Transformers
- FAISS
- Pydantic
- NumPy
- python-dotenv

## System Workflow

1. User uploads research papers.
2. PDF text is extracted using PyMuPDF.
3. Extracted text is divided into overlapping chunks.
4. Each chunk is converted into a semantic embedding.
5. Embeddings are stored in a FAISS vector index.
6. The user's request is analyzed by the Agent router.
7. Relevant chunks are retrieved from the vector database.
8. The retrieved context is provided to Gemini through a task-specific prompt.
9. The LLM generates the final response.
10. Sources and page numbers are displayed when applicable.

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd research_assistant


Create a virtual environment:
python -m venv .venv

Activate the virtual environment on Windows:
.venv\Scripts\activate

Install the required dependencies:
pip install -r requirements.txt

Create a .env file:
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.8-flash

Usage
Run the Streamlit application:
streamlit run app.py

The application provides four main areas:
- Ask — Ask questions about the research papers.
- Paper Tools — Summarize or extract structured information from a paper.
- Compare — Compare multiple research papers.
- Insights — Explore research gaps and generate literature-review style analysis.
Users can also upload new research papers directly through the application.
Evaluation
The project includes a retrieval evaluation framework using a predefined set of research questions.
The evaluation covers:
- Single-paper retrieval
- Cross-paper retrieval
- Source recall
- Keyword coverage in retrieved chunks
- Out-of-scope questions
The latest retrieval evaluation achieved:
- Single-paper source recall: 100%
- Cross-paper source recall: 75%
- Single-paper chunk keyword coverage: 67%
- Cross-paper chunk keyword coverage: 100%
The evaluation corpus is fixed to provide reproducible measurements, while the application itself supports dynamically uploaded research papers.
Demo
The application provides an interactive Streamlit interface where users can:
1. Upload research papers.
2. Ask questions about their content.
3. Summarize papers.
4. Extract structured research information.
5. Compare studies.
6. Explore research gaps.
Future Improvements
Possible future improvements include:
- Support for local and open-source LLMs
- Reranking retrieved documents
- Improved multilingual and Arabic research-paper support
- Better handling of tables, figures, and scanned PDFs
- Larger and more diverse evaluation datasets
- More advanced citation and evidence tracking
- Additional research analysis capabilities
About the Internship
This project was developed as the final project for the LLM/Generative AI internship.
The project applies concepts covered during the internship, including:
- LLM Fundamentals
- Prompt Engineering
- Chains
- Retrieval-Augmented Generation (RAG)
- Embeddings and Vector Search
- Output Parsers / Structured Output
- Agents
Additional tools and techniques were integrated where useful to build the complete application.



| Field | Information |
|---|---|
| Full Name | Tukka Mohamed Atef Ragab |
| Project Name | AI Research Assistant |
| GitHub Username | `<tukkaMohamed>` |
| Internship Batch | `<Aug-Batch>` |
| Training Program | `<LLMs>` |
| Organization | Tips Hindawi |