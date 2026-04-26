# AI Science Lab – Curriculum-Driven Intelligent Learning Platform

This project implements a strict Retrieval-Augmented Generation (RAG) workflow for textbook-grounded science learning.

## What this project does

- Upload a textbook PDF
- Extract and chunk textbook text
- Embed chunks using `sentence-transformers`
- Store and search chunks in FAISS
- Answer user queries using retrieved chunks only
- Generate labs and quizzes from retrieved context
- Show explicit source snippets
- Provide pre-built labs from `data/labs.json`

## Project Structure

```text
AI-Science-Lab/
  backend/
    rag_pipeline.py
    embedding.py
    vector_db.py
    lab_generator.py
    quiz_generator.py
    requirements.txt
  frontend/
    app.py
  data/
    labs.json
  README.md
```

## Setup

1. Open terminal in `AI-Science-Lab`
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

4. Set OpenAI API key (required for LLM generation):

- Windows PowerShell:
  ```powershell
  $env:OPENAI_API_KEY="your_api_key_here"
  ```

## Run

From `AI-Science-Lab` folder:

```bash
streamlit run frontend/app.py
```

## Usage Flow

1. Upload PDF in sidebar.
2. Click **Process PDF** to build embeddings + FAISS index.
3. Enter a query and click **Get Explanation**.
4. Click:
   - **Generate Lab** for a context-grounded coding lab
   - **Generate Quiz** for 5 MCQs + 2 short questions
   - **Show Pre-built Labs** to browse local labs from JSON

## Important RAG Rule Implemented

- Answers, labs, and quizzes are all retrieval-first.
- If no context is retrieved, the app returns an error/status instead of generating generic output.

## Notes

- `llm_mode="openai"` is active by default in `frontend/app.py`.
- If OpenAI key is unavailable, lab/quiz modules use fallback templates still tied to retrieved snippets.
