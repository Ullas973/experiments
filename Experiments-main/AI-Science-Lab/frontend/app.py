from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
DATA_DIR = ROOT / "data"
sys.path.append(str(BACKEND_DIR))

from rag_pipeline import RAGPipeline  # noqa: E402
from lab_generator import LabGenerator  # noqa: E402
from quiz_generator import QuizGenerator  # noqa: E402


st.set_page_config(page_title="AI Science Lab", page_icon="🔬", layout="wide")
st.title("AI Science Lab – Curriculum-Driven Intelligent Learning Platform")


def init_state() -> None:
    if "rag" not in st.session_state:
        st.session_state.rag = RAGPipeline(
            embedding_model="all-MiniLM-L6-v2",
            llm_mode="openai",
            llm_model="gpt-4o-mini",
        )
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False
    if "lab_gen" not in st.session_state:
        st.session_state.lab_gen = LabGenerator(st.session_state.rag)
    if "quiz_gen" not in st.session_state:
        st.session_state.quiz_gen = QuizGenerator(st.session_state.rag)


def render_sources(sources: list[dict]) -> None:
    for i, src in enumerate(sources, start=1):
        st.markdown(
            f"**Source {i}:** `{src.get('source', 'unknown')}` "
            f"(chunk `{src.get('chunk_id', 'n/a')}`)"
        )
        st.info(f"Source: {src.get('snippet', '')}")


init_state()

with st.sidebar:
    st.header("Upload Textbook (PDF)")
    uploaded_pdf = st.file_uploader("Choose a textbook PDF", type=["pdf"])
    process_clicked = st.button("Process PDF")

    st.markdown("---")
    st.caption("Set your OpenAI key in environment variable `OPENAI_API_KEY`.")

if process_clicked:
    if not uploaded_pdf:
        st.error("Please upload a PDF first.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.getvalue())
            temp_pdf_path = tmp.name
        try:
            result = st.session_state.rag.build_index_from_pdf(temp_pdf_path)
            st.session_state.index_ready = True
            st.success(
                f"Indexed {result['chunks_indexed']} chunks "
                f"(dim={result['embedding_dim']})."
            )
        except Exception as exc:
            st.session_state.index_ready = False
            st.error(f"Failed to process PDF: {exc}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

st.subheader("Query System (RAG Answer)")
query = st.text_input("Ask a textbook-based question or topic")
ask_clicked = st.button("Get Explanation")

if ask_clicked:
    if not st.session_state.index_ready:
        st.warning("Upload and process a PDF before querying.")
    elif not query.strip():
        st.warning("Enter a query first.")
    else:
        try:
            result = st.session_state.rag.answer_query(query=query, top_k=4)
            st.markdown("### Explanation")
            st.write(result["answer"])
            st.markdown("### Retrieved Sources")
            render_sources(result["sources"])
        except Exception as exc:
            st.error(f"Query failed: {exc}")

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Generate Lab"):
        if not st.session_state.index_ready:
            st.warning("Upload and process a PDF first.")
        elif not query.strip():
            st.warning("Enter a concept/query first.")
        else:
            try:
                result = st.session_state.lab_gen.generate_lab(query, top_k=4)
                if result.get("error"):
                    st.error(result["error"])
                else:
                    lab = result["lab"]
                    st.markdown("### Problem Statement")
                    st.write(lab.get("problem_statement", ""))
                    st.markdown("### Dataset")
                    st.write(lab.get("dataset", ""))
                    st.markdown("### Python Code")
                    st.code(lab.get("python_code", ""), language="python")
                    st.markdown("### Explanation")
                    st.write(lab.get("explanation", ""))
                    st.markdown("### Retrieved Sources")
                    render_sources(result["sources"])
            except Exception as exc:
                st.error(f"Lab generation failed: {exc}")

with col2:
    if st.button("Generate Quiz"):
        if not st.session_state.index_ready:
            st.warning("Upload and process a PDF first.")
        elif not query.strip():
            st.warning("Enter a concept/query first.")
        else:
            try:
                result = st.session_state.quiz_gen.generate_quiz(query, top_k=4)
                if result.get("error"):
                    st.error(result["error"])
                else:
                    quiz = result["quiz"]
                    st.markdown("### MCQs")
                    for i, mcq in enumerate(quiz.get("mcqs", []), start=1):
                        st.write(f"**Q{i}.** {mcq.get('question', '')}")
                        for option in mcq.get("options", []):
                            st.write(f"- {option}")
                        st.write(f"**Answer:** {mcq.get('answer', '')}")
                        st.write("")
                    st.markdown("### Short Questions")
                    for i, q in enumerate(quiz.get("short_questions", []), start=1):
                        st.write(f"{i}. {q}")
                    st.markdown("### Retrieved Sources")
                    render_sources(result["sources"])
            except Exception as exc:
                st.error(f"Quiz generation failed: {exc}")

with col3:
    if st.button("Show Pre-built Labs"):
        labs_file = DATA_DIR / "labs.json"
        if not labs_file.exists():
            st.error("labs.json not found.")
        else:
            with open(labs_file, "r", encoding="utf-8") as f:
                labs = json.load(f)
            st.markdown("### Pre-generated Labs")
            topic_names = [item["topic"] for item in labs]
            selected = st.selectbox("Select a lab topic", topic_names)
            picked = next((item for item in labs if item["topic"] == selected), None)
            if picked:
                st.write(picked.get("description", ""))
                st.code(picked.get("code", ""), language="python")
