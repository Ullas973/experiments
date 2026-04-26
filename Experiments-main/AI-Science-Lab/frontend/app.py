from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DIR = ROOT_DIR / "data"
sys.path.append(str(BACKEND_DIR))

from auth import (  # noqa: E402
    admin_delete_user,
    admin_set_role,
    authenticate_user,
    change_password,
    ensure_default_admin,
    signup_user,
)
from db import get_admin_totals, get_admin_user_activity_summary, get_user_activity, list_users  # noqa: E402
from progress import log_lab, log_query, log_quiz  # noqa: E402

try:
    from lab_generator import LabGenerator  # noqa: E402
    from quiz_generator import QuizGenerator  # noqa: E402
    from rag_pipeline import RAGPipeline  # noqa: E402

    RAG_IMPORT_OK = True
except Exception:
    RAG_IMPORT_OK = False


st.set_page_config(page_title="AI Science Lab", page_icon="🔬", layout="wide")
st.title("AI Science Lab – Curriculum-Driven Intelligent Learning Platform")


def init_auth() -> None:
    ensure_default_admin()
    if "user" not in st.session_state:
        st.session_state.user = None


def init_rag_state() -> None:
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False
    if "rag" not in st.session_state and RAG_IMPORT_OK:
        st.session_state.rag = RAGPipeline(
            embedding_model="all-MiniLM-L6-v2",
            llm_mode="openai",
            llm_model="gpt-4o-mini",
        )
    if "lab_generator" not in st.session_state and RAG_IMPORT_OK:
        st.session_state.lab_generator = LabGenerator(st.session_state.rag)
    if "quiz_generator" not in st.session_state and RAG_IMPORT_OK:
        st.session_state.quiz_generator = QuizGenerator(st.session_state.rag)


def render_auth_ui() -> None:
    tab_login, tab_signup = st.tabs(["Login", "Signup"])

    with tab_login:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn"):
            user = authenticate_user(username, password)
            if not user:
                st.error("Invalid username or password.")
            else:
                st.session_state.user = user
                st.success(f"Welcome, {user['username']} ({user['role']}).")
                st.rerun()

    with tab_signup:
        st.subheader("Signup")
        username = st.text_input("New username", key="signup_username")
        password = st.text_input("New password", type="password", key="signup_password")
        role = st.selectbox("Role", ["student", "admin"], key="signup_role")
        if st.button("Create account", key="signup_btn"):
            ok, message = signup_user(username, password, role)
            if ok:
                st.success(message)
            else:
                st.error(message)

    st.info("Default admin credentials: username `admin`, password `admin123`.")


def render_admin_dashboard() -> None:
    st.header("Admin Dashboard")
    totals = get_admin_totals()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users", totals["total_users"])
    c2.metric("Total Queries", totals["total_queries"])
    c3.metric("Total Labs Generated", totals["total_labs"])
    c4.metric("Total Quizzes Attempted", totals["total_quizzes"])

    st.markdown("---")
    st.subheader("Queries Per User")
    rows = get_admin_user_activity_summary()
    if rows:
        chart_df = pd.DataFrame(rows)[["username", "queries"]].set_index("username")
        st.bar_chart(chart_df)
    else:
        st.write("No user activity yet.")

    st.subheader("User Activity Summary (Sortable)")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.write("No user activity yet.")

    st.markdown("---")
    st.subheader("User Management")
    users = list_users()
    if not users:
        st.info("No users found.")
        return

    user_map = {f"{u['username']} ({u['role']})": u for u in users}
    selected_label = st.selectbox("Select user", list(user_map.keys()), key="admin_user_select")
    selected_user = user_map[selected_label]

    manage_col1, manage_col2 = st.columns(2)
    with manage_col1:
        target_role = st.selectbox(
            "Set role",
            ["student", "admin"],
            index=0 if selected_user["role"] == "student" else 1,
            key="admin_set_role_value",
        )
        if st.button("Update Role", type="primary", key="admin_update_role_btn"):
            ok, message = admin_set_role(
                target_user_id=int(selected_user["id"]),
                target_role=target_role,
                actor_user_id=int(st.session_state.user["id"]),
            )
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with manage_col2:
        st.warning("Deleting a user also deletes their activity logs.")
        if st.button("Delete User", key="admin_delete_user_btn"):
            ok, message = admin_delete_user(
                target_user_id=int(selected_user["id"]),
                actor_user_id=int(st.session_state.user["id"]),
            )
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.error(message)


def render_student_dashboard(user_id: int) -> None:
    st.header("Student Dashboard")
    q_rows = get_user_activity(user_id, "query")
    l_rows = get_user_activity(user_id, "lab")
    z_rows = get_user_activity(user_id, "quiz")

    c1, c2, c3 = st.columns(3)
    c1.metric("Your Queries", len(q_rows))
    c2.metric("Labs Generated", len(l_rows))
    c3.metric("Quizzes Attempted", len(z_rows))

    st.subheader("Your Past Queries")
    if q_rows:
        st.table(q_rows)
    else:
        st.write("No query history.")

    st.subheader("Your Labs")
    if l_rows:
        st.table(l_rows)
    else:
        st.write("No lab activity yet.")

    st.subheader("Your Quizzes")
    if z_rows:
        st.table(z_rows)
    else:
        st.write("No quiz activity yet.")


def render_change_password(user_id: int) -> None:
    st.subheader("Change Password")
    col1, col2 = st.columns(2)
    with col1:
        old_password = st.text_input("Old password", type="password", key="old_password")
    with col2:
        new_password = st.text_input("New password", type="password", key="new_password")
    if st.button("Change Password", type="primary", key="change_password_btn"):
        ok, message = change_password(user_id, old_password, new_password)
        if ok:
            st.success(message)
        else:
            st.error(message)


def render_sources(sources: list[dict]) -> None:
    for idx, item in enumerate(sources, start=1):
        st.markdown(
            f"**Source {idx}:** `{item.get('source', 'unknown')}` "
            f"(chunk `{item.get('chunk_id', 'n/a')}`)"
        )
        st.info(f"Source: {item.get('snippet', '')}")


def render_rag_interface(user_id: int) -> None:
    st.header("RAG Learning Interface")
    if not RAG_IMPORT_OK:
        st.error(
            "RAG modules are not available. Ensure backend files exist: "
            "rag_pipeline.py, lab_generator.py, quiz_generator.py."
        )
        return

    with st.sidebar:
        st.subheader("Upload Textbook PDF")
        uploaded_pdf = st.file_uploader("Choose PDF", type=["pdf"], key="pdf_upload")
        process = st.button("Process PDF", type="primary", key="process_pdf_btn")

    if process:
        if not uploaded_pdf:
            st.warning("Please upload a PDF first.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.getvalue())
                tmp_path = tmp_file.name
            try:
                out = st.session_state.rag.build_index_from_pdf(tmp_path)
                st.session_state.index_ready = True
                st.success(
                    f"Indexed {out['chunks_indexed']} chunks "
                    f"(embedding_dim={out['embedding_dim']})."
                )
            except Exception as exc:
                st.session_state.index_ready = False
                st.error(f"PDF processing failed. Please try another file. Details: {exc}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.markdown("---")
    st.subheader("Query and Generation")
    query = st.text_input("Ask a textbook-based question or enter a topic")

    if st.button("Get Explanation", type="primary", key="explain_btn"):
        if not st.session_state.index_ready:
            st.warning("Upload and process a PDF first.")
        elif not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                result = st.session_state.rag.answer_query(query, top_k=4)
                if not result.get("sources"):
                    st.warning(
                        "No relevant chunks retrieved. Try a more specific question "
                        "or upload a better textbook PDF."
                    )
                log_query(user_id, query)
                st.subheader("Explanation")
                st.write(result.get("answer", "No answer generated."))
                st.subheader("Retrieved Context Sources")
                render_sources(result.get("sources", []))
            except Exception as exc:
                st.error(f"Query failed due to API or retrieval error: {exc}")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Generate Lab", type="primary", key="lab_btn"):
            if not st.session_state.index_ready:
                st.warning("Upload and process a PDF first.")
            elif not query.strip():
                st.warning("Enter a concept/topic first.")
            else:
                try:
                    result = st.session_state.lab_generator.generate_lab(query, top_k=4)
                    if not result.get("sources"):
                        st.warning("No context retrieved for lab generation.")
                    log_lab(user_id, query)
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
                        st.markdown("### Sources")
                        render_sources(result.get("sources", []))
                except Exception as exc:
                    st.error(f"Lab generation failed due to API or retrieval error: {exc}")

    with c2:
        if st.button("Generate Quiz", type="primary", key="quiz_btn"):
            if not st.session_state.index_ready:
                st.warning("Upload and process a PDF first.")
            elif not query.strip():
                st.warning("Enter a concept/topic first.")
            else:
                try:
                    result = st.session_state.quiz_generator.generate_quiz(query, top_k=4)
                    if not result.get("sources"):
                        st.warning("No context retrieved for quiz generation.")
                    log_quiz(user_id, query)
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        quiz = result["quiz"]
                        st.markdown("### MCQs")
                        for idx, item in enumerate(quiz.get("mcqs", []), start=1):
                            st.write(f"**Q{idx}.** {item.get('question', '')}")
                            for option in item.get("options", []):
                                st.write(f"- {option}")
                            st.write(f"**Answer:** {item.get('answer', '')}")
                        st.markdown("### Short Questions")
                        for idx, item in enumerate(quiz.get("short_questions", []), start=1):
                            st.write(f"{idx}. {item}")
                        st.markdown("### Sources")
                        render_sources(result.get("sources", []))
                except Exception as exc:
                    st.error(f"Quiz generation failed due to API or retrieval error: {exc}")

    with c3:
        if st.button("Show Pre-built Labs", key="show_prebuilt_btn"):
            labs_path = DATA_DIR / "labs.json"
            if not labs_path.exists():
                st.error("data/labs.json not found.")
            else:
                with open(labs_path, "r", encoding="utf-8") as f:
                    labs = json.load(f)
                selected_topic = st.selectbox(
                    "Select pre-built lab topic",
                    [item["topic"] for item in labs],
                    key="prebuilt_select",
                )
                selected = next((item for item in labs if item["topic"] == selected_topic), None)
                if selected:
                    st.write(selected.get("description", ""))
                    st.code(selected.get("code", ""), language="python")


init_auth()
init_rag_state()

if st.session_state.user is None:
    render_auth_ui()
else:
    user = st.session_state.user
    st.sidebar.success(f"Logged in as {user['username']} ({user['role']})")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.index_ready = False
        st.rerun()
    st.sidebar.markdown("---")
    render_change_password(user["id"])

    if user["role"] == "admin":
        render_admin_dashboard()
        st.markdown("---")
        with st.expander("Open RAG Interface"):
            render_rag_interface(user["id"])
    else:
        render_student_dashboard(user["id"])
        st.markdown("---")
        render_rag_interface(user["id"])
