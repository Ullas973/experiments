from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent / "ai_science_lab.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('student', 'admin'))
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action_type TEXT NOT NULL CHECK(action_type IN ('query', 'lab', 'quiz')),
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, role FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_user(username: str, password_hash: str, role: str = "student") -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_users() -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, username, role FROM users ORDER BY username ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_user_role(user_id: int, role: str) -> None:
    if role not in {"student", "admin"}:
        raise ValueError("Role must be student or admin.")
    conn = get_connection()
    try:
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        conn.commit()
    finally:
        conn.close()


def update_user_password(user_id: int, password_hash: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_user(user_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM activity_logs WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def log_activity(user_id: int, action_type: str, content: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO activity_logs (user_id, action_type, content) VALUES (?, ?, ?)",
            (user_id, action_type, content),
        )
        conn.commit()
    finally:
        conn.close()


def get_admin_totals() -> Dict[str, int]:
    conn = get_connection()
    try:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_queries = conn.execute(
            "SELECT COUNT(*) FROM activity_logs WHERE action_type = 'query'"
        ).fetchone()[0]
        total_labs = conn.execute(
            "SELECT COUNT(*) FROM activity_logs WHERE action_type = 'lab'"
        ).fetchone()[0]
        total_quizzes = conn.execute(
            "SELECT COUNT(*) FROM activity_logs WHERE action_type = 'quiz'"
        ).fetchone()[0]
        return {
            "total_users": int(total_users),
            "total_queries": int(total_queries),
            "total_labs": int(total_labs),
            "total_quizzes": int(total_quizzes),
        }
    finally:
        conn.close()


def get_admin_user_activity_summary() -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                u.username AS username,
                COALESCE(SUM(CASE WHEN a.action_type = 'query' THEN 1 ELSE 0 END), 0) AS queries,
                COALESCE(SUM(CASE WHEN a.action_type = 'lab' THEN 1 ELSE 0 END), 0) AS labs,
                COALESCE(SUM(CASE WHEN a.action_type = 'quiz' THEN 1 ELSE 0 END), 0) AS quizzes
            FROM users u
            LEFT JOIN activity_logs a ON u.id = a.user_id
            GROUP BY u.id, u.username
            ORDER BY u.username ASC;
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_user_activity(user_id: int, action_type: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, content, timestamp
            FROM activity_logs
            WHERE user_id = ? AND action_type = ?
            ORDER BY timestamp DESC
            LIMIT 30;
            """,
            (user_id, action_type),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
