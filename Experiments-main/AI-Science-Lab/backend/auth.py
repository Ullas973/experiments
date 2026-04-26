from __future__ import annotations

import hashlib
import hmac
import os
from typing import Dict, Optional, Tuple

from db import (
    create_user,
    delete_user,
    get_user_by_id,
    get_user_by_username,
    init_db,
    update_user_password,
    update_user_role,
)


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt_bytes = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 100_000)
    return f"{salt_bytes.hex()}${digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split("$", maxsplit=1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except Exception:
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return hmac.compare_digest(actual, expected)


def ensure_default_admin() -> None:
    init_db()
    admin = get_user_by_username("admin")
    if admin is None:
        create_user("admin", _hash_password("admin123"), role="admin")


def signup_user(username: str, password: str, role: str = "student") -> Tuple[bool, str]:
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."
    if role not in {"student", "admin"}:
        return False, "Role must be student or admin."
    if get_user_by_username(username) is not None:
        return False, "Username already exists."

    create_user(username, _hash_password(password), role=role)
    return True, "Signup successful. Please login."


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    user = get_user_by_username(username.strip())
    if user is None:
        return None
    if not _verify_password(password, user["password_hash"]):
        return None
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
    }


def change_password(user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
    if not old_password or not new_password:
        return False, "Old and new passwords are required."
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    user = get_user_by_id(user_id)
    if user is None:
        return False, "User not found."
    if not _verify_password(old_password, user["password_hash"]):
        return False, "Old password is incorrect."

    update_user_password(user_id, _hash_password(new_password))
    return True, "Password updated successfully."


def admin_set_role(target_user_id: int, target_role: str, actor_user_id: int) -> Tuple[bool, str]:
    if target_role not in {"student", "admin"}:
        return False, "Invalid role."
    if target_user_id == actor_user_id:
        return False, "You cannot change your own role."

    user = get_user_by_id(target_user_id)
    if user is None:
        return False, "Target user not found."

    update_user_role(target_user_id, target_role)
    return True, f"Role updated to {target_role} for {user['username']}."


def admin_delete_user(target_user_id: int, actor_user_id: int) -> Tuple[bool, str]:
    if target_user_id == actor_user_id:
        return False, "You cannot delete your own account."

    user = get_user_by_id(target_user_id)
    if user is None:
        return False, "Target user not found."
    if user["username"] == "admin":
        return False, "Default admin cannot be deleted."

    delete_user(target_user_id)
    return True, f"Deleted user {user['username']}."
