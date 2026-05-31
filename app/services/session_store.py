import os
import sqlite3
import threading
import time
import uuid
from typing import Dict, List

from loguru import logger


def now_ms() -> int:
    return int(time.time() * 1000)


def now() -> int:
    return int(time.time())


class SessionStore:
    def __init__(self, path: str, expiry_days: int = 30):
        self.path = path
        self.expiry_days = expiry_days
        self._db_lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._db_lock:
            conn = self._connect()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT '新对话',
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        history TEXT NOT NULL DEFAULT '[]',
                        summary TEXT NOT NULL DEFAULT ''
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_updated
                    ON sessions(updated_at DESC)
                """)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _to_seconds(timestamp: int, current_time_sec: int) -> int:
        return timestamp // 1000 if timestamp > current_time_sec * 1000 else timestamp

    @staticmethod
    def _history_to_messages(history) -> List[Dict[str, str]]:
        messages = []
        for question, answer in SessionStore._normalize_history(history):
            messages.extend([
                {"role": "user", "content": question or ""},
                {"role": "assistant", "content": answer or ""},
            ])
        return messages

    @staticmethod
    def _normalize_history(history) -> List[List[str]]:
        if not isinstance(history, list):
            return []
        normalized_history = []
        for pair in history:
            if isinstance(pair, list) and len(pair) == 2:
                normalized_history.append([
                    "" if pair[0] is None else str(pair[0]),
                    "" if pair[1] is None else str(pair[1]),
                ])
        return normalized_history

    def is_session_expired(self, session_data: Dict) -> bool:
        created_at = session_data.get("created_at", 0)
        if created_at == 0:
            return False
        current_time_sec = now()
        created_at_sec = self._to_seconds(created_at, current_time_sec)
        expiry_seconds = self.expiry_days * 24 * 60 * 60
        return current_time_sec - created_at_sec > expiry_seconds

    def load(self) -> Dict[str, Dict]:
        try:
            with self._db_lock:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        "SELECT id, title, created_at, updated_at, history, summary FROM sessions"
                    ).fetchall()
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning(f"failed to load sessions from SQLite: {exc}")
            return {}

        sessions = {}
        for row in rows:
            try:
                history = self._normalize_history(json_loads_safe(row["history"]))
            except Exception:
                history = []

            session_data = {
                "title": row["title"] or "新对话",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "history": history,
                "summary": row["summary"] or "",
            }
            if self.is_session_expired(session_data):
                continue
            sessions[row["id"]] = session_data

        return sessions

    def save(self, sessions: Dict[str, Dict]) -> None:
        valid_sessions = {
            sid: data
            for sid, data in sessions.items()
            if not self.is_session_expired(data)
        }
        try:
            with self._db_lock:
                conn = self._connect()
                try:
                    conn.execute("DELETE FROM sessions")
                    rows = [
                        (
                            sid,
                            data.get("title", "新对话"),
                            data.get("created_at", now_ms()),
                            data.get("updated_at", now_ms()),
                            json_dumps_safe(data.get("history", [])),
                            data.get("summary", ""),
                        )
                        for sid, data in valid_sessions.items()
                    ]
                    conn.executemany(
                        "INSERT INTO sessions (id, title, created_at, updated_at, history, summary) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        rows,
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            logger.error(f"failed to save sessions: {exc}")

    def ensure_session(self, sessions: Dict[str, Dict], session_id: str) -> Dict:
        current_ms = now_ms()
        if session_id not in sessions:
            sessions[session_id] = {
                "title": "新对话",
                "updated_at": current_ms,
                "created_at": current_ms,
                "history": [],
                "summary": "",
            }
        else:
            sessions[session_id]["updated_at"] = current_ms
            sessions[session_id].setdefault("created_at", current_ms)
            sessions[session_id].setdefault("summary", "")
        return sessions[session_id]

    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        try:
            uuid.UUID(session_id)
            return True
        except ValueError:
            return False

    def list_sessions(self, sessions: Dict[str, Dict]) -> List[Dict]:
        items = [
            {
                "id": sid,
                "title": info.get("title", "新对话"),
                "updated_at": info.get("updated_at", 0),
                "message_count": len(info.get("history", [])) * 2,
            }
            for sid, info in sessions.items()
        ]
        items.sort(key=lambda item: item["updated_at"], reverse=True)
        return items

    def get_session_payload(self, sessions: Dict[str, Dict], session_id: str) -> Dict:
        if session_id not in sessions:
            raise KeyError(session_id)
        session = sessions[session_id]
        if not isinstance(session, dict):
            session = {
                "title": "新对话",
                "summary": "",
                "history": self._normalize_history(session),
            }
            sessions[session_id] = session
        return {
            "id": session_id,
            "title": session.get("title", "新对话"),
            "summary": session.get("summary", ""),
            "messages": self._history_to_messages(session.get("history", [])),
        }

    def create_session(self, sessions: Dict[str, Dict], session_id: str) -> Dict:
        session = self.ensure_session(sessions, session_id)
        self.save(sessions)
        return session

    def rename_session(self, sessions: Dict[str, Dict], session_id: str, title: str) -> None:
        if session_id not in sessions:
            raise KeyError(session_id)
        sessions[session_id]["title"] = title[:100]
        sessions[session_id]["updated_at"] = now_ms()
        self.save(sessions)

    def delete_session(self, sessions: Dict[str, Dict], session_id: str) -> bool:
        removed = sessions.pop(session_id, None)
        if removed is not None:
            self.save(sessions)
            return True
        return False

    def clear_sessions(self, sessions: Dict[str, Dict]) -> None:
        sessions.clear()
        self.save(sessions)

    def store_chat_turn(self, sessions: Dict[str, Dict], session_id: str, message: str, answer: str) -> List[List[str]]:
        session = self.ensure_session(sessions, session_id)
        final_history = session.get("history", []) + [[message, answer]]
        session["history"] = final_history
        if session["title"] == "新对话":
            session["title"] = message[:24]
        session["updated_at"] = now_ms()
        self.save(sessions)
        return final_history

    def apply_compression_result(
        self,
        sessions: Dict[str, Dict],
        session_id: str,
        history_to_compress,
        updated_history,
        updated_summary: str,
    ) -> bool:
        if session_id not in sessions:
            return False
        current_session_history = sessions[session_id].get("history", [])
        if current_session_history != history_to_compress:
            return False
        sessions[session_id]["history"] = updated_history
        sessions[session_id]["summary"] = updated_summary
        self.save(sessions)
        return True


def json_dumps_safe(obj) -> str:
    try:
        import json
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "[]"


def json_loads_safe(text) -> object:
    try:
        import json
        return json.loads(text)
    except Exception:
        return []
