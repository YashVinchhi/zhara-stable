"""
Session Management System for Zhara AI Assistant
Handles session creation, management, and persistence with improved efficiency
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class SessionMetadata:
    """Session metadata structure"""
    session_id: str
    title: str
    created_at: str
    last_updated: str
    message_count: int
    model_used: str = "default"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class SessionMessage:
    """Individual message in a session"""
    id: str
    session_id: str
    user_message: str
    ai_response: str
    timestamp: str
    model_used: str
    audio_file: Optional[str] = None
    viseme_file: Optional[str] = None

class SessionManager:
    """
    Dedicated session manager that uses lightweight metadata storage
    and separates session data from ChromaDB operations
    """

    def __init__(self, storage_dir: str = "storage"):
        self.storage_path = Path(storage_dir)
        self.storage_path.mkdir(exist_ok=True)

        self.sessions_file = self.storage_path / "sessions.json"
        self.messages_dir = self.storage_path / "messages"
        self.messages_dir.mkdir(exist_ok=True)

        self._sessions_cache: Dict[str, SessionMetadata] = {}
        self._load_sessions()

    def _load_sessions(self):
        """Load sessions from disk into memory cache"""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    sessions_data = json.load(f)

                for session_id, session_dict in sessions_data.items():
                    self._sessions_cache[session_id] = SessionMetadata(**session_dict)

                logger.info(f"Loaded {len(self._sessions_cache)} sessions from storage")
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            self._sessions_cache = {}

    def _save_sessions(self):
        """Save sessions cache to disk"""
        try:
            sessions_data = {
                session_id: asdict(metadata)
                for session_id, metadata in self._sessions_cache.items()
            }

            with open(self.sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving sessions: {e}")

    def create_session(self, title: Optional[str] = None) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        if not title:
            title = f"Chat Session {len(self._sessions_cache) + 1}"

        session_metadata = SessionMetadata(
            session_id=session_id,
            title=title,
            created_at=current_time,
            last_updated=current_time,
            message_count=0
        )

        self._sessions_cache[session_id] = session_metadata
        self._save_sessions()

        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """Get session metadata by ID"""
        return self._sessions_cache.get(session_id)

    def update_session(self, session_id: str, title: Optional[str] = None,
                      model_used: Optional[str] = None) -> bool:
        """Update session metadata"""
        if session_id not in self._sessions_cache:
            return False

        session = self._sessions_cache[session_id]

        if title:
            session.title = title
        if model_used:
            session.model_used = model_used

        session.last_updated = datetime.now().isoformat()
        self._save_sessions()

        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages"""
        if session_id not in self._sessions_cache:
            return False

        # Remove session metadata
        del self._sessions_cache[session_id]

        # Remove messages file
        messages_file = self.messages_dir / f"{session_id}.json"
        if messages_file.exists():
            messages_file.unlink()

        self._save_sessions()
        logger.info(f"Deleted session: {session_id}")
        return True

    def list_sessions(self, limit: Optional[int] = None) -> List[SessionMetadata]:
        """List all sessions, optionally limited"""
        sessions = list(self._sessions_cache.values())

        # Sort by last updated, most recent first
        sessions.sort(key=lambda s: s.last_updated, reverse=True)

        if limit:
            sessions = sessions[:limit]

        return sessions

    def add_message(self, session_id: str, user_message: str, ai_response: str,
                   model_used: str, audio_file: Optional[str] = None,
                   viseme_file: Optional[str] = None) -> str:
        """Add a message to a session"""
        if session_id not in self._sessions_cache:
            # Auto-create session if it doesn't exist
            self.create_session(f"Auto Session {session_id[:8]}")

        message_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        message = SessionMessage(
            id=message_id,
            session_id=session_id,
            user_message=user_message,
            ai_response=ai_response,
            timestamp=current_time,
            model_used=model_used,
            audio_file=audio_file,
            viseme_file=viseme_file
        )

        # Save message to session file
        messages_file = self.messages_dir / f"{session_id}.json"
        messages = []

        if messages_file.exists():
            try:
                with open(messages_file, 'r') as f:
                    messages = json.load(f)
            except Exception as e:
                logger.error(f"Error loading messages for session {session_id}: {e}")

        messages.append(asdict(message))

        try:
            with open(messages_file, 'w') as f:
                json.dump(messages, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving message to session {session_id}: {e}")
            return message_id

        # Update session metadata
        session = self._sessions_cache[session_id]
        session.message_count = len(messages)
        session.last_updated = current_time
        session.model_used = model_used

        self._save_sessions()
        return message_id

    def get_session_messages(self, session_id: str,
                           limit: Optional[int] = None) -> List[SessionMessage]:
        """Get messages for a session"""
        messages_file = self.messages_dir / f"{session_id}.json"

        if not messages_file.exists():
            return []

        try:
            with open(messages_file, 'r') as f:
                messages_data = json.load(f)

            messages = [SessionMessage(**msg_dict) for msg_dict in messages_data]

            if limit:
                messages = messages[-limit:]  # Get most recent messages

            return messages

        except Exception as e:
            logger.error(f"Error loading messages for session {session_id}: {e}")
            return []

    def get_session_stats(self) -> Dict[str, Any]:
        """Get overall session statistics"""
        total_sessions = len(self._sessions_cache)
        total_messages = sum(session.message_count for session in self._sessions_cache.values())

        # Get model usage stats
        model_usage = {}
        for session in self._sessions_cache.values():
            model = session.model_used
            model_usage[model] = model_usage.get(model, 0) + session.message_count

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "model_usage": model_usage,
            "active_sessions": len([s for s in self._sessions_cache.values()
                                   if s.message_count > 0])
        }

    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days"""
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days_old)
        sessions_to_delete = []

        for session_id, session in self._sessions_cache.items():
            try:
                last_updated = datetime.fromisoformat(session.last_updated)
                if last_updated < cutoff_date:
                    sessions_to_delete.append(session_id)
            except Exception:
                continue

        deleted_count = 0
        for session_id in sessions_to_delete:
            if self.delete_session(session_id):
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old sessions")
        return deleted_count
