"""
Centralized Application State Management
Provides singleton instances for consistent dependency injection
"""

from typing import Optional
from pathlib import Path
from session_manager import SessionManager
from local_cache import LocalCache
import config
import logging

logger = logging.getLogger(__name__)

class AppState:
    """Centralized application state container"""
    _instance: Optional['AppState'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.session_manager: Optional[SessionManager] = None
            self.local_cache: Optional[LocalCache] = None
            self.tts_service = None
            self._initialized = True

    def get_session_manager(self) -> SessionManager:
        """Get or create session manager instance"""
        if self.session_manager is None:
            self.session_manager = SessionManager()
            logger.info("Session manager initialized")
        return self.session_manager

    def get_local_cache(self) -> LocalCache:
        """Get or create local cache instance"""
        if self.local_cache is None:
            cache_path = str(Path(config.STORAGE_DIR) / "response_cache.json")
            self.local_cache = LocalCache(cache_path)
            logger.info("Local cache initialized")
        return self.local_cache

# Global singleton instance
app_state = AppState()
