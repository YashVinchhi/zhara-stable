import json
import os
import time
from typing import Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path

# Conditional import for fcntl for cross-platform compatibility
try:
    import fcntl
except ImportError:
    fcntl = None

class LocalCache:
    def __init__(self, cache_file: str = "response_cache.json", ttl_hours: int = 24, 
                 save_interval: int = 300, max_entries: int = 1000):
        self.cache_file = cache_file
        self.ttl_hours = ttl_hours
        self.save_interval = save_interval
        self.max_entries = max_entries
        self.cache: Dict[str, dict] = {}
        self.last_save = time.time()
        self.modified = False
        self._ensure_cache_dir()
        self.load_cache()

    def _ensure_cache_dir(self):
        """Ensure the cache directory exists"""
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _acquire_lock(self, file_obj):
        """Acquire an exclusive lock on the file if fcntl is available"""
        if fcntl:
            try:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            except (AttributeError, OSError) as e:
                print(f"Warning: File locking not supported: {e}")

    def _release_lock(self, file_obj):
        """Release the file lock if fcntl is available"""
        if fcntl:
            try:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
            except (AttributeError, OSError) as e:
                print(f"Warning: File locking not supported: {e}")

    def load_cache(self):
        """Load cache from disk if it exists"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._acquire_lock(f)
                    try:
                        self.cache = json.load(f)
                    finally:
                        self._release_lock(f)
                self._cleanup_expired()
                self._enforce_size_limit()
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.cache = {}

    def save_cache(self, force: bool = False):
        """Save cache to disk if modified or forced"""
        current_time = time.time()
        if not force and not self.modified:
            return
        if not force and current_time - self.last_save < self.save_interval:
            return

        temp_file = f"{self.cache_file}.tmp"
        try:
            # Save to temporary file first
            with open(temp_file, 'w', encoding='utf-8') as f:
                self._acquire_lock(f)
                try:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
                finally:
                    self._release_lock(f)

            # Atomic rename
            os.replace(temp_file, self.cache_file)
            self.last_save = current_time
            self.modified = False

        except Exception as e:
            print(f"Error saving cache: {e}")
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass

    def _enforce_size_limit(self):
        """Enforce the maximum number of entries in the cache"""
        if len(self.cache) > self.max_entries:
            # Sort by timestamp and keep only the most recent entries
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: datetime.fromisoformat(x[1]['timestamp']),
                reverse=True
            )
            self.cache = dict(sorted_entries[:self.max_entries])
            self.modified = True

    def get(self, key: str) -> Optional[dict]:
        """Get a value from cache if it exists and is not expired"""
        if key in self.cache:
            entry = self.cache[key]
            expiry_time = datetime.fromisoformat(entry['timestamp']) + timedelta(hours=self.ttl_hours)

            if expiry_time > datetime.now():
                return entry['data']
            else:
                del self.cache[key]
                self.modified = True
                # Don't save immediately, wait for save_interval or explicit save
        return None

    def set(self, key: str, value: dict):
        """Set a value in the cache with current timestamp"""
        self.cache[key] = {
            'timestamp': datetime.now().isoformat(),
            'data': value
        }
        self.modified = True
        
        # Enforce size limit
        self._enforce_size_limit()

        # Periodic cleanup and save
        if time.time() - self.last_save >= self.save_interval:
            self._cleanup_expired()
            self.save_cache()

    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        now = datetime.now()
        expiry_threshold = now - timedelta(hours=self.ttl_hours)

        expired = [
            k for k, v in self.cache.items()
            if datetime.fromisoformat(v['timestamp']) <= expiry_threshold
        ]

        if expired:
            for k in expired:
                del self.cache[k]
            self.modified = True
            self.save_cache(force=True)

    def __del__(self):
        """Ensure cache is saved when object is destroyed"""
        if self.modified:
            self.save_cache(force=True)
