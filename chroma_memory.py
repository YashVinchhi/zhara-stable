# ChromaDB integration for semantic memory
import os
# Reduce threading-related crashes on Windows for tokenizers/BLAS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import platform
import time
import uuid
import torch

IS_WINDOWS = platform.system() == "Windows"

class ChromaMemory:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize Chroma client lazily to avoid import-time side effects on Windows.
        On Windows, semantic memory is disabled to avoid onnxruntime dependency issues.
        """
        self.client = None
        self.collection = None
        self.embedder = None

        if IS_WINDOWS:
            # Skip initializing chromadb on Windows to avoid onnxruntime requirement
            print("Info: Windows detected; semantic memory (ChromaDB) is disabled to avoid onnxruntime.")
            return

        # Import chromadb lazily only on non-Windows platforms
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
        except Exception as e:
            print(f"Warning: chromadb import failed; semantic memory disabled: {e}")
            return

        # Prefer persistent client to avoid ephemeral conflicts
        try:
            # chromadb >=0.4 provides PersistentClient
            self.client = chromadb.PersistentClient(path=persist_directory)
        except AttributeError:
            # Fallback for older versions
            self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        except Exception as e:
            print(f"Warning: Failed to initialize Chroma client; semantic memory disabled: {e}")
            self.client = None

        if self.client is not None:
            try:
                # Ensure a single consistent collection namespace
                self.collection = self.client.get_or_create_collection(name="chat_memory")
            except Exception as e:
                print(f"Warning: Failed to create/get Chroma collection: {e}")
                self.collection = None

        # Force or skip SentenceTransformer based on OS
        if not IS_WINDOWS:
            try:
                # Lazy import to prevent crashes
                from sentence_transformers import SentenceTransformer
                device = "cpu"
                print(f"Initializing SentenceTransformer on {device} for compatibility...")
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
                print(f"✅ SentenceTransformer initialized successfully on {device}")
            except Exception as e:
                print(f"Warning: SentenceTransformer init failed, embedding disabled: {e}")
                self.embedder = None
        else:
            print("Info: Running on Windows, skipping SentenceTransformer to avoid crashes.")

    def _is_ready(self) -> bool:
        return self.collection is not None

    def get_embedding(self, text):
        # Provide dummy embedding if embedder unavailable
        if self.embedder is None:
            # all-MiniLM-L6-v2 dimension is 384
            return [0.0] * 384
        with torch.no_grad():
            emb = self.embedder.encode(text, device="cpu", show_progress_bar=False)
            # encode may return np.ndarray; convert to list
            try:
                return emb.tolist()
            except AttributeError:
                return list(emb)

    def add_memory(self, text, metadata=None, session_id=None):
        if not self._is_ready():
            return
        unique_id = str(uuid.uuid4())
        embedding = self.get_embedding(text)

        # Add session_id and timestamp to metadata
        metadata = metadata or {}
        metadata.update({
            "session_id": session_id,
            "timestamp": time.time(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        try:
            self.collection.add(
                ids=[unique_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        except Exception as e:
            print(f"ChromaDB add error: {e}")

    def query_memory(self, text, n_results=5, session_id=None, max_age_hours=24):
        if not self._is_ready():
            return {"documents": [], "ids": [], "metadatas": []}
        embedding = self.get_embedding(text)

        # Build where clause with explicit operators for compatibility across Chroma versions
        conditions = []
        if session_id:
            conditions.append({"session_id": {"$eq": session_id}})
        cutoff_time = time.time() - (max_age_hours * 3600)
        conditions.append({"timestamp": {"$gte": cutoff_time}})

        if len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = {"$and": conditions}

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where_clause
            )
            return results
        except Exception as e:
            print(f"ChromaDB query error: {e}")
            # Fallback to query without where clause if filtering fails
            try:
                results = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=min(n_results, 3)  # Reduce results for fallback
                )
                return results
            except Exception as fallback_error:
                print(f"ChromaDB fallback query error: {fallback_error}")
                return {"documents": [], "ids": [], "metadatas": []}

    def clear_old_memories(self, max_age_days=30):
        """Clean up old memories to prevent database bloat"""
        if not self._is_ready():
            return
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 3600)

            # Get all items
            all_items = self.collection.get()

            # Find items to delete
            ids_to_delete = []
            for i, metadata in enumerate(all_items.get("metadatas", []) or []):
                if metadata and "timestamp" in metadata:
                    if metadata["timestamp"] < cutoff_time:
                        ids_to_delete.append(all_items["ids"][i])

            # Delete old items
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                print(f"Cleaned up {len(ids_to_delete)} old memories")

        except Exception as e:
            print(f"Error cleaning up old memories: {e}")

    def get_session_context(self, session_id, max_items=10):
        """Get recent conversation context for a specific session"""
        if not self._is_ready():
            return []
        try:
            # Prefer explicit filter with $eq
            results = self.collection.get(where={"session_id": {"$eq": session_id}}, limit=max_items)

            # Sort by timestamp if available
            if results.get("metadatas"):
                items = list(zip(results.get("documents", []), results.get("metadatas", [])))
                items.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
                return [item[0] for item in items[:max_items]]

            return (results.get("documents") or [])[:max_items]

        except Exception as e:
            print(f"Error getting session context: {e}")
            return []
