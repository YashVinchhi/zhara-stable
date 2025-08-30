"""
Zhara AI Assistant - Refactored Main Application
Modular architecture with separated concerns and improved maintainability
"""

import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
import platform  # Added to detect OS for TTS GPU disabling

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Modular imports - avoiding wildcard imports
from config import (
    TTS_MODEL_PATH, AUDIO_OUTPUT_DIR,
    VISEME_OUTPUT_DIR, HOST, PORT
)
from utils import SystemInfo, FileManager, MemoryManager
from tts_service import initialize_tts_service, get_tts_service
from api_router import router as api_router
# from chroma_memory import ChromaMemory  # Not initialized here to avoid duplicate client
from local_cache import LocalCache
from app_state import app_state

# Setup structured logging with UTF-8 encoding support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zhara.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Application state container is imported from app_state module

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with proper startup/shutdown"""
    # Startup
    logger.info("Starting Zhara AI Assistant...")

    # Detect system environment
    is_sbc, sbc_type = SystemInfo.detect_sbc_environment()
    gpu_info = SystemInfo.detect_gpu_info()

    if is_sbc:
        logger.info(f"SBC environment detected: {sbc_type}")

    # Initialize directories
    FileManager.ensure_directory_exists(AUDIO_OUTPUT_DIR)
    FileManager.ensure_directory_exists(VISEME_OUTPUT_DIR)

    # Initialize global services
    session_manager = app_state.get_session_manager()
    local_cache = app_state.get_local_cache()

    try:
        # Initialize lightweight local cache (optional)
        from config import STORAGE_DIR
        app_state.local_cache = LocalCache(str(Path(STORAGE_DIR) / "response_cache.json"))
        logger.info("Local cache initialized")

        # Initialize TTS service with background workers
        max_workers = 1 if is_sbc else 2
        # Disable GPU for TTS on Windows due to potential access violations
        gpu_for_tts = gpu_info["available"] and platform.system() != "Windows"
        tts_service = initialize_tts_service(
            TTS_MODEL_PATH,
            gpu_enabled=gpu_for_tts,
            max_workers=max_workers
        )
        app_state.tts_service = tts_service
        logger.info(f"TTS service initialized with {max_workers} workers")

        # Periodic memory cleanup for better performance
        asyncio.create_task(periodic_cleanup())

        logger.info("Zhara AI Assistant started successfully!")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Zhara AI Assistant...")

    try:
        # Cleanup TTS service
        tts_service = get_tts_service()
        tts_service.shutdown()

        # Cleanup memory
        MemoryManager.cleanup_gpu_memory()

        logger.info("Shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

async def periodic_cleanup():
    """Periodic memory cleanup task for better performance"""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes

            logger.debug("Running periodic cleanup...")
            MemoryManager.cleanup_gpu_memory()

            # Clean old sessions (optional)
            session_manager = app_state.get_session_manager()
            session_manager.cleanup_old_sessions(days_old=7)

        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

# Create FastAPI app with improved configuration
app = FastAPI(
    title="Zhara AI Assistant",
    description="Modular AI Assistant with TTS, STT, and Session Management",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware with proper port configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        f"http://localhost:{PORT}",
        f"http://127.0.0.1:{PORT}"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

# Static file serving with proper caching
from config import STATIC_DIR as _STATIC_DIR
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

@app.get("/")
async def root():
    """Root endpoint serving the main interface"""
    return FileResponse(str(_STATIC_DIR / "index.html"))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check service health
        memory_stats = MemoryManager.get_memory_usage()
        tts_stats = get_tts_service().get_stats()
        session_stats = app_state.get_session_manager().get_session_stats() if app_state.get_session_manager() else {}

        return {
            "status": "healthy",
            "memory": memory_stats,
            "tts": tts_stats,
            "sessions": session_stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Zhara AI Assistant on {HOST}:{PORT}")

    uvicorn.run(
        "zhara:app",
        host=HOST,
        port=PORT,
        reload=False,  # Disable reload in production
        log_level="info",
        access_log=False
    )
