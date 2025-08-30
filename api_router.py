"""
API Router for Zhara AI Assistant
Centralized endpoint definitions with proper separation of concerns
"""

import asyncio
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
import platform  # Add
import tempfile
import json

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    whisper = None
    HAS_WHISPER = False

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
import soundfile as sf
import numpy as np
import aiohttp

from session_manager import SessionManager
from tts_service import get_tts_service
from utils import TextProcessor, RateLimiter
import config
import logging
from chroma_memory import ChromaMemory

logger = logging.getLogger(__name__)

# Import centralized app state for consistent dependency injection
from app_state import app_state

# Initialize router
router = APIRouter()

# Initialize rate limiter
rate_limiter = RateLimiter(max_requests=60, time_window=60)

# Request/Response Models
class ChatRequest(BaseModel):
    text: str
    model: str
    session_id: Optional[str] = None
    tts_model: Optional[str] = None  # newly added

class ChatResponse(BaseModel):
    reply: str
    audio_file: str
    viseme_file: str
    session_id: str
    model_used: str
    # New URL fields for frontend compatibility
    audio_url: Optional[str] = None
    viseme_url: Optional[str] = None
    # Session change notification for frontend
    session_changed: Optional[bool] = None
    original_session_id: Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: str
    last_updated: str
    message_count: int

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None

class UpdateSessionRequest(BaseModel):
    title: str

class ModelInfo(BaseModel):
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"

class TTSModelInfo(BaseModel):
    name: str
    gender: Optional[str] = None
    display_name: Optional[str] = None

class TTSProviderInfo(BaseModel):
    id: str
    label: str

# Dependency Functions
async def get_rate_limit():
    """Rate limiting dependency"""
    # In a real implementation, you'd extract client IP or user ID
    client_id = "default"  # Placeholder

    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    return True

async def get_session_manager() -> SessionManager:
    """Get session manager instance from centralized app state"""
    return app_state.get_session_manager()

SKIP_CHROMA = platform.system() == "Windows"  # Skip semantic memory on Windows

async def get_chroma_memory() -> Optional[ChromaMemory]:
    """Get ChromaMemory singleton instance"""
    if SKIP_CHROMA:
        return None
    if not hasattr(get_chroma_memory, '_instance'):
        # Persist in project storage directory
        persist_dir = str((Path(config.STORAGE_DIR) / 'chroma_db').resolve()) if hasattr(config, 'STORAGE_DIR') else './chroma_db'
        get_chroma_memory._instance = ChromaMemory(persist_directory=persist_dir)
    return get_chroma_memory._instance

# Audio Processing Utilities
def decode_audio_file(file_content: bytes) -> tuple[np.ndarray, int]:
    """Decode audio file content to numpy array"""
    try:
        import io
        audio_data, sample_rate = sf.read(io.BytesIO(file_content))

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy.signal import resample
            audio_data = resample(audio_data, int(len(audio_data) * 16000 / sample_rate))

        return audio_data, 16000
    except Exception as e:
        logger.error(f"Error decoding audio file: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio file format")

# AI Response Generation
async def generate_ai_response(text: str, model: str, session_id: Optional[str] = None) -> str:
    """Generate AI response using the specified model with session-aware context"""
    try:
        session_mgr = await get_session_manager()
        memory = None
        if not SKIP_CHROMA:
            try:
                memory = await get_chroma_memory()
            except Exception as mem_err:
                logger.warning(f"Chroma memory unavailable, continuing without semantic memory: {mem_err}")
                memory = None

        # Recent conversation context (last 8 exchanges)
        recent_msgs = []
        try:
            if session_id:
                msgs = session_mgr.get_session_messages(session_id, limit=16)
                # Format as simple plain text turns (avoid markdown)
                for m in msgs:
                    recent_msgs.append(f"User: {m.user_message}")
                    recent_msgs.append(f"Assistant: {m.ai_response}")
        except Exception:
            recent_msgs = []

        # Semantic memories relevant to current input
        similar = {"documents": []}
        try:
            if memory:
                similar = memory.query_memory(text, n_results=5, session_id=session_id, max_age_hours=24*7)
        except Exception:
            pass
        similar_docs = []
        try:
            docs = similar.get('documents') or []
            if docs and isinstance(docs[0], list):
                similar_docs = docs[0]
            elif isinstance(docs, list):
                similar_docs = docs
        except Exception:
            similar_docs = []

        # Build context block (keep modest size)
        ctx_parts = []
        if recent_msgs:
            ctx_parts.append("Recent conversation:\n" + "\n".join(recent_msgs[-12:]))
        if similar_docs:
            ctx_parts.append("Relevant memory:\n" + "\n".join(similar_docs[:5]))
        context_block = ("\n\n".join(ctx_parts)).strip()
        if len(context_block) > 1500:
            context_block = context_block[:1500]

        # Skip offline model handling
        if model == "offline" or config.OFFLINE_MODE:
            logger.info("Using offline mode for response generation")
            return get_offline_response(text)

        # Plain-text instruction to minimize markdown/special characters in output
        system_plain_text_instruction = (
            "You are a helpful assistant. Respond in plain English text only. "
            "Do not use markdown, code blocks, or any special characters beyond standard punctuation. "
            "Avoid headings, lists, emojis, hashes (#), asterisks (*), underscores (_), backticks (`), tildes (~), "
            "brackets [], braces {}, angle brackets <>, pipes |, or inline HTML. "
            "Write complete sentences as normal prose."
        )

        # Compose final prompt with instruction + context
        final_prompt = (
            f"{system_plain_text_instruction}\n\n" +
            (f"Context:\n{context_block}\n\n" if context_block else "") +
            f"User message:\n{text}\n\nAnswer:"
        ).strip()

        # Try to get response from Ollama
        logger.info(f"Generating response with model: {model}")
        timeout_total = getattr(config, 'LLM_TIMEOUT_SECONDS', 45) or 45
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_total)) as session:
            try:
                payload = {
                    "model": model,
                    # Include instruction both as system and in prompt for broader model compatibility
                    "system": system_plain_text_instruction,
                    "prompt": final_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }

                async with session.post(
                    f"{config.OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get('response', '').strip()

                        if ai_response:
                            logger.info(f"Successfully generated response using {model}")
                            return ai_response
                        else:
                            logger.warning("Empty response from Ollama")
                            return get_offline_response(text)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Ollama API error {response.status}: {error_text}")
                        return get_offline_response(text)

            except asyncio.TimeoutError:
                logger.warning(f"Timeout calling Ollama API with model {model}")
                return get_offline_response(text)
            except Exception as e:
                logger.warning(f"Ollama API failed with model {model}: {e}")
                return get_offline_response(text)

    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return get_offline_response(text)

def get_offline_response(text: str) -> str:
    """Generate offline response when API is unavailable"""
    responses = [
        "I understand you're asking about that. Let me think about it for a moment.",
        "That's an interesting question. Based on what I know, here's what I think.",
        "I hear what you're saying. Let me provide you with some thoughts on that.",
        "Thank you for sharing that with me. Here's my perspective on the matter."
    ]

    # Simple keyword-based responses
    text_lower = text.lower()

    if any(word in text_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! How can I help you today?"
    elif any(word in text_lower for word in ['thank', 'thanks']):
        return "You're welcome! Is there anything else I can help you with?"
    elif any(word in text_lower for word in ['bye', 'goodbye']):
        return "Goodbye! Have a great day!"
    else:
        import random
        return random.choice(responses)

# API Endpoints
@router.post("/ask", response_model=ChatResponse)
@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    rate_limit: None = Depends(get_rate_limit)
):
    """
    Main chat endpoint with async TTS processing
    Supports both /ask (legacy) and /api/chat endpoints
    """
    try:
        # Input validation
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty message")

        if len(request.text) > config.MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message too long (max {config.MAX_MESSAGE_LENGTH} characters)"
            )

        # Get or create session with proper change tracking
        session_id = request.session_id
        session_changed = False
        original_session_id = session_id

        if not session_id:
            session_id = session_manager.create_session()
            session_changed = True
            logger.info(f"Created new session: {session_id}")
        elif not session_manager.get_session(session_id):
            # Session doesn't exist, create new one and notify frontend
            old_session_id = session_id
            session_id = session_manager.create_session()
            session_changed = True
            logger.warning(f"Session {old_session_id} not found, created new session: {session_id}")

        # Generate response using AI model
        reply = await generate_ai_response(request.text, request.model, session_id)

        # Persist to memory (user + assistant turns) using effective session_id
        if not SKIP_CHROMA:
            try:
                memory = await get_chroma_memory()
                if session_id and memory:
                    memory.add_memory(request.text, metadata={"role": "user", "model": request.model}, session_id=session_id)
                    memory.add_memory(reply, metadata={"role": "assistant", "model": request.model}, session_id=session_id)
            except Exception as e:
                logger.warning(f"Could not persist to Chroma memory: {e}")

        # Clean text for TTS
        cleaned_reply = TextProcessor.clean_text_for_tts(reply)

        # Generate unique filenames
        response_id = str(uuid.uuid4())
        audio_filename = f"response_{response_id}.wav"
        viseme_filename = f"visemes_{response_id}.json"

        audio_path = Path(config.AUDIO_OUTPUT_DIR) / audio_filename
        viseme_path = Path(config.VISEME_OUTPUT_DIR) / viseme_filename

        # Submit TTS job to background service
        logger.info(f"Starting TTS processing for text: {cleaned_reply[:50]}...")

        # Compute wait timeout based on text length (buffer > worker timeout)
        text_len = len(cleaned_reply)
        if text_len > 1500:
            wait_timeout = 180  # must exceed worker 150s
        elif text_len > 1000:
            wait_timeout = 140  # exceed worker 120s
        elif text_len > 500:
            wait_timeout = 110  # exceed worker 90s
        else:
            wait_timeout = 60   # exceed worker 45s

        tts_result = None
        try:
            tts_service = get_tts_service()
            logger.info(f"TTS service stats before processing: {tts_service.get_stats()}")

            tts_result = await tts_service.process_text_to_speech(
                cleaned_reply,
                str(audio_path),
                timeout=wait_timeout,
                model_name=(request.tts_model or "default")
            )

            if tts_result and tts_result.success:
                logger.info(f"TTS successful: {audio_filename}, processing time: {tts_result.processing_time:.2f}s")
                # Verify the file actually exists and has content
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    logger.info(f"Audio file verified: {audio_path} ({audio_path.stat().st_size} bytes)")
                else:
                    logger.warning(f"Audio file missing or empty after TTS success: {audio_path}")
                    audio_filename = await create_placeholder_audio(audio_path, cleaned_reply)
            else:
                error_msg = tts_result.error_message if tts_result else "TTS service returned None"
                logger.error(f"TTS failed: {error_msg}")
                audio_filename = await create_placeholder_audio(audio_path, cleaned_reply)

        except Exception as tts_error:
            logger.error(f"TTS service error: {tts_error}", exc_info=True)
            audio_filename = await create_placeholder_audio(audio_path, cleaned_reply)

        # Generate visemes (if TTS was successful)
        visemes = tts_result.visemes if tts_result and tts_result.success else []
        await save_visemes(visemes, viseme_path)

        # Save message to session
        session_manager.add_message(
            session_id=session_id,
            user_message=request.text,
            ai_response=reply,
            model_used=request.model,
            audio_file=audio_filename,
            viseme_file=viseme_filename
        )

        # Build URLs for frontend playback
        audio_url = f"/audio/{audio_filename}"
        viseme_url = f"/visemes/{viseme_filename}"

        return ChatResponse(
            reply=reply,
            audio_file=audio_filename,
            viseme_file=viseme_filename,
            session_id=session_id,
            model_used=request.model,
            audio_url=audio_url,
            viseme_url=viseme_url,
            session_changed=session_changed,  # Notify session change
            original_session_id=original_session_id  # Pass original session ID
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/api/tts")
@router.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    rate_limit: None = Depends(get_rate_limit)
):
    """Standalone TTS endpoint with comprehensive file validation"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text")

        # Generate unique filename
        response_id = str(uuid.uuid4())
        audio_filename = f"tts_{response_id}.wav"
        audio_path = Path(config.AUDIO_OUTPUT_DIR) / audio_filename

        # Process with TTS service
        tts_service = get_tts_service()
        result = await tts_service.process_text_to_speech(
            request.text,
            str(audio_path)
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message)

        # Validate the generated audio file
        if not await validate_audio_file(audio_path):
            logger.error(f"Generated audio file failed validation: {audio_path}")
            # Create placeholder audio as fallback
            audio_filename = await create_placeholder_audio(audio_path, request.text)

        # Build URL for frontend playback
        audio_url = f"/audio/{audio_filename}"

        return {"audio_file": audio_filename, "audio_url": audio_url, "processing_time": result.processing_time}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail="TTS processing failed")

@router.post("/api/stt")
@router.post("/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    rate_limit: None = Depends(get_rate_limit)
):
    """Speech-to-text endpoint with proper async handling"""
    try:
        # Validate file
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty audio file")

        if len(file_content) > config.MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum of {config.MAX_AUDIO_SIZE / 1024 / 1024:.2f}MB"
            )

        # Decode audio
        audio_data, sample_rate = decode_audio_file(file_content)

        # Check duration
        duration = len(audio_data) / sample_rate
        if duration > config.MAX_AUDIO_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Audio duration ({duration:.1f}s) exceeds maximum of {config.MAX_AUDIO_DURATION}s"
            )

        # Process with Whisper (assuming whisper_model is available globally)
        # This would need to be adapted to your specific Whisper model setup
        text = await process_speech_to_text(audio_data)

        return {"text": text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Speech recognition failed")

# Session Management Endpoints
@router.post("/api/sessions", response_model=SessionInfo)
async def create_session(
    request: CreateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Create a new chat session"""
    session_id = session_manager.create_session(request.title)
    session = session_manager.get_session(session_id)

    return SessionInfo(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at,
        last_updated=session.last_updated,
        message_count=session.message_count
    )

@router.get("/api/sessions", response_model=List[SessionInfo])
async def list_sessions(
    limit: Optional[int] = 50,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """List all chat sessions"""
    sessions = session_manager.list_sessions(limit)

    return [
        SessionInfo(
            session_id=session.session_id,
            title=session.title,
            created_at=session.created_at,
            last_updated=session.last_updated,
            message_count=session.message_count
        )
        for session in sessions
    ]

@router.get("/api/sessions/{session_id}")
async def get_session_details(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get detailed session information including messages"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = session_manager.get_session_messages(session_id)

    return {
        "session": SessionInfo(
            session_id=session.session_id,
            title=session.title,
            created_at=session.created_at,
            last_updated=session.last_updated,
            message_count=session.message_count
        ),
        "messages": messages
    }

@router.put("/api/sessions/{session_id}")
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Update session title"""
    success = session_manager.update_session(session_id, title=request.title)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session updated successfully"}

@router.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Delete a session"""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session deleted successfully"}

@router.post("/api/sessions/{session_id}/suggest_title")
async def suggest_session_title(session_id: str, session_manager: SessionManager = Depends(get_session_manager)):
    """Suggest a concise session title based on recent conversation."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # Collect last few messages
    messages = session_manager.get_session_messages(session_id, limit=6)
    parts = []
    for m in messages or []:
        try:
            if getattr(m, 'user_message', None):
                parts.append(f"User: {m.user_message}")
            if getattr(m, 'ai_response', None):
                parts.append(f"Assistant: {m.ai_response}")
        except Exception:
            # Fallback if message is a dict
            if isinstance(m, dict):
                if m.get('user_message'):
                    parts.append(f"User: {m.get('user_message')}")
                if m.get('ai_response'):
                    parts.append(f"Assistant: {m.get('ai_response')}")
    context = "\n".join(parts)
    if not context:
        return {"title": session.title or "New chat"}
    try:
        system = (
            "You generate short, descriptive chat titles. "
            "Return 3-6 words, no punctuation, no quotes."
        )
        prompt = (
            f"{system}\n\nConversation so far:\n{context}\n\nTitle:"
        )
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            payload = {
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "max_tokens": 20}
            }
            async with s.post(f"{config.OLLAMA_BASE_URL}/api/generate", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    title = (data.get('response') or '').strip().strip('"').strip()
                    # Basic cleanup
                    if len(title) > 60:
                        title = title[:60]
                    if not title:
                        title = session.title or "New chat"
                else:
                    title = session.title or "New chat"
    except Exception:
        title = session.title or "New chat"
    return {"title": title}

# Model and Service Information Endpoints
@router.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of available Ollama models installed locally (via OLLAMA_BASE_URL)."""
    try:
        base = getattr(config, 'OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(f"{base}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models: List[ModelInfo] = []
                    for model in data.get("models", []):
                        name = model.get("name", "")
                        size = model.get("size")
                        # Normalize size to string
                        size_str = str(size) if size is not None else None
                        models.append(ModelInfo(
                            name=name,
                            size=size_str,
                            modified_at=model.get("modified_at")
                        ))
                    return models
                else:
                    logger.warning(f"Ollama /api/tags returned {response.status}")
                    return []
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return []

@router.get("/tts/providers", response_model=List[TTSProviderInfo])
async def get_tts_providers():
    """Get list of available TTS providers on this device."""
    try:
        svc = get_tts_service()
        providers = svc.get_available_providers() or []
        # Map to schema
        return [TTSProviderInfo(id=p.get('id'), label=p.get('label')) for p in providers if p.get('id') and p.get('label')]
    except Exception as e:
        logger.error(f"Error fetching TTS providers: {e}")
        return []

@router.get("/tts/voices", response_model=List[TTSModelInfo])
async def get_tts_voices(provider: Optional[str] = None):
    """Get list of available TTS voices/models for a specific provider on this device."""
    try:
        svc = get_tts_service()
        prov = provider
        if not prov:
            # default to first available provider
            ps = svc.get_available_providers()
            prov = ps[0]['id'] if ps else None
        if not prov:
            return []
        voices = svc.get_models_for_provider(prov) or []
        items: List[TTSModelInfo] = []
        for v in voices:
            items.append(TTSModelInfo(
                name=v.get('name', ''),
                gender=v.get('gender'),
                display_name=v.get('display_name')
            ))
        return items
    except Exception as e:
        logger.error(f"Error fetching TTS voices: {e}")
        return []

@router.get("/avatars", response_model=List[Dict[str, str]])
async def get_avatar_models():
    """Get list of available 3D avatar models"""
    return [
        {"id": "default", "name": "Default Avatar", "description": "Standard 3D avatar"},
        {"id": "professional", "name": "Professional", "description": "Business-style avatar"},
        {"id": "casual", "name": "Casual", "description": "Relaxed, friendly avatar"},
        {"id": "futuristic", "name": "Futuristic", "description": "Sci-fi inspired avatar"},
        {"id": "minimalist", "name": "Minimalist", "description": "Clean, simple design"},
        {"id": "expressive", "name": "Expressive", "description": "Highly animated avatar"}
    ]

# File Serving Endpoints
@router.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve audio files with proper validation"""
    file_path = Path(config.AUDIO_OUTPUT_DIR) / filename

    # Enhanced file validation
    if not await validate_audio_file(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found or invalid")

    return FileResponse(
        file_path,
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=3600"}
    )

@router.get("/visemes/{filename}")
async def serve_visemes(filename: str):
    """Serve viseme files with proper validation"""
    file_path = Path(config.VISEME_OUTPUT_DIR) / filename

    # Enhanced file validation
    if not await validate_viseme_file(file_path):
        raise HTTPException(status_code=404, detail="Viseme file not found or invalid")

    return FileResponse(
        file_path,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=3600"}
    )

# WebSocket Endpoint
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Lightweight WS endpoint to avoid 403 spam from stray clients (e.g., HMR).
    Accepts connections and echoes a minimal ack, then keeps the socket open until the client disconnects.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Respond minimally to keep clients satisfied
            await websocket.send_text("ack")
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass

@router.head("/queue")
async def queue_head():
    """No-op endpoint to satisfy health/queue probes."""
    return Response(status_code=204)

@router.get("/queue")
async def queue_get():
    """Simple queue status endpoint."""
    return {"status": "ok"}

# Helper Functions - Proper implementations
async def fetch_available_models() -> List[Dict[str, Any]]:
    """Fetch available models from Ollama"""
    try:
        if config.OFFLINE_MODE:
            return [{"name": "offline", "size": "N/A"}]

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(f"{config.OLLAMA_BASE_URL}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = []

                    for model in data.get('models', []):
                        model_info = {
                            "name": model.get('name', 'unknown'),
                            "size": format_bytes(model.get('size', 0)),
                            "modified_at": model.get('modified_at', '')
                        }
                        models.append(model_info)

                    logger.info(f"Found {len(models)} Ollama models")
                    return models
                else:
                    logger.warning(f"Ollama API returned status {response.status}")
                    return [{"name": "offline", "size": "N/A"}]

    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return [{"name": "offline", "size": "N/A"}]

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable format"""
    if bytes_val == 0:
        return "0 B"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"

# Global Whisper model instance for reuse
_whisper_model = None

async def get_whisper_model():
    """Get or initialize Whisper model"""
    global _whisper_model
    if not HAS_WHISPER:
        raise RuntimeError("OpenAI Whisper is not installed. Please install it with: pip install openai-whisper")

    if _whisper_model is None:
        try:
            # Load the base Whisper model (you can change to 'small', 'medium', 'large' for better accuracy)
            _whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    return _whisper_model

async def process_speech_to_text(audio_data: np.ndarray) -> str:
    """Process audio with Whisper for speech-to-text conversion"""
    try:
        if not HAS_WHISPER:
            logger.error("Whisper not available - speech-to-text functionality disabled")
            return "Error: Speech-to-text functionality not available. Please install openai-whisper."

        # Get Whisper model
        model = await get_whisper_model()

        # Create temporary file for Whisper processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write audio data to temporary file
            sf.write(temp_path, audio_data, 16000)

            # Process with Whisper in a thread pool to avoid blocking
            import concurrent.futures
            loop = asyncio.get_event_loop()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    lambda: model.transcribe(temp_path, language="en")
                )

            # Extract transcribed text
            transcribed_text = result.get("text", "").strip()

            if transcribed_text:
                logger.info(f"Speech-to-text successful: {transcribed_text[:50]}...")
                return transcribed_text
            else:
                logger.warning("Whisper returned empty transcription")
                return "Sorry, I couldn't understand the audio."

        finally:
            # Clean up temporary file
            try:
                Path(temp_path).unlink()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Speech-to-text error: {e}")
        return "Error: Could not transcribe audio. Please try again."

async def save_visemes(visemes: List[Dict], output_path: Path):
    """Save visemes to file"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(visemes, f, indent=2)
        logger.debug(f"Saved visemes to {output_path}")
    except Exception as e:
        logger.error(f"Error saving visemes: {e}")

async def create_placeholder_audio(output_path: Path, text: str) -> str:
    """Create a simple placeholder audio file when TTS fails"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate a simple beep sound - 800Hz tone for 0.5 seconds
        sample_rate = 16000
        frequency = 800  # 800Hz beep
        duration = 0.5  # 0.5 seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Create a simple beep with fade in/out to avoid clicks
        waveform = 0.3 * np.sin(2 * np.pi * frequency * t)
        fade_samples = int(0.05 * sample_rate)  # 50ms fade
        waveform[:fade_samples] *= np.linspace(0, 1, fade_samples)
        waveform[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # Write to WAV file
        sf.write(output_path, waveform, sample_rate)

        logger.info(f"Created placeholder audio file: {output_path}")
        return str(output_path.name)

    except Exception as e:
        logger.error(f"Error creating placeholder audio: {e}")
        # Return the filename anyway so the frontend doesn't break
        return str(output_path.name)

async def validate_audio_file(file_path: Path, min_size_bytes: int = 100) -> bool:
    """Validate that audio file exists, has content, and is a valid audio file"""
    try:
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Audio file does not exist: {file_path}")
            return False

        # Check file size
        file_size = file_path.stat().st_size
        if file_size < min_size_bytes:
            logger.warning(f"Audio file too small ({file_size} bytes): {file_path}")
            return False

        # Try to read the audio file to validate format
        try:
            audio_data, sample_rate = sf.read(str(file_path))
            if len(audio_data) == 0:
                logger.warning(f"Audio file contains no audio data: {file_path}")
                return False
            logger.debug(f"Audio file validated: {file_path} ({file_size} bytes, {len(audio_data)} samples)")
            return True
        except Exception as audio_error:
            logger.warning(f"Audio file format validation failed: {file_path} - {audio_error}")
            return False

    except Exception as e:
        logger.error(f"Error validating audio file {file_path}: {e}")
        return False

async def validate_viseme_file(file_path: Path) -> bool:
    """Validate that viseme file exists and contains valid JSON"""
    try:
        # Check if file exists
        if not file_path.exists():
            logger.debug(f"Viseme file does not exist: {file_path}")
            return False

        # Check file size
        file_size = file_path.stat().st_size
        if file_size < 2:  # At least "{}"
            logger.warning(f"Viseme file too small ({file_size} bytes): {file_path}")
            return False

        # Try to read and parse the JSON file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.debug(f"Viseme file validated: {file_path} ({file_size} bytes)")
            return True
        except json.JSONDecodeError as json_error:
            logger.warning(f"Viseme file JSON validation failed: {file_path} - {json_error}")
            return False

    except Exception as e:
        logger.error(f"Error validating viseme file {file_path}: {e}")
        return False
