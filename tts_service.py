"""
TTS Service with Async Processing and Queue Management
Handles text-to-speech generation with background workers and proper error handling
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from queue import Queue, Empty
import threading
import time
import logging
import requests
import subprocess

# Try optional imports; degrade gracefully if unavailable
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    from TTS.api import TTS as CoquiTTS  # type: ignore
    HAS_COQUI = True
except Exception:
    CoquiTTS = None  # type: ignore
    HAS_COQUI = False

from pydub import AudioSegment

from utils import MemoryManager, TextProcessor
from config import ELEVENLABS_API_KEY, ELEVENLABS_BASE_URL, PIPER_EXE_PATH, PIPER_MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass
class TTSJob:
    """TTS job structure for queue processing"""
    job_id: str
    text: str
    output_path: str
    model_name: str
    priority: int = 1  # Lower number = higher priority
    created_at: float = None
    callback: Optional[callable] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class TTSResult:
    """TTS processing result"""
    job_id: str
    success: bool
    audio_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0
    visemes: Optional[List[Dict]] = None


class TTSService:
    """
    Async TTS service with background worker and queue management
    Decouples TTS processing from API endpoints to prevent timeouts
    """

    def __init__(self, model_path: str, gpu_enabled: bool = True, max_workers: int = 2):
        self.model_path = model_path
        # Safely determine GPU availability
        cuda_available = False
        try:
            cuda_available = bool(torch and hasattr(torch, 'cuda') and torch.cuda.is_available())  # type: ignore[attr-defined]
        except Exception:
            cuda_available = False
        self.gpu_enabled = bool(gpu_enabled and cuda_available)
        self.max_workers = max_workers

        # Queue and worker management
        self.job_queue = Queue()
        self.result_store: Dict[str, TTSResult] = {}
        self.workers: List[threading.Thread] = []
        self.shutdown_event = threading.Event()

        # TTS model instances cache per worker and per model name
        self.models: Dict[int, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "jobs_processed": 0,
            "jobs_failed": 0,
            "average_processing_time": 0.0,
            "queue_size": 0
        }

        self._initialize_workers()

    def _initialize_workers(self):
        """Initialize background worker threads"""
        for worker_id in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                daemon=True,
                name=f"TTSWorker-{worker_id}"
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Initialized {self.max_workers} TTS workers")

    def _get_model(self, worker_id: int, model_name: Optional[str] = None):
        """Get or create TTS model for specific worker and model name (Coqui provider)."""
        if not HAS_COQUI:
            raise RuntimeError("Coqui TTS not available in this environment")

        if worker_id not in self.models:
            self.models[worker_id] = {}

        cache = self.models[worker_id]
        requested = (model_name or "default").strip() or "default"
        resolved_model = self.model_path if requested in ("default", "", None) else requested

        if resolved_model in cache:
            return cache[resolved_model]

        try:
            logger.info(f"Loading TTS model for worker {worker_id}: {resolved_model}")
            cache[resolved_model] = CoquiTTS(resolved_model, gpu=self.gpu_enabled)  # type: ignore
            logger.info(f"TTS model loaded successfully for worker {worker_id}")
            return cache[resolved_model]
        except Exception as model_error:
            logger.warning(f"Requested TTS model failed for worker {worker_id}: {model_error}")

        # Try fallback models
        for fallback_model in self._fallback_models():
            try:
                logger.info(f"Trying fallback TTS model: {fallback_model}")
                cache[fallback_model] = CoquiTTS(fallback_model, gpu=False)  # type: ignore
                logger.info(f"Fallback TTS model {fallback_model} loaded successfully for worker {worker_id}")
                return cache[fallback_model]
            except Exception as fallback_error:
                logger.warning(f"Fallback model {fallback_model} failed: {fallback_error}")
                continue

        raise RuntimeError(f"Could not load any TTS model for worker {worker_id}")

    def _check_model_availability(self, model_path: str) -> bool:
        """Check if a TTS model is available without downloading it"""
        try:
            from TTS.utils.manage import ModelManager
            manager = ModelManager()
            try:
                if hasattr(manager, 'list_models'):
                    available_models = manager.list_models()
                    return model_path in available_models
            except Exception:
                pass

            # Fallback: check cache directory
            import os
            tts_cache_dir = os.path.expanduser("~/.local/share/tts")
            if os.path.exists(tts_cache_dir):
                model_name = model_path.replace('/', '_').replace('tts_models_', '')
                for root, dirs, files in os.walk(tts_cache_dir):
                    if model_name in root.lower():
                        return True
            return False
        except Exception as e:
            logger.debug(f"Model availability check failed for {model_path}: {e}")
            return False

    def _get_available_fallback_models(self) -> List[str]:
        """Get list of fallback models that are actually available on the system"""
        logger.info("Skipping fallback model discovery to prevent unwanted downloads")
        return []

    def _fallback_models(self) -> List[str]:
        if not hasattr(self, '_cached_fallback_models'):
            self._cached_fallback_models = self._get_available_fallback_models()
        return self._cached_fallback_models

    def _worker_loop(self, worker_id: int):
        logger.info(f"TTS Worker {worker_id} started")

        while not self.shutdown_event.is_set():
            try:
                job = self.job_queue.get(timeout=1.0)
                if job is None:
                    break
                self._process_job(worker_id, job)
                self.job_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"TTS Worker {worker_id} stopped")

    def _split_for_tts(self, text: str, max_chars: int = 300) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for s in sentences:
            if not s:
                continue
            s = s.strip()
            if current_len + len(s) + (1 if current else 0) <= max_chars:
                current.append(s)
                current_len += len(s) + (1 if current else 0)
            else:
                if current:
                    chunks.append(' '.join(current))
                if len(s) > max_chars:
                    for i in range(0, len(s), max_chars):
                        chunks.append(s[i:i+max_chars])
                    current = []
                    current_len = 0
                else:
                    current = [s]
                    current_len = len(s)
        if current:
            chunks.append(' '.join(current))
        return chunks

    def _synthesize_chunks_and_merge(self, tts_model: Any, text: str, output_path: Path, timeout_per_chunk: int = 30):
        tmp_files: List[Path] = []
        try:
            chunks = self._split_for_tts(text, max_chars=300)
            logger.info(f"TTS chunking: {len(chunks)} chunks")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            for idx, chunk in enumerate(chunks):
                part_path = output_path.parent / f"{output_path.stem}_part{idx}.wav"
                tmp_files.append(part_path)

                def call_chunk():
                    tts_model.tts_to_file(text=chunk, file_path=str(part_path))
                    return True

                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(call_chunk)
                    fut.result(timeout=timeout_per_chunk)

                if not part_path.exists() or part_path.stat().st_size == 0:
                    raise RuntimeError(f"Chunk {idx} synthesis failed")

                MemoryManager.cleanup_gpu_memory()

            combined = None
            for fp in tmp_files:
                seg = AudioSegment.from_file(fp, format='wav')
                combined = seg if combined is None else combined + seg
            if combined is None:
                raise RuntimeError("No audio generated from chunks")
            combined.export(str(output_path), format='wav')
        finally:
            for fp in tmp_files:
                try:
                    if fp.exists():
                        fp.unlink()
                except Exception:
                    pass

    def _infer_gender(self, model_identifier: str) -> str:
        ident = (model_identifier or '').lower()
        if any(k in ident for k in ['female', 'femail', 'feminine', 'f_', '_f', '-f']):
            return 'female'
        if any(k in ident for k in ['male', 'masculine', 'm_', '_m', '-m']):
            return 'male'
        return 'unknown'

    def _display_name(self, base: str, gender: Optional[str] = None) -> str:
        label = base.split(':', 1)[-1]
        label = Path(label).stem if '/' in label or '\\' in label else label
        if gender and gender != 'unknown':
            return f"{label} ({gender})"
        return label

    def get_available_providers(self) -> List[Dict[str, str]]:
        providers: List[Dict[str, str]] = []
        if HAS_COQUI:
            providers.append({"id": "coqui", "label": "Coqui TTS (local)"})
        return providers

    def get_models_for_provider(self, provider: str) -> List[Dict[str, str]]:
        provider = (provider or 'coqui').lower()
        if provider == 'coqui':
            return self._list_coqui_models()
        return []

    def get_available_models(self, provider: Optional[str] = None) -> List[Dict[str, str]]:
        if provider:
            return self.get_models_for_provider(provider)
        models: List[Dict[str, str]] = []
        if HAS_COQUI:
            models.extend(self._list_coqui_models())
        return models

    def _list_coqui_models(self) -> List[Dict[str, str]]:
        if not HAS_COQUI:
            return []
        candidates = [self.model_path] + self._fallback_models()
        seen = set()
        models = []
        for m in candidates:
            if m and m not in seen:
                seen.add(m)
                gender = self._infer_gender(m)
                display_name = self._display_name(f"coqui:{m}", gender)
                models.append({
                    "provider": "coqui",
                    "name": f"coqui:{m}",
                    "gender": gender,
                    "display_name": display_name
                })
        return models

    def _list_elevenlabs_voices(self) -> List[Dict[str, str]]:
        voices: List[Dict[str, str]] = []
        if not ELEVENLABS_API_KEY:
            return voices
        try:
            url = f"{ELEVENLABS_BASE_URL.rstrip('/')}/v1/voices"
            headers = {"xi-api-key": ELEVENLABS_API_KEY}
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                logger.warning(f"ElevenLabs voices list failed: {r.status_code} {r.text}")
                return voices
            data = r.json() or {}
            for v in data.get('voices', []):
                vid = v.get('voice_id') or v.get('voiceID') or v.get('id')
                name = v.get('name') or 'Voice'
                labels = v.get('labels') or {}
                gender = (labels.get('gender') or '').lower() or 'unknown'
                display = self._display_name(f"elevenlabs:{name}", gender)
                voices.append({
                    "provider": "elevenlabs",
                    "name": f"elevenlabs:{vid}",
                    "gender": gender,
                    "display_name": display
                })
        except Exception as e:
            logger.error(f"Error listing ElevenLabs voices: {e}")
        return voices

    def _list_piper_models(self) -> List[Dict[str, str]]:
        voices: List[Dict[str, str]] = []
        try:
            models_dir = Path(PIPER_MODELS_DIR)
            if not (PIPER_EXE_PATH and models_dir.exists()):
                return voices
            for fp in models_dir.glob('**/*'):
                if fp.suffix.lower() in ('.onnx', '.onnx.gz', '.pth', '.pt', '.tar', '.tar.gz'):
                    gender = 'unknown'
                    display = self._display_name(f"piper:{fp.name}", gender)
                    voices.append({
                        "provider": "piper",
                        "name": f"piper:{str(fp.resolve())}",
                        "gender": gender,
                        "display_name": display
                    })
        except Exception as e:
            logger.error(f"Error listing Piper models: {e}")
        return voices

    def _synthesize_with_provider(self, provider: str, inner: str, text: str, output_path: Path):
        if provider == 'coqui' and not HAS_COQUI:
            fallback_tried_errors: List[str] = []
            try:
                default_piper = self._default_piper_model()
                if default_piper:
                    return self._synthesize_piper(default_piper, text, output_path)
            except Exception as e:
                fallback_tried_errors.append(f"piper: {e}")
            try:
                if ELEVENLABS_API_KEY:
                    raise RuntimeError("Coqui TTS not available. Provide tts_model='elevenlabs:<voice_id>' to use ElevenLabs.")
            except Exception as e:
                fallback_tried_errors.append(f"elevenlabs: {e}")
            raise RuntimeError("; ".join(fallback_tried_errors) or "No TTS provider available")

        if provider == 'coqui':
            self._synthesize_coqui(inner, text, output_path)
        elif provider == 'elevenlabs':
            self._synthesize_elevenlabs(inner, text, output_path)
        elif provider == 'piper':
            self._synthesize_piper(inner, text, output_path)
        else:
            raise RuntimeError(f"Unknown TTS provider: {provider}")

    def _synthesize_coqui(self, model_name_or_path: str, text: str, output_path: Path):
        if not HAS_COQUI:
            raise RuntimeError("Coqui TTS not available in this environment")
        tts_model = self._get_model(0, model_name_or_path)
        cleaned_text = TextProcessor.clean_text_for_tts(text)
        if len(cleaned_text) > 600:
            self._synthesize_chunks_and_merge(tts_model, cleaned_text, output_path)
        else:
            tts_model.tts_to_file(text=cleaned_text, file_path=str(output_path))

    def _synthesize_elevenlabs(self, voice_id: str, text: str, output_path: Path):
        if not ELEVENLABS_API_KEY:
            raise RuntimeError("ElevenLabs API key not configured")
        url = f"{ELEVENLABS_BASE_URL.rstrip('/')}/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "accept": "audio/mpeg",
            "content-type": "application/json"
        }
        payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs TTS failed: {r.status_code} {r.text[:200]}")
        tmp_mp3 = output_path.with_suffix('.mp3')
        with open(tmp_mp3, 'wb') as f:
            f.write(r.content)
        seg = AudioSegment.from_file(tmp_mp3, format='mp3')
        seg.export(str(output_path), format='wav')
        try:
            tmp_mp3.unlink()
        except Exception:
            pass

    def _synthesize_piper(self, model_path: str, text: str, output_path: Path):
        if not (PIPER_EXE_PATH and Path(PIPER_EXE_PATH).exists()):
            raise RuntimeError("Piper executable not configured or not found")
        if not Path(model_path).exists():
            raise RuntimeError(f"Piper model not found: {model_path}")
        cmd = [PIPER_EXE_PATH, "-m", model_path, "-f", str(output_path)]
        try:
            timeout = 60 if len(text) < 600 else 120
            completed = subprocess.run(cmd, input=text.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            if completed.returncode != 0:
                raise RuntimeError(f"Piper failed: {completed.stderr.decode('utf-8', errors='ignore')[:200]}")
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("Piper produced no audio")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Piper synthesis timed out")

    def _default_piper_model(self) -> Optional[str]:
        try:
            models_dir = Path(PIPER_MODELS_DIR)
            if not (PIPER_MODELS_DIR and models_dir.exists()):
                return None
            for fp in models_dir.glob('**/*'):
                if fp.suffix.lower() in ('.onnx', '.onnx.gz', '.pth', '.pt', '.tar', '.tar.gz'):
                    return str(fp.resolve())
        except Exception:
            return None
        return None

    def _parse_model_identifier(self, model_name: Optional[str]) -> Tuple[str, Optional[str]]:
        if model_name and model_name not in ("default", "coqui"):
            if ':' in model_name:
                provider, rest = model_name.split(':', 1)
                return provider.lower(), rest
            return 'coqui', model_name
        if HAS_COQUI:
            return 'coqui', self.model_path
        default_piper = self._default_piper_model()
        if default_piper:
            return 'piper', default_piper
        return 'coqui', self.model_path

    def _process_job(self, worker_id: int, job: TTSJob):
        start_time = time.time()
        result: Optional[TTSResult] = None
        try:
            MemoryManager.cleanup_gpu_memory()
            cleaned_text = TextProcessor.clean_text_for_tts(job.text)
            if not cleaned_text:
                raise ValueError("Empty text after cleaning")
            output_path = Path(job.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            provider, inner = self._parse_model_identifier(job.model_name)

            def do_synth():
                self._synthesize_with_provider(provider, inner, cleaned_text, output_path)
                return True

            with ThreadPoolExecutor(max_workers=1) as executor:
                fut = executor.submit(do_synth)
                text_length = len(cleaned_text)
                if text_length > 1500:
                    timeout = 150
                elif text_length > 1000:
                    timeout = 120
                elif text_length > 500:
                    timeout = 90
                else:
                    timeout = 60
                fut.result(timeout=timeout)

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("TTS file generation failed - empty or missing file")

            processing_time = time.time() - start_time
            visemes = self._generate_visemes(cleaned_text)
            result = TTSResult(job_id=job.job_id, success=True, audio_path=str(output_path), processing_time=processing_time, visemes=visemes)
            self.stats["jobs_processed"] += 1
            self.stats["average_processing_time"] = (
                (self.stats["average_processing_time"] * (self.stats["jobs_processed"] - 1) + processing_time)
                / self.stats["jobs_processed"]
            )
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"TTS job {job.job_id} failed: {error_msg}")
            result = TTSResult(job_id=job.job_id, success=False, error_message=error_msg, processing_time=processing_time)
            self.stats["jobs_failed"] += 1
            try:
                if Path(job.output_path).exists():
                    Path(job.output_path).unlink()
            except Exception:
                pass
        finally:
            if result is not None:
                self.result_store[job.job_id] = result
            if callable(job.callback) and result is not None:
                try:
                    job.callback(result)
                except Exception as e:
                    logger.error(f"Error in TTS job callback: {e}")
            MemoryManager.cleanup_gpu_memory()

    def _generate_visemes(self, text: str) -> List[Dict]:
        return []

    async def submit_job(self, text: str, output_path: str, model_name: str = "default", priority: int = 1) -> str:
        job_id = str(uuid.uuid4())
        job = TTSJob(job_id=job_id, text=text, output_path=output_path, model_name=model_name, priority=priority)
        self.job_queue.put(job)
        self.stats["queue_size"] = self.job_queue.qsize()
        logger.info(f"Submitted TTS job {job_id}, queue size: {self.stats['queue_size']}")
        return job_id

    async def get_result(self, job_id: str, timeout: float = 60) -> Optional[TTSResult]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if job_id in self.result_store:
                return self.result_store.pop(job_id)
            await asyncio.sleep(0.1)
        logger.warning(f"Timeout waiting for TTS job {job_id}")
        return None

    async def process_text_to_speech(self, text: str, output_path: str, timeout: float = 60, model_name: Optional[str] = None) -> TTSResult:
        job_id = await self.submit_job(text, output_path, model_name=model_name or "default")
        result = await self.get_result(job_id, timeout)
        if result is None:
            return TTSResult(job_id=job_id, success=False, error_message="TTS processing timed out")
        return result

    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats["queue_size"] = self.job_queue.qsize()
        stats["active_workers"] = len([w for w in self.workers if w.is_alive()])
        return stats

    def shutdown(self):
        logger.info("Shutting down TTS service...")
        try:
            self.shutdown_event.set()
        except Exception:
            logger.debug("Error setting shutdown event", exc_info=True)
        try:
            for _ in range(self.max_workers):
                self.job_queue.put(None)
        except Exception:
            logger.debug("Error waking up worker threads", exc_info=True)
        try:
            for worker in list(self.workers):
                try:
                    worker.join(timeout=5)
                except Exception:
                    logger.debug(f"Error joining worker {worker}", exc_info=True)
        except Exception:
            logger.debug("Error while waiting for workers to finish", exc_info=True)
        try:
            for worker_cache in list(self.models.values()):
                try:
                    for name, inst in list(worker_cache.items()):
                        try:
                            if hasattr(inst, 'unload'):
                                try:
                                    inst.unload()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    worker_cache.clear()
                except Exception:
                    logger.debug("Error clearing a worker model cache", exc_info=True)
            self.models.clear()
        except Exception:
            logger.debug("Error clearing models dict", exc_info=True)
        try:
            if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    logger.debug("Failed to empty torch CUDA cache", exc_info=True)
        except Exception:
            logger.debug("Error while checking/clearing torch cache", exc_info=True)
        try:
            MemoryManager.cleanup_gpu_memory()
        except Exception:
            logger.debug("Error during final MemoryManager cleanup", exc_info=True)
        logger.info("TTS service shutdown complete")


# Global TTS service instance
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Return the globally-initialized TTSService instance.

    Raises RuntimeError if the service has not been initialized.
    """
    global _tts_service
    if _tts_service is None:
        raise RuntimeError("TTS service not initialized")
    return _tts_service


def init_tts_service(model_path: str, gpu_enabled: bool = True, max_workers: int = 2) -> TTSService:
    """Initialize (or re-initialize) the global TTSService instance.

    If a service already exists it will be shut down first.
    """
    global _tts_service
    if _tts_service is not None:
        try:
            _tts_service.shutdown()
        except Exception:
            logger.debug("Error shutting down existing TTS service", exc_info=True)
        _tts_service = None

    _tts_service = TTSService(model_path, gpu_enabled, max_workers)
    return _tts_service


# Backwards-compatible alias: some modules import initialize_tts_service
def initialize_tts_service(model_path: str, gpu_enabled: bool = True, max_workers: int = 2) -> TTSService:
    """Compatibility wrapper for older import name initialize_tts_service."""
    return init_tts_service(model_path=model_path, gpu_enabled=gpu_enabled, max_workers=max_workers)

