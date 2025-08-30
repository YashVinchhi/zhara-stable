import os
from pathlib import Path
from typing import Union
import validators

def get_env_int(key: str, default: int, min_value: int = None, max_value: int = None) -> int:
    """Get integer environment variable with validation"""
    try:
        value = int(os.getenv(key, default))
        if min_value is not None and value < min_value:
            print(f"Warning: {key} value {value} is below minimum {min_value}, using minimum")
            return min_value
        if max_value is not None and value > max_value:
            print(f"Warning: {key} value {value} is above maximum {max_value}, using maximum")
            return max_value
        return value
    except ValueError:
        print(f"Warning: Invalid {key} value, using default {default}")
        return default

def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable"""
    return str(os.getenv(key, str(default))).lower() == 'true'

def validate_url(url: str) -> str:
    """Validate URL and ensure it uses HTTPS in production"""
    if not validators.url(url):
        raise ValueError(f"Invalid URL: {url}")
    if not os.getenv('DEVELOPMENT') and not url.startswith('https://'):
        raise ValueError(f"URL must use HTTPS in production: {url}")
    return url

def validate_model_path(path: str) -> str:
    """Validate TTS model path exists"""
    if not Path(path).exists() and not path.startswith('tts_models/'):
        raise ValueError(f"Invalid TTS model path: {path}")
    return path

# --- Base Paths using pathlib ---
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
AUDIO_OUTPUT_DIR = STORAGE_DIR / "audio"
VISEME_OUTPUT_DIR = STORAGE_DIR / "visemes"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
STORAGE_DIR.mkdir(exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)
VISEME_OUTPUT_DIR.mkdir(exist_ok=True)

# --- Application Constants ---
MAX_AUDIO_DURATION = get_env_int('MAX_AUDIO_DURATION', 300, min_value=1, max_value=3600)
MAX_TEXT_LENGTH = get_env_int('MAX_TEXT_LENGTH', 1000, min_value=1, max_value=10000)
MAX_MESSAGE_LENGTH = get_env_int('MAX_MESSAGE_LENGTH', 2000, min_value=1, max_value=10000)
MAX_FILE_AGE = get_env_int('MAX_FILE_AGE', 24, min_value=1)
MAX_AUDIO_SIZE = get_env_int('MAX_AUDIO_SIZE', 10 * 1024 * 1024, min_value=1024)  # Minimum 1KB
LLM_TIMEOUT_SECONDS = get_env_int('LLM_TIMEOUT_SECONDS', 30, min_value=1)
RHUBARB_TIMEOUT_SECONDS = get_env_int('RHUBARB_TIMEOUT_SECONDS', 30, min_value=1)

# --- Server Configuration ---
HOST = os.getenv('HOST', '0.0.0.0')
PORT = get_env_int('PORT', 8000, min_value=1000, max_value=65535)
OFFLINE_MODE = get_env_bool('OFFLINE_MODE', False)

# --- Model Settings ---
WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'base')
VALID_WHISPER_SIZES = ['tiny', 'base', 'small', 'medium', 'large']
if WHISPER_MODEL_SIZE not in VALID_WHISPER_SIZES:
    print(f"Warning: Invalid WHISPER_MODEL_SIZE '{WHISPER_MODEL_SIZE}', using 'base'")
    WHISPER_MODEL_SIZE = 'base'

# Model paths using pathlib
WHISPER_MODEL_PATH = BASE_DIR / "models" / f"whisper-{WHISPER_MODEL_SIZE}"
TTS_MODEL_PATH = os.getenv('TTS_MODEL_PATH', 'tts_models/en/ljspeech/tacotron2-DDC')

# --- Performance Settings ---
MAX_CONCURRENT_REQUESTS = get_env_int('MAX_CONCURRENT_REQUESTS', 5, min_value=1, max_value=20)
MAX_CACHE_SIZE = get_env_int('MAX_CACHE_SIZE', 1000, min_value=10)
TTS_WORKER_COUNT = get_env_int('TTS_WORKER_COUNT', 2, min_value=1, max_value=4)

# --- API Configuration ---
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
API_TIMEOUT = get_env_int('API_TIMEOUT', 30, min_value=5, max_value=120)

# --- Rate Limiting ---
RATE_LIMIT_REQUESTS = get_env_int('RATE_LIMIT_REQUESTS', 60, min_value=1)
RATE_LIMIT_WINDOW = get_env_int('RATE_LIMIT_WINDOW', 60, min_value=1)

# --- Logging Configuration ---
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE = BASE_DIR / "logs" / "zhara.log"
LOG_FILE.parent.mkdir(exist_ok=True)

# --- Third-party TTS Providers ---
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '')
ELEVENLABS_BASE_URL = os.getenv('ELEVENLABS_BASE_URL', 'https://api.elevenlabs.io')
PIPER_EXE_PATH = os.getenv('PIPER_EXE_PATH', '')  # e.g., C:\tools\piper\piper.exe
PIPER_MODELS_DIR = os.getenv('PIPER_MODELS_DIR', '')  # dir containing *.onnx or *.onnx.gz
