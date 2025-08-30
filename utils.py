"""
Utility functions and helpers for Zhara AI Assistant
Centralized utilities for file management, text processing, and system detection
"""

import gc
import time
import subprocess
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from contextlib import contextmanager
import torch
import psutil
import logging

# Setup structured logging
logger = logging.getLogger(__name__)

class SystemInfo:
    """System information and environment detection utilities"""

    @staticmethod
    def detect_sbc_environment() -> tuple[bool, str]:
        """Detect if running on SBC (Single Board Computer) and adjust settings accordingly"""
        is_sbc = False
        sbc_type = "unknown"

        # Check for Jetson devices
        if Path('/proc/device-tree/model').exists():
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().strip()
                    if 'jetson' in model.lower() or 'nvidia' in model.lower():
                        is_sbc = True
                        sbc_type = "jetson"
            except Exception:
                pass

        # Check for Raspberry Pi
        if Path('/proc/cpuinfo').exists():
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'raspberry pi' in cpuinfo or 'bcm' in cpuinfo:
                        is_sbc = True
                        sbc_type = "raspberry_pi"
            except Exception:
                pass

        # Check memory constraints (less than 8GB indicates likely SBC)
        try:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            if total_memory_gb < 8:
                is_sbc = True
                if sbc_type == "unknown":
                    sbc_type = "low_memory"
        except Exception:
            pass

        return is_sbc, sbc_type

    @staticmethod
    def detect_gpu_info() -> Dict[str, Any]:
        """Enhanced GPU detection that works universally with all NVIDIA GPUs"""
        gpu_info = {
            "available": False,
            "name": "Unknown",
            "memory_gb": 0,
            "compute_capability": None,
            "driver_version": None
        }

        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device)

                gpu_info.update({
                    "available": True,
                    "name": properties.name,
                    "memory_gb": properties.total_memory / (1024**3),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "multi_processor_count": properties.multi_processor_count
                })

                # Try to get driver version
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    gpu_info["driver_version"] = driver_version.decode() if isinstance(driver_version, bytes) else driver_version
                except ImportError:
                    # Fallback method without pynvml
                    try:
                        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'],
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            gpu_info["driver_version"] = result.stdout.strip()
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"Could not get driver version: {e}")

                logger.info(f"GPU Detected: {gpu_info['name']}")
                logger.info(f"   Memory: {gpu_info['memory_gb']:.1f} GB")
                logger.info(f"   Compute Capability: {gpu_info['compute_capability']}")
                if gpu_info["driver_version"]:
                    logger.info(f"   Driver Version: {gpu_info['driver_version']}")

            except Exception as e:
                logger.error(f"Error getting GPU info: {e}")
                gpu_info["available"] = torch.cuda.is_available()
        else:
            logger.info("No CUDA-compatible GPU detected")

        return gpu_info

class FileManager:
    """Efficient file management utilities"""

    @staticmethod
    @contextmanager
    def managed_temp_files(*file_paths: str):
        """Context manager for efficient temporary file cleanup"""
        paths = [Path(p) for p in file_paths if p]
        try:
            yield paths
        finally:
            for file_path in paths:
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except OSError as e:
                        logger.warning(f"Could not remove temporary file {file_path}: {e}")

    @staticmethod
    def cleanup_files_efficiently(file_paths: List[str]) -> None:
        """Efficient file cleanup function that processes multiple files in one pass"""
        if not file_paths:
            return

        for file_path in file_paths:
            if file_path and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                except OSError as e:
                    logger.warning(f"Could not remove file {file_path}: {e}")

    @staticmethod
    def ensure_directory_exists(directory: str) -> Path:
        """Ensure directory exists and return Path object"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

class MemoryManager:
    """Memory and GPU management utilities"""

    @staticmethod
    def cleanup_gpu_memory():
        """Force GPU memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory_info = {}

        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_used_gb'] = system_memory.used / (1024**3)
        memory_info['system_total_gb'] = system_memory.total / (1024**3)
        memory_info['system_percent'] = system_memory.percent

        # GPU memory
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                memory_info['gpu_allocated_gb'] = gpu_memory.get('allocated_bytes.all.current', 0) / (1024**3)
                memory_info['gpu_reserved_gb'] = gpu_memory.get('reserved_bytes.all.current', 0) / (1024**3)
            except Exception:
                memory_info['gpu_allocated_gb'] = 0
                memory_info['gpu_reserved_gb'] = 0

        return memory_info

class TextProcessor:
    """Text processing utilities"""

    @staticmethod
    def clean_text_for_tts(text: str) -> str:
        """Clean text for TTS processing - comprehensive markdown and special character removal"""
        if not text:
            return ""

        import re

        # Remove code blocks first (including language specifiers)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]*`', '', text)

        # Remove markdown headers - enhanced to catch all variations
        text = re.sub(r'^#{1,6}\s*.*$', '', text, flags=re.MULTILINE)  # Remove entire header lines
        text = re.sub(r'#{1,6}\s*', '', text)  # Remove remaining hash symbols

        # Remove markdown bold/italic/strikethrough
        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)  # Bold+italic
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)      # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)          # Italic
        text = re.sub(r'___(.*?)___', r'\1', text)        # Bold+italic underscore
        text = re.sub(r'__(.*?)__', r'\1', text)          # Bold underscore
        text = re.sub(r'_(.*?)_', r'\1', text)            # Italic underscore
        text = re.sub(r'~~(.*?)~~', r'\1', text)          # Strikethrough

        # Remove markdown links but keep the text
        text = re.sub(r'\[([^\]]*)\]\(([^)]*)\)', r'\1', text)
        text = re.sub(r'<([^>]*)>', r'\1', text)

        # Remove markdown lists - enhanced
        text = re.sub(r'^\s*[-*+•]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

        # Remove blockquotes
        text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)

        # Remove horizontal rules
        text = re.sub(r'^[-*_=]{3,}$', '', text, flags=re.MULTILINE)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove special markdown characters that might remain
        text = re.sub(r'[#*_~`\[\](){}\\|]', '', text)

        # Remove step numbers at beginning of sentences (like "Step 1:", "Step 2:")
        text = re.sub(r'\bStep\s+\d+:\s*', '', text, flags=re.IGNORECASE)

        # Clean up problematic characters for TTS
        text = re.sub(r'[^\w\s.,!?;:()\-"\'\n]', ' ', text)

        # Fix multiple spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)

        # Remove leading/trailing whitespace from each line
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

        # Split very long texts into smaller chunks for TTS
        if len(text) > 1500:  # If text is very long
            sentences = re.split(r'[.!?]+', text)
            # Take first few sentences to keep under reasonable length
            selected_sentences = []
            total_length = 0
            for sentence in sentences:
                if total_length + len(sentence) > 1500:
                    break
                selected_sentences.append(sentence.strip())
                total_length += len(sentence)

            text = '. '.join(selected_sentences)
            if text and not text.endswith('.'):
                text += '.'

        # Final cleanup
        text = text.strip()

        # If text is too short after cleaning, provide a fallback
        if len(text.strip()) < 3:
            text = "I have a response for you."

        return text

    @staticmethod
    def generate_cache_key(text: str, model: str) -> str:
        """Generate cache key for responses"""
        content = f"{text}:{model}"
        return hashlib.md5(content.encode()).hexdigest()

class RateLimiter:
    """Simple rate limiting utility"""

    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier"""
        current_time = time.time()

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.time_window
        ]

        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(current_time)
            return True

        return False
