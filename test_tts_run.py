import asyncio, uuid, sys
from pathlib import Path
from config import AUDIO_OUTPUT_DIR, TTS_MODEL_PATH
from tts_service import init_tts_service, get_tts_service

out = Path(AUDIO_OUTPUT_DIR) / f"test_synth_{uuid.uuid4().hex[:8]}.wav"
print('OUTPUT_PATH', out)
# Initialize service (no GPU)
svc = init_tts_service(TTS_MODEL_PATH, gpu_enabled=False, max_workers=1)

async def main():
    res = await svc.process_text_to_speech("Hello from automated test.", str(out), timeout=60, model_name='default')
    print('RESULT_SUCCESS', res.success)
    print('RESULT_ERROR', res.error_message)
    print('AUDIO_EXISTS', out.exists())
    if out.exists():
        print('SIZE', out.stat().st_size)

try:
    asyncio.run(main())
finally:
    try:
        get_tts_service().shutdown()
    except Exception:
        pass

print('DONE')

