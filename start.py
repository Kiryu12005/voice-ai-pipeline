import threading
import queue
import logging
import sounddevice as sd
import time
import sys

from stt.audio_input import AudioStream
from stt.transcriber import Transcriber
from ttt.llm_core import LLMCore
from tts.tts_core import TTSCore
from stt.config import SAMPLE_RATE, CHUNK_DURATION, DEVICE_INDEX
from config.log_config import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Jiji AI... Press Ctrl+C to stop.")

    prompt_queue = queue.Queue()
    response_queue = queue.Queue()

    audio_stream = AudioStream()

    transcriber = Transcriber(audio_stream, prompt_queue)
    llm_core = LLMCore(prompt_queue, response_queue)
    tts_core = TTSCore(response_queue)

    transcriber_thread = threading.Thread(target=transcriber.transcribe, daemon=True)
    llm_thread = threading.Thread(target=llm_core.process_loop, daemon=True)
    tts_thread = threading.Thread(target=tts_core.process_responses, daemon=True)

    transcriber_thread.start()
    logger.info("transcriber started...")
    llm_thread.start()
    logger.info("llm_core started...")
    tts_thread.start()
    logger.info("tts_core started...")

    logger.info("\nAll processes started. Listening for audio input...")
    try:
        with sd.InputStream(
            device=DEVICE_INDEX,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            latency="low",
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
            callback=audio_stream.audio_callback
        ):
            logger.info("System ready. Start speaking!")

            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\nStopping Jiji AI...")
    
    except Exception as e:
        logger.debug(f"An error occurred: {e}")
    
    finally:
        # clean up threads
        logger.info("Programm exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()