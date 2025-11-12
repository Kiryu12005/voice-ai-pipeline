import threading
import queue
import sounddevice as sd
import time
import sys

from stt.audio_input import AudioStream
from stt.transcriber import Transcriber
from ttt.llm_core import LLMCore
from tts.tts_core import TTSCore
from stt.config import SAMPLE_RATE, CHUNK_DURATION, DEVICE_INDEX

def main():
    print("Starting Jiji AI... Press Ctrl+C to stop.")

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
    print("transcriber started...")
    llm_thread.start()
    print("llm_core started...")
    tts_thread.start()
    print("tts_core started...")

    print("\nAll processes started. Listening for audio input...")
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
            print("System ready. Start speaking!")

            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping Jiji AI...")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # clean up threads
        print("Programm exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()