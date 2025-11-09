import threading
import sounddevice as sd
import numpy as np

from audio_input import AudioStream
from transcriber import Transcriber
from config import SAMPLE_RATE, CHUNK_DURATION, DEVICE_INDEX

def main():
    print("Starting transcription... Press Ctrl+C to stop.")
    audio_stream = AudioStream()
    transcriber = Transcriber(audio_stream)

    processing_thread = threading.Thread(target=transcriber.transcribe, daemon=True)
    processing_thread.start()

    with sd.InputStream(device=DEVICE_INDEX, samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
                        callback=audio_stream.audio_callback):
        while True:
            sd.sleep(int(CHUNK_DURATION * 1000))

if __name__ == "__main__":
    main()