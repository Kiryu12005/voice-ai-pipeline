import queue
import time
import torch
import threading
import numpy as np
import sounddevice as sd
import logging

from TTS.api import TTS

logger = logging.getLogger(__name__)

class TTSCore:
    def __init__(self, response_queue: queue.Queue):
        self.response_queue = response_queue

        logger.info("TTS-Core initialized.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Loading Coqui TTS on {device}...")

        self.tts = TTS("tts_models/en/ljspeech/vits").to(device)
        logger.info("TTS Model loaded!")

        self.word_buffer = []
        self.audio_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def _playback_loop(self):
        with sd.OutputStream(samplerate=22050, channels=1, dtype="float32") as stream:
            while True:
                audio_data = self.audio_queue.get()

                if audio_data is None:
                    break

                audio_data = np.asarray(audio_data, dtype=np.float32)
                stream.write(audio_data)

    def process_responses(self):
        logger.info("TTS-Core processing responses...")

        last_flush_time = time.time()

        while True:
            item = self.response_queue.get()

            if item is None:
                self.audio_queue.put(None)
                break

            text, is_final = item

            now = time.time()

            if text:
                self.word_buffer.append(text)

            joined = " ".join(self.word_buffer).strip()

            sentence_finished = any(joined.endswith(p) for p in [".", "...", "?", "!"])
            buffer_long_enough = len(joined) > 80
            waited_long_enough = (now - last_flush_time) > 0.7

            should_speak = (is_final or sentence_finished or (buffer_long_enough and waited_long_enough))

            if should_speak and joined:
                sentence = joined
                self.word_buffer.clear()
                last_flush_time = now
                
                logger.info(f"[JIJI SAYS]: {sentence}")

                wav = self.tts.tts(text=sentence)
                audio = np.array(wav, dtype=np.float32)

                self.audio_queue.put(audio)

            self.response_queue.task_done()