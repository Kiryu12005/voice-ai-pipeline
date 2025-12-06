import queue
import time
import torch
import numpy as np
import sounddevice as sd
from TTS.api import TTS

class TTSCore:
    def __init__(self, response_queue: queue.Queue):
        self.response_queue = response_queue

        print("TTS-Core initialized.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Coqui TTS on {device}...")

        self.tts = TTS("tts_models/en/ljspeech/vits").to(device)
        print("TTS Model loaded!")

    def process_responses(self):
        print("TTS-Core processing responses...")

        while True:
            llm_response = self.response_queue.get()

            if llm_response is None:
                break

            print(f"[JIJI SAYS]: {llm_response}")

            wav = self.tts.tts(text=llm_response)

            audio_data = np.array(wav, dtype=np.float32)

            sd.play(audio_data, samplerate=22050)
            sd.wait()

            self.response_queue.task_done()
            time.sleep(0)