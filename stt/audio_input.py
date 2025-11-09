import queue
import numpy as np
from config import BUFFER_SIZE, SAMPLE_RATE, OVERLAP_DURATION

class AudioStream:
    def __init__(self):
        self.audio_queue = queue.Queue(BUFFER_SIZE)
        self.context_buffer = np.zeros(int(SAMPLE_RATE * OVERLAP_DURATION), dtype=np.float32)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        
        combined_audio = np.concatenate((self.context_buffer, indata[:, 0]))

        self.context_buffer = indata[-int(SAMPLE_RATE * OVERLAP_DURATION):, 0]
        self.audio_queue.put(combined_audio.copy())