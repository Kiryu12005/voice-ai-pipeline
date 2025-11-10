import queue
import numpy as np
from config import BUFFER_SIZE

class AudioStream:
    def __init__(self):
        self.audio_queue = queue.Queue(BUFFER_SIZE)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        
        arr = indata.astype(np.float32, copy=False)
        mono = arr.mean(axis=1) if arr.ndim == 2 else arr

        try:
            self.audio_queue.put_nowait(mono.copy())
        except queue.Full:
            try:
                _ = self.audio_queue.get_nowait() # Drop oldest
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(mono.copy())
            except queue.Full:
                pass