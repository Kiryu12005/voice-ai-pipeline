import time
import numpy as np
from collections import deque

from stt.config import SAMPLE_RATE, DECODE_HOP_S, WINDOW_S, EPS, initial_prompt
from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, audio_stream, prompt_queue, model_size="small.en", device="cuda", compute_type="float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.audio_stream = audio_stream
        self.prompt_queue = prompt_queue

        self._chunks = deque()
        self._last_decode_time = 0.0
        self._total_samples = 0
        self._last_printed_time = 0.0
        
        self.sentence_buffer = ""

    def append_chunk(self, chunk):
        self._chunks.append(chunk)
        self._total_samples += int(chunk.shape[0])

        total = np.concatenate(list(self._chunks)) if len(self._chunks) else np.array([], dtype=np.float32)
        max_samples = int(SAMPLE_RATE * WINDOW_S)

        if total.size > max_samples:
            total = total[-max_samples:]
            self._chunks.clear()
            self._chunks.append(total)
        else:
            pass
    
    def _get_window(self):
        if not self._chunks:
            return None
        total = np.concatenate(list(self._chunks))
        return total

    def transcribe(self):
        last_printed_text = ""

        while True:
            audio_chunk = self.audio_stream.audio_queue.get()
            self.append_chunk(audio_chunk)

            now = time.time()

            if now - self._last_decode_time < DECODE_HOP_S:
                continue
            self._last_decode_time = now

            window_audio = self._get_window()
            if window_audio is None or window_audio.size < int(SAMPLE_RATE * 0.25):
                continue

            win_start_sec = (self._total_samples - window_audio.size) / float(SAMPLE_RATE)

            segments, info = self.model.transcribe(
                window_audio,
                language="en",
                beam_size=3,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=1200),
                word_timestamps=True,
                # Halluzination suppression
                temperature=0.0,
                no_speech_threshold=0.5,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-0.25,
                condition_on_previous_text=False,
                #initial_prompt=initial_prompt
            )

            out = []
            for seg in segments:
                if not getattr(seg, "words", None):
                    continue
                for w in seg.words:
                    # Timestamps
                    ws = w.start if (w.start is not None) else 0.0
                    we = w.end if (w.end is not None) else ws
                    abs_start = win_start_sec + ws
                    abs_end = win_start_sec + we

                    if abs_end > self._last_printed_time + EPS:
                        out.append(w.word)
                        self._last_printed_time = abs_end

            if out:
                current_text = " ".join(out).strip()

                if current_text and current_text != last_printed_text:
                    if (not last_printed_text or
                        current_text not in last_printed_text or
                        len(current_text) > len(last_printed_text) * 0.7):

                        print(current_text, end=' ', flush=True)
                        last_printed_text = current_text

                        self.sentence_buffer += current_text + " "

                        if any(p in current_text for p in [".", "!", "?"]):
                            prompt_to_send = self.sentence_buffer.strip()
                            self.sentence_buffer = ""

                            print(f"\n[STT -> LLM]: {prompt_to_send}")     # Debug print
                            self.prompt_queue.put(prompt_to_send)