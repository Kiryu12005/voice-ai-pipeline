from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, audio_stream, model_size="small", device="cuda", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.audio_stream = audio_stream
        self.last_text = ""

    def transcribe(self):
        while True:
            audio_chunk = self.audio_stream.audio_queue.get()
            segments, info = self.model.transcribe(audio_chunk, language="en", beam_size=3, vad_filter=True)

            for segment in segments:
                print(segment.text)