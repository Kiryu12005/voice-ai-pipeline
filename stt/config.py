SAMPLE_RATE = 16000
CHUNK_DURATION = 0.25
BUFFER_SIZE = 32
DEVICE_INDEX = 2
DECODE_HOP_S = 0.25
WINDOW_S = 1.5
EPS = 0.15

hot_words = [
    "Jiji (spelled J i j i)",
    "GG", "okay", "lets go", "Twitch", "POG", "Poggers", "VTuber"
]

initial_prompt = (
    "The streamers Name is Jiji(spelled J i j i)."
    "Casual gaming chat style."
    "Common words and names: " + ", ".join(hot_words) + "."
)