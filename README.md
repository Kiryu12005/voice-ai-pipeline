# Jiji AI — Local Voice Pipeline (STT → LLM → TTS)

A low-latency, fully local voice assistant pipeline: it listens to microphone input, transcribes speech, generates a response with a locally running LLM, and speaks the response back. Inspired by AI VTubers like Neuro-sama, built as a personal side-project to see whether a similar real-time pipeline was achievable from scratch.

## Features

- **Fully local, no cloud APIs.** Speech recognition, language model, and speech synthesis all run on-device.
- **Multi-threaded pipeline architecture.** Audio capture, transcription, LLM inference, and speech synthesis each run in their own thread, connected through queues (producer/consumer pattern), so no stage blocks another and latency stays low.
- **Local LLM comparison.** Both a Mistral 7B and a Llama model were tested locally; Mistral produced better responses for this use case and is the one actually used.

## Tech Stack

- **Python** (`threading` + `queue` for the pipeline)
- **faster-whisper** for speech-to-text
- **PyTorch** + **Transformers** for local LLM inference (Mistral 7B v0.3, quantized)
- **Coqui TTS** for text-to-speech

## Project Structure

```
config/
└── log_config.py       # logging setup

stt/
├── audio_input.py       # microphone capture (sounddevice input stream + callback)
├── config.py             # sample rate, chunk duration, device index
├── transcriber.py        # runs faster-whisper on incoming audio, pushes text to a queue
└── VAD.py                # placeholder for voice activity detection (not implemented, see Notes)

ttt/
├── config.py
├── filter.py              # placeholder for a word/content filter (not implemented, see Notes)
├── llm_core.py            # local LLM inference (Mistral 7B via torch/transformers)
└── models/                 # local model weights (not included in the repo, see Setup)

tts/
├── tts_core.py             # text-to-speech
└── models/

start.py                    # entry point, wires the pipeline together and starts all threads
```

## How It Works

1. `start.py` sets up logging and creates the queues that connect the pipeline stages.
2. `AudioStream` continuously captures microphone input via a callback (`sounddevice.InputStream`).
3. `Transcriber` runs **faster-whisper** on the incoming audio and pushes the recognized text onto a prompt queue.
4. `LLMCore` picks prompts off the queue and generates a response with a locally running **Mistral 7B** model.
5. `TTSCore` converts the generated text into speech.
6. Each stage runs in its own thread. Audio capture, transcription, generation, and speech synthesis therefore happen concurrently instead of waiting on each other, which keeps end-to-end latency low.

## Notes / Known Limitations

- **VAD.py** is a placeholder. The plan was to implement custom voice activity detection so transcription reacts to natural speech pauses; for now, transcription runs on fixed-length audio chunks (`CHUNK_DURATION`) instead.
- **filter.py** is a placeholder for a word/content filter, originally intended for a possible future use on stream (e.g. Twitch). Not implemented.
- The pipeline currently only supports local microphone input; there is no chat platform integration in this repo.

## Setup

1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place the local model weights in `ttt/models/` (a quantized **Mistral 7B v0.3** base model, loaded via PyTorch/Transformers). Model weights are not included in this repo due to their size.
4. Adjust `stt/config.py` (`SAMPLE_RATE`, `CHUNK_DURATION`, `DEVICE_INDEX`) to match your microphone.
5. Run:
   ```
   python start.py
   ```

## Background

Built after watching AI VTuber Neuro-sama on Twitch and wanting to find out whether a comparable real-time voice pipeline was buildable from scratch. The project stopped at a working prototype stage once the core pipeline (STT → LLM → TTS) ran reliably with low latency.