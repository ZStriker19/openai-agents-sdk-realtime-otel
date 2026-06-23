# OpenAI Agent SDK Realtime Demo

A voice-driven multi-agent demo built on the OpenAI Agents SDK Realtime API. A triage agent listens to your mic and routes requests to one of four specialist agents: Weather, Calculator, Python Code Runner, or File Writer.

## Requirements

- Python **3.13+**
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager)
- A microphone and speakers
- An OpenAI API key with Realtime API access

### Platform notes for PyAudio

`pyaudio` needs PortAudio installed on your system:

- **macOS**: `brew install portaudio`
- **Ubuntu/Debian**: `sudo apt-get install portaudio19-dev python3-pyaudio`
- **Windows**: usually works out of the box via the prebuilt wheel — no extra steps

## Setup

```bash
# 1. Install uv if you don't have it
#    macOS/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
#    Windows:      powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install dependencies (creates a .venv automatically)
uv sync

# 3. Configure your API key
cp .env.example .env
# then edit .env and paste your OPENAI_API_KEY
```

## Find your audio device indices

```bash
uv run mic_detect.py
```

This prints something like:

```
Index 0: Microphone (Realtek Audio)
Index 1: Speakers (Realtek Audio)
Index 2: Headset Microphone
...
```

Pick the index for the mic you want to use and the index for the speaker you want to use.

## Run

```bash
# Uses default device indices (input=0, output=1)
uv run openai_agent_sdk_realtime.py

# Or pick specific devices
uv run openai_agent_sdk_realtime.py --input-device 2 --output-device 3
```

You should see:

```
--- Multi-Agent Session Active (Speak into mic) ---
Agents: Weather | Calculator | Python Code | File Writer
Triage agent will route your requests.
```

Talk into the mic. Press `Ctrl+C` to exit.

## Try it

- *"What's the weather in Tokyo?"* → routed to **WeatherAgent**
- *"What is 47 times 39?"* → routed to **CalculatorAgent**
- *"Write a Python script that prints the first ten Fibonacci numbers and run it."* → **PythonCodeAgent**
- *"Save a file at notes.txt that says hello world."* → **FileWriterAgent**

## Datadog Tracing (OTel)

The instrumented version of this file (`openai_agent_sdk_realtime.py`) sends GenAI traces to Datadog using OpenTelemetry. Each model turn becomes its own trace, and all turns share a `gen_ai.conversation.id` so Datadog groups them into a single conversation view.

### Configure

Two things to fill in before running:

**1. Datadog API key** — add to `.env`:

```
DD_API_KEY=your-datadog-api-key
```

**2. Service name** — update the placeholder near the top of `openai_agent_sdk_realtime.py`:

```python
provider = TracerProvider(resource=Resource({SERVICE_NAME: "your-service-name"}))
```

Replace `"your-service-name"` with whatever you want to appear as the service in Datadog (e.g. `"my-voice-agent"`). This is how your traces are grouped in the LLM Observability UI.

The instrumentation reads `DD_API_KEY` from the environment and exports to `https://otlp.datadoghq.com/v1/traces` (US1). If you're on a different Datadog site (EU, US3, etc.), update the endpoint in the OTel setup block at the top of the file — see [Datadog OTLP endpoints](https://docs.datadoghq.com/opentelemetry/setup/otlp_ingest_in_the_agent/?tab=host).

### How it works

Three things are instrumented in `handle_events()`:

**1. One span per model turn** — `agent_start` opens a root OTel span; `agent_end` closes it. Each span gets `gen_ai.conversation.id` set to the same value for the session, which Datadog uses to group turns into a conversation.

**2. Input/output transcripts** — The Agents SDK doesn't surface transcripts as top-level session events. They arrive inside `raw_model_event`:
- `data.type == "transcript_delta"` — model output, accumulated across deltas and written to the span at `agent_end`
- `data.type == "input_audio_transcription_completed"` — user input, buffered and written as a single `gen_ai.input.messages` span event at `agent_end`

> **Important:** All user utterances for a turn must be batched into **one** `add_event` call. Calling `add_event` with `gen_ai.input.messages` multiple times causes Datadog to string-concatenate the values, producing invalid JSON that falls back to raw text in the UI.

**3. Token usage** — `data.type == "raw_server_event"` with `response.done` carries the token counts. It arrives before `agent_end`, so the values are available when the span closes.

**4. Audio blobs** — Both sides of the conversation are captured as playable audio in Datadog:

- **User input**: mic audio is buffered throughout each turn and attached as a WAV blob on the input message at `agent_end`
- **Model output**: audio chunks from `event.type == "audio"` are accumulated and attached as a WAV blob on the output message at `agent_end`

The raw PCM16 mono audio from the Realtime API is wrapped in a WAV container before encoding so the Datadog UI audio player can decode it. After the traces arrive, each turn's input and output messages will have a **Load Audio** button — click it to play back the audio for that turn.

> **Note on turn-boundary audio capture**: mic audio is snapshotted at `agent_end` rather than `agent_start`. This avoids a race condition where chunks from the user's just-finished utterance could still be in flight when `agent_start` fires.

### Note on `event.type == "transcript_delta"`

The original event loop in this file checks `event.type == "transcript_delta"` — this branch **never fires** in the current Agents SDK. Transcripts are wrapped inside `raw_model_event`; check `event.data.type` instead.

## Files

| File | Purpose |
| --- | --- |
| `openai_agent_sdk_realtime.py` | Main entry point — defines agents, handoffs, audio loop, and OTel instrumentation |
| `tool_definitions.py` | The four tool implementations |
| `mic_detect.py` | Lists available PyAudio input/output devices |
| `pyproject.toml` | Dependencies (managed by `uv`) |
| `.env.example` | Template for your `OPENAI_API_KEY` and `DD_API_KEY` |

## Troubleshooting

- **`OSError: [Errno -9996] Invalid input device`** — wrong `--input-device` index. Re-run `uv run mic_detect.py` and pick a different one.
- **No audio playback** — same fix as above, but for `--output-device`.
- **`401 Unauthorized`** — your `OPENAI_API_KEY` is missing, invalid, or doesn't have Realtime API access. Confirm `.env` exists in this folder and contains a valid key.
- **PyAudio install fails on Linux** — install PortAudio first: `sudo apt-get install portaudio19-dev`.
