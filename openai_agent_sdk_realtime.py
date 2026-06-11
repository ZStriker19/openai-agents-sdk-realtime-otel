import argparse
import asyncio
import json
import os
import queue
import threading
import uuid

from dotenv import load_dotenv
load_dotenv()

import pyaudio
import numpy as np
from agents import function_tool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.realtime import RealtimeAgent, RealtimeRunner, realtime_handoff

# --- OTel setup ---------------------------------------------------------
# pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "https://otlp.datadoghq.com/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = f"dd-api-key={os.getenv('DD_API_KEY')}"
os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"

provider = TracerProvider(resource=Resource({SERVICE_NAME: "your-service-name"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
# ------------------------------------------------------------------------

from tool_definitions import (
    calculate,
    get_weather,
    run_python_code,
    write_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent realtime example")
    parser.add_argument("--input-device", type=int, default=0)
    parser.add_argument("--output-device", type=int, default=1)
    return parser.parse_args()


weather_tool = function_tool(get_weather)
calculator_tool = function_tool(calculate)
python_code_tool = function_tool(run_python_code)
file_write_tool = function_tool(write_file)

weather_agent = RealtimeAgent(
    name="WeatherAgent",
    handoff_description="A helpful agent that can look up weather for any city.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a weather specialist. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the city the customer is asking about.
    2. Use the get_weather tool to look up the weather. Do not rely on your own knowledge.
    3. If the customer asks a question that is not related to weather, transfer back to the triage agent.""",
    tools=[weather_tool],
)

calculator_agent = RealtimeAgent(
    name="CalculatorAgent",
    handoff_description="A helpful agent that can evaluate math expressions and calculations.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a math specialist. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the math expression the customer wants evaluated.
    2. Use the calculate tool to evaluate the expression. Do not rely on your own knowledge.
    3. If the customer asks a question that is not related to math, transfer back to the triage agent.""",
    tools=[calculator_tool],
)

python_code_agent = RealtimeAgent(
    name="PythonCodeAgent",
    handoff_description="A helpful agent that can write and execute Python scripts.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a Python code execution specialist. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Understand what Python code the customer wants to run.
    2. Use the run_python_code tool to write and execute the script.
    3. Report the results back to the customer.
    4. If the customer asks a question that is not related to running Python code, transfer back to the triage agent.""",
    tools=[python_code_tool],
)

file_writer_agent = RealtimeAgent(
    name="FileWriterAgent",
    handoff_description="A helpful agent that can write content to files on disk.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a file writing specialist. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask the customer for the file path and content if not already provided.
    2. Use the write_file tool to write the content to the file.
    3. Confirm the file was written successfully.
    4. If the customer asks a question that is not related to writing files, transfer back to the triage agent.""",
    tools=[file_write_tool],
)

triage_agent = RealtimeAgent(
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        weather_agent,
        realtime_handoff(calculator_agent),
        realtime_handoff(python_code_agent),
        realtime_handoff(file_writer_agent),
    ],
)

weather_agent.handoffs.append(triage_agent)
calculator_agent.handoffs.append(triage_agent)
python_code_agent.handoffs.append(triage_agent)
file_writer_agent.handoffs.append(triage_agent)


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024
OUTPUT_CHUNK = 4096
MIC_METER_EVERY_N_CHUNKS = 5

REALTIME_MODEL = "gpt-4o-realtime-preview"


async def main(*, input_device_index: int = 0, output_device_index: int = 1):
    p = pyaudio.PyAudio()
    mic = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                 frames_per_buffer=CHUNK, input_device_index=input_device_index)
    speaker = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True,
                     frames_per_buffer=OUTPUT_CHUNK, output_device_index=output_device_index)

    playback_queue: queue.Queue[bytes | None] = queue.Queue()

    def playback_worker():
        while True:
            chunk = playback_queue.get()
            if chunk is None:
                return
            try:
                speaker.write(chunk)
            except Exception:
                return

    def flush_playback():
        try:
            while True:
                playback_queue.get_nowait()
        except queue.Empty:
            pass

    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

    runner = RealtimeRunner(triage_agent, config={"model_settings": {"voice": "shimmer"}})

    # --- Tracing state --------------------------------------------------
    # Each model turn becomes its own root trace. gen_ai.conversation.id
    # is set on every span so Datadog groups them into a single conversation.
    conversation_id = f"conv-{uuid.uuid4().hex[:12]}"

    current_turn_span = None
    pending_input_transcripts = []  # user utterances buffered until agent_end
    current_output_transcript = ""
    current_usage = {}
    # --------------------------------------------------------------------

    print("--- Multi-Agent Session Active (Speak into mic) ---")
    print("Agents: Weather | Calculator | Python Code | File Writer")
    print(f"Triage agent will route your requests. Conversation: {conversation_id}\n")

    async with await runner.run() as session:
        async def send_mic_audio():
            tick = 0
            try:
                while True:
                    raw_data = await asyncio.to_thread(mic.read, CHUNK, False)
                    tick += 1
                    if tick % MIC_METER_EVERY_N_CHUNKS == 0:
                        audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
                        rms = np.sqrt(np.mean(audio_data**2))
                        meter = int(min(rms / 50, 50))
                        print(f"Mic Level: {'█' * meter}{' ' * (50-meter)} |", end="\r")
                    await session.send_audio(raw_data)
            except Exception:
                pass

        async def handle_events():
            nonlocal current_turn_span, pending_input_transcripts
            nonlocal current_output_transcript, current_usage

            async for event in session:
                if event.type == "audio":
                    playback_queue.put_nowait(event.audio.data)

                elif event.type == "audio_interrupted":
                    flush_playback()
                    print("\n[interrupted]")

                # --- Turn start: open a new root span -------------------
                elif event.type == "agent_start":
                    current_output_transcript = ""
                    current_usage = {}
                    # Pass an empty context so this span has no parent —
                    # each turn is its own trace.
                    current_turn_span = tracer.start_span(
                        f"chat {REALTIME_MODEL}",
                        kind=trace.SpanKind.CLIENT,
                        context=otel_context.Context(),
                    )
                    current_turn_span.set_attribute("gen_ai.operation.name", "chat")
                    current_turn_span.set_attribute("gen_ai.provider.name", "openai")
                    current_turn_span.set_attribute("gen_ai.request.model", REALTIME_MODEL)
                    current_turn_span.set_attribute("gen_ai.response.model", REALTIME_MODEL)
                    current_turn_span.set_attribute("gen_ai.agent.name", event.agent.name)
                    # Links this trace to all other turns in the same session.
                    current_turn_span.set_attribute("gen_ai.conversation.id", conversation_id)

                # --- Turn end: attach I/O + tokens, close span ----------
                elif event.type == "agent_end":
                    if current_turn_span is None:
                        continue

                    # Input: all user utterances from this turn as one event.
                    # Important: do NOT call add_event for input multiple times —
                    # Datadog concatenates duplicate keys, producing invalid JSON.
                    if pending_input_transcripts:
                        current_turn_span.add_event(
                            "gen_ai.client.inference.operation.details",
                            {"gen_ai.input.messages": json.dumps([
                                {"role": "user", "parts": [{"type": "text", "content": t}]}
                                for t in pending_input_transcripts
                            ])},
                        )
                        pending_input_transcripts = []

                    # Output: full transcript assembled from deltas.
                    if current_output_transcript:
                        current_turn_span.add_event(
                            "gen_ai.client.inference.operation.details",
                            {"gen_ai.output.messages": json.dumps([{
                                "role": "assistant",
                                "parts": [{"type": "text", "content": current_output_transcript}],
                                "finish_reason": "stop",
                            }])},
                        )

                    # Token usage from response.done (captured below).
                    if current_usage:
                        if tok := current_usage.get("input_tokens"):
                            current_turn_span.set_attribute("gen_ai.usage.input_tokens", tok)
                        if tok := current_usage.get("output_tokens"):
                            current_turn_span.set_attribute("gen_ai.usage.output_tokens", tok)

                    current_turn_span.end()
                    current_turn_span = None

                elif event.type == "raw_model_event":
                    data = event.data

                    # NOTE: event.type == "transcript_delta" does NOT exist at
                    # the session level in the Agents SDK. Transcripts arrive
                    # wrapped in raw_model_event — check data.type instead.

                    if data.type == "transcript_delta":
                        # Accumulate the model's spoken response.
                        current_output_transcript += data.delta
                        print(data.delta, end="", flush=True)

                    elif data.type == "input_audio_transcription_completed":
                        # User's speech transcribed. Arrives asynchronously —
                        # often after agent_start — so always buffer here.
                        print(f"\n[You]: {data.transcript}")
                        pending_input_transcripts.append(data.transcript)

                    elif data.type == "raw_server_event":
                        # response.done carries token counts. It arrives before
                        # agent_end so the values are ready when we close the span.
                        raw = data.data
                        if isinstance(raw, dict) and raw.get("type") == "response.done":
                            current_usage = raw.get("response", {}).get("usage", {})

                elif event.type == "error":
                    print(f"\n[Error]: {event.error}")

        mic_task = asyncio.create_task(send_mic_audio())
        try:
            await handle_events()
        finally:
            mic_task.cancel()

    playback_queue.put(None)
    playback_thread.join(timeout=1.0)
    mic.close()
    speaker.close()
    p.terminate()
    provider.force_flush()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(input_device_index=args.input_device, output_device_index=args.output_device))
