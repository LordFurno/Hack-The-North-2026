from openai import OpenAI
import base64
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv(Path(__file__).resolve().with_name(".env"))

api_key = os.getenv("YIBU_API_KEY")

client = OpenAI(
    base_url="https://yibuapi.com/v1",
    api_key=api_key,
)

# Read and base64-encode the input audio
audio_path = "input.wav"

with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

completion = client.chat.completions.create(
    model="qwen3.5-omni-flash",

    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:;base64,{audio_b64}",
                        "format": "wav",
                    },
                },
                {
                    "type": "text",
                    "text": "Listen to this audio and respond to me."
                },
            ],
        }
    ],

    # Ask for both text and speech
    modalities=["text", "audio"],

    # Voice/audio format for the response
    audio={
        "voice": "Ethan",
        "format": "wav",
    },

    # Required for Omni audio output
    stream=True,
)

# Collect text and audio from the streaming response
text_output = ""
audio_b64_output = ""

for chunk in completion:
    if not chunk.choices:
        continue

    delta = chunk.choices[0].delta

    # Text response
    if delta.content:
        text_output += delta.content
        print(delta.content, end="", flush=True)

    # Audio response
    if hasattr(delta, "audio") and delta.audio:
        audio_b64_output += delta.audio.get("data", "")

print()

# Save the generated audio
if audio_b64_output:
    audio_bytes = base64.b64decode(audio_b64_output)

    # Qwen's WAV audio output is 24 kHz, 16-bit PCM
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    sf.write(
        "output.wav",
        audio_np,
        samplerate=24000
    )

print("Text:", text_output)
print("Audio saved to output.wav")