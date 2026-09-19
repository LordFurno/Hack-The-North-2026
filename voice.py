from dataclasses import dataclass
from pathlib import Path
import numpy as np
import omni
import argparse, base64, json, math, os, queue, re, threading, time
import urllib.error, urllib.parse, urllib.request, wave

#Omni's second job: the voice loop. Wake phrase -> the world model's whole belief ->
#one spoken answer. Phrasing is the model's job and grounding is the world model's,
#which is the only reason this can hand Omni a microphone and not get invented
#locations back: it never asks Omni where anything is, it TELLS it, and asks for a
#sentence. The camera frame goes along for referent resolution ("what about this one")
#and for nothing else.
#
#This process never touches the world. It reads /state and /events like any other
#client, so a crash here costs a voice and not a belief.

SERVICE_URL = "http://127.0.0.1:8000"
VOICE_DIR = "voice" #Question, reply and frame for the turn just taken. Overwritten each time

WAKE_PHRASE = "test test" #Say this, then the question. Matches whisperTest.py

RATE = 16000 #What whisper wants, so the mic is opened there and never resampled
CHUNK = 1024
SILENCE_LIMIT = 1.2 #Seconds of quiet that end an utterance
MAX_UTTERANCE_S = 15.0 #A stuck-open mic must not buffer forever
WHISPER_MODEL = "base.en"

#What counts as quiet is a property of the room, not a number worth hard-coding. A mic
#whose noise floor sits above a fixed threshold never goes quiet, so no utterance ever
#ends on silence and every question runs to MAX_UTTERANCE_S before it is sent -- fifteen
#seconds of room tone with the question buried in it. Measured at startup instead.
NOISE_CHUNKS = 16 #~1 s of room, listened to before listening for a voice
#Set from a recording made on this rig: a floor around 1000-1500 with speech reaching only
#2200-3000, so the usable band is narrow. Higher and the threshold climbs over the voice
#and the desk goes deaf, which is the worse failure of the two; lower and the room's own
#wobble keeps the utterance open until the clock cuts it.
NOISE_MARGIN = 1.8
THRESHOLD_MIN = 300.0 #A silent room must not leave the gate hair-trigger
THRESHOLD_MAX = 4000.0 #Somebody talking over the measurement must not deafen it

BASE_URL = os.environ.get("YIBU_BASE_URL", "https://yibuapi.com/v1")
MODEL = os.environ.get("YIBU_MODEL", "qwen3.5-omni-flash")
PURPOSE = "spatial_memory_voice" #Separable from the labelling spend in the ledger
VOICE = "Ethan"
REPLY_RATE = 24000 #Omni's wav output is 24 kHz 16-bit mono PCM
MAX_TOKENS = 256
TIMEOUT_S = 120.0

NEAR_LIMIT = 3 #Neighbours listed per object, nearest first
EVENT_LIMIT = 8 #Recent events handed over, oldest first

SYSTEM = (
    "You are the voice of a desk that remembers where things are. A world model watches "
    "the desk through an overhead camera and hands you its entire belief as a WORLD STATE "
    "block. Every claim you make about where something is MUST come from that block. "
    "Never invent a location, never infer one from the camera image, and never smooth "
    "over a low confidence score. If what was asked about is not in the block, say you "
    "do not know. The image, when there is one, is only for working out WHICH object is "
    "being asked about.\n"
    "Answer out loud, in one short sentence, the way a person would. Never read out ids, "
    "millimetres, centimetres or JSON. Let the confidence you were given pick the wording:\n"
    "  above 0.9  state it plainly -- \"it's under the box\"\n"
    "  0.7 to 0.9 state it with the caveat -- \"it should be under the box; I haven't seen "
    "it in a while, but the box hasn't moved\"\n"
    "  0.4 to 0.7 hedge -- \"probably under the box, though I'm not certain\"\n"
    "  below 0.4  admit it -- \"I've lost track of it\"\n"
    "If the world state says a belief was falsified, say so before anything else.")


@dataclass
class Heard: #One finished utterance: what was said, and the audio it was said in
    text: str
    audio: np.ndarray | None = None #float32 mono at RATE, exactly what whisper read


# ---- the wake phrase ----------------------------------------------------

def clean(text: str) -> str:
    #Whisper punctuates however it likes, and "Test, test." has to match "test test".
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()


def wakeSplit(text: str, phrase: str = WAKE_PHRASE) -> str | None:
    #What is left after the wake phrase is the question. "" means the phrase stood alone,
    #so the question is in the NEXT utterance; None means this was not addressed to us.
    i = text.find(clean(phrase))
    return None if i < 0 else text[i + len(clean(phrase)):].strip(" ,.?!")


# ---- the world, as context ----------------------------------------------

def getJson(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def fetchState(url: str) -> dict:
    return getJson(f"{url.rstrip('/')}/state")


def fetchEvents(url: str, limit: int = EVENT_LIMIT) -> list[dict]:
    return (getJson(f"{url.rstrip('/')}/events").get("events") or [])[-limit:]


def fetchFrame(url: str, path: str) -> str:
    #The frame a person is looking at while they ask. Optional in every sense: no camera
    #running is a 404, and a 404 costs referent resolution and nothing else.
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/preview.jpg", timeout=5.0) as r:
            jpeg = r.read()
    except (urllib.error.URLError, OSError):
        return ""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(jpeg)
    return path


def cm(mm: float) -> float:
    return round(mm / 10.0, 1)


def nearest(obj: dict, objs: list[dict], limit: int = NEAR_LIMIT) -> list[dict]:
    #"Next to the keyboard" is the answer people actually want, and the world model stores
    #millimetres, so the neighbours are computed here rather than left to Omni's arithmetic.
    out = [{"object": o["label"], "id": o["id"],
            "distance_cm": cm(math.dist(obj["pos"], o["pos"]))}
           for o in objs if o["id"] != obj["id"]]
    out.sort(key=lambda o: o["distance_cm"])
    return out[:limit]


def objectJson(obj: dict, objs: list[dict]) -> dict:
    #`where` is service.phraseLocation: the parent chain already walked and spoken. It is
    #the line Omni should be repeating, and everything else here is for the questions it
    #does not answer -- which one, how sure, how long ago, what is beside it.
    return {"id": obj["id"], "object": obj["label"], "where": obj["location"],
            "status": obj["status"], "confidence": obj["confidence"],
            "seconds_since_confirmed": obj["seconds_since_confirmed"],
            "size_cm": [cm(v) for v in obj["footprint"]],
            "position_cm": [cm(v) for v in obj["pos"]],
            "is_container": obj["is_container"],
            "nearest": nearest(obj, objs)}


def causeText(cause: str, labels: dict[str, str]) -> str:
    #A cause names an id -- "covered_by:4f2a91c8" -- and an id is not a thing to say out
    #loud. Omni is told not to read them; not handing it one is better than telling it.
    kind, _, who = cause.partition(":")
    return f"{kind}:{labels.get(who, who)}" if who else cause


def eventJson(ev: dict, now: float, labels: dict[str, str] = {}) -> dict:
    return {"seconds_ago": round(now - ev["ts"]), "object": ev.get("label") or ev["entity"],
            "kind": ev["kind"], "cause": causeText(ev["cause"], labels), "note": ev["note"]}


def contextJson(state: dict, events: list[dict] = ()) -> dict:
    #Everything the model is allowed to say, in one block. The agent is dropped: "in your
    #hand" is already in the phrase of whatever it holds, and a hand is not a thing to find.
    now = state.get("ts") or time.time()
    objs = [e for e in state.get("entities", []) if not e.get("is_agent")]
    labels = {e["id"]: e["label"] for e in state.get("entities", [])}
    x0, y0, x1, y1 = state.get("mat") or (0, 0, 0, 0)
    return {"desk_size_cm": [cm(x1 - x0), cm(y1 - y0)],
            "position_origin": "top-left corner of the desk, seen from above",
            "object_count": len(objs),
            "objects": [objectJson(o, objs) for o in objs],
            "recent_events": [eventJson(ev, now, labels) for ev in events]}


def promptFor(ctx: dict, question: str) -> str:
    asked = question.strip() or "(the question is in the audio)"
    return ("WORLD STATE -- the only thing you know about this desk:\n"
            + json.dumps(ctx, indent=1)
            + f"\n\nThe person at the desk just asked, out loud: \"{asked}\"\n"
            + "Answer them in one spoken sentence, grounded in the block above.")


# ---- omni ---------------------------------------------------------------

def yibuAudit():
    #Streaming audio out means a hand-rolled request -- their chat_completion is one shot
    #and text only -- so the token record has to be appended by hand. Same ledger, same
    #shape, because a call nobody can account for afterwards is the thing to avoid.
    omni.yibuHttp() #Puts the examples dir on sys.path, which is all this needs
    import yibu_audit
    return yibu_audit


def decodeAudio(parts: list[str]) -> bytes:
    #Concatenate first, decode once: a single chunk is not required to be a multiple of 4.
    b64 = "".join(parts)
    return base64.b64decode(b64 + "=" * (-len(b64) % 4)) if b64 else b""


def collect(lines) -> tuple[str, bytes, dict]:
    #Server-sent events: text arrives as delta.content, speech as base64 in delta.audio,
    #and usage rides the final chunk, which is the only reason the ledger gets token counts.
    text, transcript, audio, usage = "", "", [], {}
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body or body == "[DONE]":
            continue
        try:
            chunk = json.loads(body)
        except ValueError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text += delta.get("content") or ""
            got = delta.get("audio") or {}
            audio.append(got.get("data") or "")
            transcript += got.get("transcript") or ""
    return (text or transcript).strip(), decodeAudio(audio), usage


def ask(prompt: str, key: str, questionWav: str = "", framePath: str = "",
        voice: str = VOICE, model: str = MODEL, baseUrl: str = BASE_URL,
        system: str = SYSTEM) -> tuple[str, bytes]:
    #Returns (what it said, raw 24 kHz PCM). The audio of the question goes up as well as
    #its transcript: whisper's guess at "Rubik's cube" is not what should decide the answer.
    import httpx #Late, and only here: the tests parse chunks without a socket in sight
    http, audit = omni.yibuHttp(), yibuAudit()
    endpoint = f"{baseUrl.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": http.build_omni_messages(
            prompt, system=system,
            image=Path(framePath) if framePath else None,
            audio=Path(questionWav) if questionWav else None),
        "modalities": ["text", "audio"], #Both, so the dashboard reader sees what was said
        "audio": {"voice": voice, "format": "wav"},
        "stream": True, #Required for audio out
        "stream_options": {"include_usage": True},
        "max_tokens": MAX_TOKENS,
        "temperature": 0.2,
    }

    started, status = time.monotonic(), None
    try:
        with httpx.Client(timeout=TIMEOUT_S, trust_env=False) as client:
            with client.stream("POST", endpoint, json=payload,
                               headers={"Authorization": f"Bearer {key}",
                                        "Content-Type": "application/json"}) as r:
                status = r.status_code
                if status >= 400:
                    r.read() #Or raise_for_status has nothing to say about why
                    r.raise_for_status()
                text, audio, usage = collect(r.iter_lines())
    except Exception as err:
        audit.append_audit_record(model=model, api_key=key, endpoint=endpoint,
                                  purpose=PURPOSE, transport="http-stream", ok=False,
                                  status_code=status, latency_s=time.monotonic() - started,
                                  error=f"{type(err).__name__}: {err}",
                                  audit_log=omni.auditLog())
        raise

    audit.append_audit_record(model=model, api_key=key, endpoint=endpoint,
                              purpose=PURPOSE, transport="http-stream", ok=True,
                              status_code=status, latency_s=time.monotonic() - started,
                              response_json={"usage": usage}, audit_log=omni.auditLog())
    return text, audio


# ---- audio in and out ---------------------------------------------------

def pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def writeWav(path: str, pcm: bytes, rate: int) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return path


def play(pcm: bytes, rate: int = REPLY_RATE):
    #Raw PCM straight at the speakers. No player, no temp file to wait on.
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True)
    try:
        stream.write(pcm)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def micFrames(frames: queue.Queue, stop: threading.Event):
    #One thread doing nothing but reading the mic, because whisper takes about a second
    #on a long utterance and a mic that is not being read drops whatever was said then.
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    try:
        while not stop.is_set():
            frames.put(stream.read(CHUNK, exception_on_overflow=False))
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def whisper(name: str = WHISPER_MODEL):
    from faster_whisper import WhisperModel #A model load, so never at import time
    print(f"loading whisper {name}...")
    model = WhisperModel(name, device="cpu", compute_type="float32")

    def transcribe(audio: np.ndarray) -> str:
        segments, _info = model.transcribe(audio, beam_size=3)
        return clean(" ".join(s.text for s in segments))
    return transcribe


def chunkOf(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def loudness(chunk: np.ndarray) -> float:
    #RMS back on the int16 scale, which is what every threshold here is written in.
    return float(np.sqrt(np.mean(chunk ** 2))) * 32768.0 if len(chunk) else 0.0


def noiseFloor(frames: queue.Queue, chunks: int = NOISE_CHUNKS) -> float:
    #Listen to the room before listening for a voice. The median rather than the minimum:
    #one unusually quiet buffer is not what the next second is going to sound like.
    levels = []
    while len(levels) < chunks:
        try:
            levels.append(loudness(chunkOf(frames.get(timeout=2.0))))
        except queue.Empty:
            break
    return float(np.median(levels)) if levels else 0.0


def thresholdFor(floor: float) -> float:
    return min(THRESHOLD_MAX, max(THRESHOLD_MIN, floor * NOISE_MARGIN))


def utterances(transcribe, frames: queue.Queue, stop: threading.Event,
               threshold: float = THRESHOLD_MIN):
    #The trigger is sound STOPPING, the same trick the settle loop upstairs runs on: RMS
    #over the threshold opens an utterance, SILENCE_LIMIT of quiet closes it. No VAD model.
    buf, quiet = [], 0
    maxQuiet = int((RATE / CHUNK) * SILENCE_LIMIT)
    maxChunks = int((RATE / CHUNK) * MAX_UTTERANCE_S)

    while not stop.is_set():
        try:
            raw = frames.get(timeout=0.1)
        except queue.Empty:
            continue
        chunk = chunkOf(raw)
        if loudness(chunk) > threshold:
            buf.append(chunk)
            quiet = 0
        elif buf: #Keep the tail: a phrase clipped at the last loud sample loses its end
            buf.append(chunk)
            quiet += 1

        if buf and (quiet > maxQuiet or len(buf) > maxChunks):
            if quiet <= maxQuiet: #Cut by the clock, not by a pause: the room never dropped
                print(f"[voice] {MAX_UTTERANCE_S:.0f}s with no pause -- the room is louder "
                      f"than the {threshold:.0f} threshold. Pass --threshold to raise it")
            audio = np.concatenate(buf)
            buf, quiet = [], 0
            text = transcribe(audio)
            if text:
                yield Heard(text=text, audio=audio)


def drain(frames: queue.Queue):
    #Everything the mic heard while Omni was talking was Omni. Without this the desk
    #answers its own answer.
    while not frames.empty():
        frames.get_nowait()


# ---- a turn -------------------------------------------------------------

def answer(question: str, audio: np.ndarray | None, key: str, state: dict,
           events: list[dict] = (), url: str = SERVICE_URL, withImage: bool = True,
           root: str = VOICE_DIR, voice: str = VOICE,
           transport=None) -> tuple[str, bytes]:
    #The world arrives already read, so a service that is down and a model that is down
    #are two different failures and get told apart by the caller.
    #
    #`transport` is resolved below rather than defaulted to ask in the signature: a default
    #bound at import is one nothing can replace afterwards, and a test that quietly reaches
    #the network and passes anyway is worse than no test at all.
    prompt = promptFor(contextJson(state, events), question)

    wav = writeWav(os.path.join(root, "question.wav"), pcm16(audio), RATE) if audio is not None else ""
    frame = fetchFrame(url, os.path.join(root, "frame.jpg")) if withImage else ""

    text, reply = (transport or ask)(prompt, key, questionWav=wav,
                                     framePath=frame, voice=voice)
    if reply:
        writeWav(os.path.join(root, "reply.wav"), reply, REPLY_RATE)
    return text, reply


def speak(question: str, audio: np.ndarray | None, key: str, url: str = SERVICE_URL,
          withImage: bool = True, root: str = VOICE_DIR, voice: str = VOICE):
    #One question, one spoken answer. Anything that goes wrong is said out loud in the
    #terminal and costs a turn: the world model is in another process and does not care.
    try:
        state, events = fetchState(url), fetchEvents(url)
    except (urllib.error.URLError, OSError) as err:
        print(f"service unreachable ({err}), question dropped")
        return
    try:
        text, reply = answer(question, audio, key, state, events, url=url,
                             withImage=withImage, root=root, voice=voice)
    except Exception as err:
        print(f"[voice] no answer: {type(err).__name__}: {err}")
        return

    print(f"desk: {text}" if text else "[voice] nothing came back")
    if not reply:
        print("[voice] text only, no audio in the reply")
        return
    try:
        print(f"[voice] speaking {len(reply) / 2 / REPLY_RATE:.1f}s")
        play(reply)
    except Exception as err: #No speakers, no pyaudio, wrong device -- the wav is still there
        print(f"[voice] cannot play audio ({type(err).__name__}: {err}); "
              f"saved to {os.path.join(root, 'reply.wav')}")


def listen(key: str, url: str = SERVICE_URL, wake: str = WAKE_PHRASE,
           withImage: bool = True, root: str = VOICE_DIR, voice: str = VOICE,
           model: str = WHISPER_MODEL, threshold: float | None = None):
    frames, stop = queue.Queue(), threading.Event()
    threading.Thread(target=micFrames, args=(frames, stop), daemon=True).start()
    transcribe = whisper(model) #Loaded first, so the room is measured with the mic settled

    if threshold is None:
        print("measuring the room, hold still for a second...")
        floor = noiseFloor(frames)
        threshold = thresholdFor(floor)
        print(f"room noise is {floor:.0f}, so quiet means under {threshold:.0f}")
    said = utterances(transcribe, frames, stop, threshold)

    print(f'\n--- standing by. Say "{wake}" and then your question ---')
    try:
        for heard in said:
            print(f'heard: "{heard.text}"')
            question = wakeSplit(heard.text, wake)
            if question is None:
                continue
            if not question: #The phrase stood alone, so the question is the next thing said
                print("listening for the question...")
                heard = next(said, None)
                if heard is None:
                    break
                question = heard.text
                print(f'asked: "{question}"')

            speak(question, heard.audio, key, url, withImage, root, voice)
            drain(frames)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=SERVICE_URL, help="where the world model is")
    ap.add_argument("--wake", default=WAKE_PHRASE, help="say this, then the question")
    ap.add_argument("--ask", help="one typed question, answered aloud, then exit")
    ap.add_argument("--no-image", action="store_true",
                    help="do not send the camera frame with the question")
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--whisper", default=WHISPER_MODEL)
    ap.add_argument("--threshold", type=float,
                    help="what counts as quiet, RMS on the int16 scale. Measured if unset")
    ap.add_argument("--dir", default=VOICE_DIR, help="where this turn's wavs land")
    args = ap.parse_args()

    key = omni.apiKey()
    if key is None:
        raise SystemExit(f"no {omni.API_KEY_ENV}: put it in .env or export it")
    try:
        state = fetchState(args.url)
    except (urllib.error.URLError, OSError) as err:
        raise SystemExit(f"no world model at {args.url} ({err}). Is service.py running?")
    print(f"{len(state.get('entities', []))} entities at {args.url}")

    if args.ask:
        speak(clean(args.ask), None, key, args.url, not args.no_image, args.dir, args.voice)
        return
    listen(key, args.url, args.wake, not args.no_image, args.dir, args.voice,
           args.whisper, args.threshold)


if __name__ == "__main__":
    main()
