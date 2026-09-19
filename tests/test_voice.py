from core import AGENT_ID
from world import World
from resolve import settle
from fake import SCRIPT, FakeDesk
from service import stateJson, eventJson
import numpy as np
import time
import json
import wave
import voice
import pytest


def labelledWorld(upto: int = 3) -> World:
    #The same desk the tools tests use: a mug, then a box put down on top of it.
    world, desk = World(), FakeDesk()
    for row in SCRIPT[:upto]:
        settle(world, desk.step(*row))
    for name in ("mug", "box"):
        v = desk.embedding(name)
        max(world.entities.values(), key=lambda e: e.bank.best(v)).label = name
    return world


def context(upto: int = 3) -> dict:
    world = labelledWorld(upto)
    now = time.time()
    return voice.contextJson(stateJson(world, now),
                             [eventJson(world, ev) for ev in world.events])


def sse(*chunks: dict) -> list[str]:
    return [f"data: {json.dumps(c)}" for c in chunks] + ["", "data: [DONE]"]


# ---- the wake phrase ----------------------------------------------------

def test_the_question_is_whatever_follows_the_wake_phrase():
    assert voice.wakeSplit("test test where is the rubik's cube") == "where is the rubik's cube"
    assert voice.wakeSplit("okay test test what's in the tin") == "what's in the tin"
    assert voice.wakeSplit("test test") == "" #Phrase alone: the question is the next utterance
    assert voice.wakeSplit("where is the mug") is None #Not addressed to us


def test_whisper_punctuation_does_not_break_the_phrase():
    #"Test, test. Where's the mug?" has to trigger, and arrive as a readable question.
    assert voice.clean("Test, test.  Where's the mug?") == "test test where's the mug"
    assert voice.wakeSplit(voice.clean("Test, test. Where's the mug?")) == "where's the mug"


# ---- the world, as context ----------------------------------------------

def test_context_carries_the_spoken_location_and_the_confidence():
    ctx = context()
    byName = {o["object"]: o for o in ctx["objects"]}

    assert byName["mug"]["where"] == "under the box, on the desk"
    assert byName["mug"]["status"] == "HIDDEN"
    assert byName["mug"]["confidence"] == 0.8 #What picks the hedge in the spoken answer
    assert byName["box"]["confidence"] == 1.0


def test_the_hand_is_not_an_object_to_find():
    ctx = context()
    assert AGENT_ID not in [o["id"] for o in ctx["objects"]]
    assert ctx["object_count"] == len(ctx["objects"]) == 2


def test_objects_carry_their_neighbours_in_centimetres():
    #"Next to the keyboard" is the answer people want, and Omni is not doing the arithmetic.
    ctx = context()
    byName = {o["object"]: o for o in ctx["objects"]}

    assert byName["mug"]["nearest"][0]["object"] == "box"
    assert byName["mug"]["nearest"][0]["distance_cm"] == 0.0 #Covered: same place
    assert byName["mug"]["position_cm"] == [15.0, 20.0] #mm in the model, cm in the prompt
    assert len(byName["box"]["nearest"]) == 1 #Only one other thing on this desk


def test_context_carries_recent_events_with_their_cause():
    ctx = context()
    kinds = [ev["kind"] for ev in ctx["recent_events"]]

    assert "COVERED" in kinds
    covered = next(ev for ev in ctx["recent_events"] if ev["kind"] == "COVERED")
    assert covered["object"] == "mug"
    assert covered["cause"] == "covered_by:box" #The occluder by name, never by id
    assert covered["seconds_ago"] >= 0


def test_no_entity_id_is_ever_put_in_front_of_the_voice():
    #An id in the prompt is an id that can be read out loud, and "covered by 4f2a91c8" is
    #not a sentence. Ids stay on the objects, where they are addressing and not prose.
    ctx = context()
    ids = {o["id"] for o in ctx["objects"]}
    for ev in ctx["recent_events"]:
        assert not (ids & set(ev["cause"].split(":")))
    assert voice.causeText("agent", {}) == "agent" #Not everything names one
    assert voice.causeText("unexplained", {"x": "y"}) == "unexplained"


def test_events_are_capped_so_the_prompt_stays_small():
    world = labelledWorld(5)
    evs = [eventJson(world, ev) for ev in world.events]
    ctx = voice.contextJson(stateJson(world, time.time()), evs[-3:])
    assert len(ctx["recent_events"]) == 3


# ---- the prompt ---------------------------------------------------------

def test_the_prompt_is_the_world_state_and_the_question():
    prompt = voice.promptFor(context(), "where is the mug")

    assert "under the box, on the desk" in prompt
    assert "where is the mug" in prompt
    assert json.loads(prompt[prompt.index("{"):prompt.rindex("}") + 1])["object_count"] == 2


def test_the_system_prompt_forbids_inventing_a_location():
    #The single line that stops a model with a camera making positions up.
    assert "MUST come from that block" in voice.SYSTEM
    assert "never infer one from the camera image" in voice.SYSTEM.lower()
    assert "do not know" in voice.SYSTEM


def test_a_question_with_no_transcript_still_asks_something():
    #Whisper can return nothing useful while the audio is perfectly clear.
    assert "in the audio" in voice.promptFor({}, "   ")


# ---- the reply ----------------------------------------------------------

def test_text_and_speech_are_collected_from_the_stream():
    text, audio, usage = voice.collect(sse(
        {"choices": [{"delta": {"content": "It's "}}]},
        {"choices": [{"delta": {"content": "under the box."}}]},
        {"choices": [{"delta": {"audio": {"data": "aGVsbG8="}}}]}, #"hello", split over
        {"choices": [{"delta": {"audio": {"data": ""}}}]},         #chunks that need joining
        {"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 40}}))

    assert text == "It's under the box."
    assert audio == b"hello"
    assert usage["prompt_tokens"] == 900 #Rides the last chunk, and the ledger needs it


def test_base64_split_mid_quantum_still_decodes():
    #One chunk of base64 is not required to be a multiple of four characters.
    _text, audio, _usage = voice.collect(sse(
        {"choices": [{"delta": {"audio": {"data": "aGV"}}}]},
        {"choices": [{"delta": {"audio": {"data": "sbG8="}}}]}))
    assert audio == b"hello"


def test_a_stream_with_only_spoken_audio_still_has_text():
    text, _audio, _usage = voice.collect(sse(
        {"choices": [{"delta": {"audio": {"transcript": "it's under the box"}}}]}))
    assert text == "it's under the box"


def test_junk_in_the_stream_is_skipped_not_raised():
    text, _audio, _usage = voice.collect([": ping", "", "data: not json",
                                          'data: {"choices": [{"delta": {"content": "hi"}}]}',
                                          "data: [DONE]"])
    assert text == "hi"


# ---- a turn -------------------------------------------------------------

def test_a_turn_sends_the_question_audio_and_saves_the_reply(tmp_path, monkeypatch):
    sent = {}

    def transport(prompt, key, questionWav="", framePath="", voice_=None, **kw):
        sent.update(prompt=prompt, key=key, wav=questionWav, frame=framePath)
        return "It's under the box.", b"\x00\x01" * 1200

    state = stateJson(labelledWorld(), time.time())
    tone = np.sin(np.linspace(0, 200, RATE_CHUNK := 4000)).astype(np.float32)
    text, reply = voice.answer("where is the mug", tone, "k", state, url="http://nowhere",
                               withImage=False, root=str(tmp_path), transport=transport)

    assert text == "It's under the box." and len(reply) == 2400
    assert "under the box, on the desk" in sent["prompt"] #Grounded, not guessed
    assert sent["frame"] == "" #--no-image, so nothing was fetched

    with wave.open(sent["wav"]) as w: #What went up is what whisper heard
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (voice.RATE, 1, 2)
        assert w.getnframes() == RATE_CHUNK
    with wave.open(str(tmp_path / "reply.wav")) as w:
        assert w.getframerate() == voice.REPLY_RATE and w.getnframes() == 1200


def test_a_typed_question_sends_no_audio(tmp_path):
    sent = {}

    def transport(prompt, key, questionWav="", framePath="", **kw):
        sent.update(wav=questionWav)
        return "It's under the box.", b""

    voice.answer("where is the mug", None, "k", stateJson(labelledWorld(), time.time()),
                 withImage=False, root=str(tmp_path), transport=transport)
    assert sent["wav"] == ""
    assert not (tmp_path / "reply.wav").exists() #Nothing came back, so nothing was written


def test_a_dead_service_costs_a_turn_and_nothing_else(tmp_path, monkeypatch, capsys):
    #The world model is in another process. This one losing its voice must not touch it.
    def boom(url):
        raise OSError("connection refused")

    monkeypatch.setattr(voice, "fetchState", boom)
    voice.speak("where is the mug", None, "k", root=str(tmp_path))
    assert "service unreachable" in capsys.readouterr().out


def test_a_failed_call_is_reported_and_not_raised(tmp_path, monkeypatch, capsys):
    def boom(*a, **kw):
        raise OSError("no route to host")

    world = labelledWorld()
    monkeypatch.setattr(voice, "fetchState", lambda url: stateJson(world, time.time()))
    monkeypatch.setattr(voice, "fetchEvents", lambda url: [])
    monkeypatch.setattr(voice, "fetchFrame", lambda url, path: "")
    monkeypatch.setattr(voice, "ask", boom) #Reached only because answer() looks ask up late

    voice.speak("where is the mug", None, "k", root=str(tmp_path))
    assert "no answer" in capsys.readouterr().out


def test_the_stubbed_call_is_the_one_that_runs(tmp_path, monkeypatch):
    #The regression this exists for: with `transport=ask` bound in the signature, a stub
    #set on the module was ignored and the suite really did POST to yibuapi, 401, and pass.
    monkeypatch.setattr(voice, "ask", lambda *a, **kw: ("stubbed", b""))

    said = voice.answer("where is the mug", None, "k",
                        stateJson(labelledWorld(), time.time()),
                        withImage=False, root=str(tmp_path))
    assert said[0] == "stubbed"


# ---- the ledger ---------------------------------------------------------

def test_every_call_is_audited_and_separable_from_labelling():
    #Streaming audio out cannot go through their one-shot chat_completion, so the token
    #record is appended by hand -- to the same ledger, or the spend is invisible.
    import inspect
    src = inspect.getsource(voice.ask)
    assert "append_audit_record" in src and "build_omni_messages" in src
    assert src.count("append_audit_record") == 2 #The failed call is a call that cost time
    assert voice.PURPOSE != __import__("omni").PURPOSE

    audit = voice.yibuAudit()
    assert callable(audit.append_audit_record)


def chunks(queue_, level: float, n: int):
    #n chunks of white noise at roughly this RMS, on the int16 scale.
    for _ in range(n):
        noise = np.random.randn(voice.CHUNK) * level
        queue_.put(noise.astype(np.int16).tobytes())


def test_what_counts_as_quiet_is_measured_not_assumed():
    #The bug this fixes, with the numbers off the mic that showed it: a noise floor of
    #~411 RMS against a hard-coded threshold of 300 means no chunk is ever quiet, no
    #utterance ever ends, and every question runs to the 15 second cap instead.
    import queue
    frames = queue.Queue()
    chunks(frames, 420, voice.NOISE_CHUNKS)

    floor = voice.noiseFloor(frames)
    assert 300 < floor < 550 #The room, as measured
    assert voice.thresholdFor(floor) > floor #Quiet is now above this room's noise, not under


def test_the_measured_threshold_stays_between_its_limits():
    assert voice.thresholdFor(0.0) == voice.THRESHOLD_MIN #A silent room, not a hair-trigger
    assert voice.thresholdFor(1e9) == voice.THRESHOLD_MAX #Someone talking over it, not deaf
    assert voice.thresholdFor(800.0) == 800.0 * voice.NOISE_MARGIN


def test_a_measurement_with_no_microphone_falls_back_rather_than_hanging():
    import queue
    assert voice.noiseFloor(queue.Queue(), chunks=1) == 0.0
    assert voice.thresholdFor(0.0) == voice.THRESHOLD_MIN


def test_a_noisy_room_still_ends_an_utterance_once_the_floor_is_measured():
    #End to end on the failing case: floor ~420, speech ~2500, and the utterance must
    #close on the pause rather than run to the cap.
    import queue, threading
    frames, stop = queue.Queue(), threading.Event()
    chunks(frames, 420, voice.NOISE_CHUNKS)
    threshold = voice.thresholdFor(voice.noiseFloor(frames))

    chunks(frames, 2500, 6) #Speaking, well over the room
    chunks(frames, 420, int((voice.RATE / voice.CHUNK) * voice.SILENCE_LIMIT) + 2)

    heard = next(voice.utterances(lambda a: "where is the mug", frames, stop, threshold))
    stop.set()
    assert len(heard.audio) < voice.RATE * voice.MAX_UTTERANCE_S #Not cut by the clock


def test_an_utterance_ends_when_the_sound_stops():
    #The trigger is sound stopping, so a loud stretch followed by quiet is one utterance.
    import queue, threading
    frames, stop = queue.Queue(), threading.Event()
    loud = (np.sin(np.linspace(0, 100, voice.CHUNK)) * 20000).astype(np.int16).tobytes()
    quiet = np.zeros(voice.CHUNK, np.int16).tobytes()

    for _ in range(5):
        frames.put(loud)
    for _ in range(int((voice.RATE / voice.CHUNK) * voice.SILENCE_LIMIT) + 2):
        frames.put(quiet)

    said = voice.utterances(lambda audio: f"{len(audio)} samples", frames, stop)
    heard = next(said)
    stop.set()

    assert heard.text.endswith("samples")
    assert heard.audio.dtype == np.float32
    assert len(heard.audio) > 5 * voice.CHUNK #The quiet tail is kept, not clipped off


def test_the_queue_is_drained_so_the_desk_does_not_answer_itself():
    import queue
    frames = queue.Queue()
    for _ in range(4):
        frames.put(b"\x00\x00")
    voice.drain(frames)
    assert frames.empty()
