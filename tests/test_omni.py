from core import Detection, Entity
from world import World
from resolve import settle
from fake import FakeDesk
import numpy as np
import omni
import threading
import pytest


@pytest.fixture(autouse=True)
def forgetLabelled():
    omni._labelled.clear()
    yield
    omni._labelled.clear()


def crop(tmp_path) -> str:
    import cv2
    path = str(tmp_path / "crop.jpg")
    cv2.imwrite(path, np.full((40, 40, 3), 120, np.uint8))
    return path


def detFor(path: str) -> Detection:
    return Detection(centroid=(150.0, 200.0), size=(80.0, 80.0),
                     embedding=np.ones(4) / 2.0, crop_path=path)


def test_a_reply_becomes_a_label_and_a_container_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(omni.API_KEY_ENV, "test-key")
    e = Entity.new()
    omni.label(e, detFor(crop(tmp_path)),
               transport=lambda path, key, **kw: '{"label": "Mug", "isContainer": true}')

    assert e.label == "mug" #Lowered, because phrasing reads it straight out
    assert e.isContainer is True


def test_a_fenced_reply_is_still_parsed():
    #Models wrap JSON in prose and code fences however they like.
    assert omni.parse("Sure!\n```json\n{\"label\": \"tin\", \"isContainer\": true}\n```") \
           == ("tin", True)
    assert omni.parse('{"label": "pen", "isContainer": false}') == ("pen", False)
    assert omni.parse("I could not tell.") is None
    assert omni.parse('{"label": "", "isContainer": true}') is None #A name or nothing


def test_every_call_goes_through_the_audited_client():
    #A hand-rolled request would work and would be invisible in the token ledger, so the
    #test is that ask() is built out of their client and nothing else.
    import inspect
    src = inspect.getsource(omni.ask)
    assert "chat_completion" in src and "build_omni_messages" in src
    assert omni.PURPOSE #Named, so this project's spend is separable in the ledger

    http = omni.yibuHttp() #Importable by path, and carrying what ask() calls on it
    assert callable(http.chat_completion) and callable(http.build_omni_messages)


def test_without_an_api_key_nothing_is_called_and_the_label_stays_unknown(tmp_path, monkeypatch):
    monkeypatch.delenv(omni.API_KEY_ENV, raising=False)
    called = []
    e = Entity.new()

    omni.label(e, detFor(crop(tmp_path)), transport=lambda *a, **k: called.append(1))
    assert called == [] and e.label == "unknown"


def test_a_missing_crop_is_not_worth_a_call(tmp_path, monkeypatch):
    monkeypatch.setenv(omni.API_KEY_ENV, "test-key")
    called = []
    e = Entity.new()

    omni.label(e, detFor(str(tmp_path / "gone.jpg")), transport=lambda *a, **k: called.append(1))
    assert called == [] and e.label == "unknown"


def test_a_failed_call_leaves_the_entity_alone(tmp_path, monkeypatch, capsys):
    #Labelling is decoration. The world model never needed the name, so a network that
    #is down must cost nothing but the name.
    monkeypatch.setenv(omni.API_KEY_ENV, "test-key")
    e = Entity.new()

    def boom(*a, **kw):
        raise OSError("no route to host")

    omni.label(e, detFor(crop(tmp_path)), transport=boom)
    assert e.label == "unknown"
    assert "stays unknown" in capsys.readouterr().out


def test_minting_fires_the_hook_and_tracking_does_not_wait_for_it(monkeypatch):
    #The entity exists, is placed and is drawn before anything leaves the machine.
    monkeypatch.delenv(omni.API_KEY_ENV, raising=False)
    world, desk = World(), FakeDesk()
    seen = []
    world.onMint = lambda e, det: seen.append((e.id, e.label, det.crop_path))

    settle(world, desk.step(0.0, "appear", "mug", (150, 200), (80, 80)))

    assert len(seen) == 1
    assert seen[0][1] == "unknown" #Minted unlabelled, patched later or never
    assert seen[0][0] in world.entities


def test_one_entity_is_only_ever_asked_about_once(tmp_path, monkeypatch):
    monkeypatch.setenv(omni.API_KEY_ENV, "test-key")
    calls, done = [], threading.Event()
    e, det = Entity.new(), detFor(crop(tmp_path))

    def transport(path, key, **kw):
        calls.append(path)
        done.set()
        return '{"label": "mug", "isContainer": false}' 

    omni.labelAsync(e, det, transport=transport)
    assert done.wait(5.0)
    omni.labelAsync(e, det, transport=transport) #Cached forever, never re-called
    assert len(calls) == 1


def test_attach_wires_a_world_up_to_labelling():
    world = World()
    omni.attach(world, threading.Lock())
    assert world.onMint is not None
