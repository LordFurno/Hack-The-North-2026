from core import AGENT_ID, EventKind, Status
from world import World
from resolve import settle
from fake import SCRIPT, FakeDesk, stream
from service import UI_PAGE, streamJson
import os
import time


def runScript(upto: int) -> World:
    world, desk = World(), FakeDesk()
    for row in SCRIPT[:upto]:
        settle(world, desk.step(*row))
    return world


def test_stream_carries_state_and_the_whole_log_first():
    #A page that connects late must be able to draw the run it missed.
    world = runScript(5)
    first = streamJson(world, time.time(), 0)

    assert first["mat"] == [0.0, 0.0, 600.0, 450.0]
    assert first["seq"] == len(world.events) == len(first["events"])
    assert [ev["seq"] for ev in first["events"]] == list(range(len(world.events)))
    assert {e["id"] for e in first["entities"]} == set(world.entities)


def test_stream_pushes_only_what_the_client_has_not_had():
    world = runScript(3)
    sent = streamJson(world, time.time(), 0)["seq"]

    quiet = streamJson(world, time.time(), sent)
    assert quiet["events"] == [] #Nothing settled, but the state still ticks
    assert quiet["entities"]

    settle(world, FakeDesk().step(*SCRIPT[0])) #Any further settle
    delta = streamJson(world, time.time(), sent)
    assert [ev["seq"] for ev in delta["events"]] == list(range(sent, len(world.events)))
    assert delta["events"] #Something did settle


def test_stream_entities_carry_what_the_map_draws():
    world = runScript(3)
    byId = {e["id"]: e for e in streamJson(world, time.time(), 0)["entities"]}
    mug = next(e for e in byId.values()
               if e["status"] == Status.HIDDEN.value and not e["is_agent"])

    assert byId[mug["parent"]]["status"] == Status.VISIBLE.value #Nested inside its parent
    assert len(mug["pos"]) == 2 and len(mug["footprint"]) == 2
    assert 0.0 <= mug["confidence"] <= 1.0 #Opacity tracks this
    assert byId[AGENT_ID]["is_agent"] is True #The map skips the hand, the header does not


def test_falsified_row_has_a_note_and_a_cause():
    world, desk = World(), FakeDesk()
    for row in SCRIPT[:3]:
        settle(world, desk.step(*row))
    desk.palm("mug")
    settle(world, desk.step(15.0, "remove", "box"))

    ev = next(ev for ev in streamJson(world, time.time(), 0)["events"]
              if ev["kind"] == EventKind.BELIEF_FALSIFIED.value)
    assert ev["note"] and ev["cause"] #The two things the red row shows
    assert ev["label"] == "unknown" or ev["label"] #A label the row can name it by


def test_fake_stream_stamps_the_wall_clock():
    #Script time against a wall clock would read as decades of decay.
    before = time.time()
    obs = next(stream(speed=1000.0))
    assert obs.ts >= before


def test_ui_page_is_served_from_the_repo():
    assert os.path.isfile(UI_PAGE)
    page = open(UI_PAGE, encoding="utf-8").read()
    assert "/stream" in page #The page reads the same stream any other client would
    assert "http" not in page.split("<script>")[0].replace("http-equiv", "") #No CDN, one file
