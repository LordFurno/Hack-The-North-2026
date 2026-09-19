from core import Status
from world import World
from fake import SCRIPT, FakeDesk
from service import applyObservation, recordSnapshot
import pytest


def runScript(upto: int) -> World:
    world, desk = World(), FakeDesk()
    for row in SCRIPT[:upto]:
        applyObservation(world, desk.step(*row))
    return world


def test_one_snapshot_per_settle():
    world = runScript(5)
    assert len(world.snapshots) == 5
    assert [s["seq"] for s in world.snapshots] == sorted(s["seq"] for s in world.snapshots)


def test_a_snapshot_carries_the_keyframe_it_was_believed_from():
    #Scrubbing shows the map state AND the image behind it, which is the whole reason
    #Event and snapshot both carry frame_ref.
    world = runScript(3)
    assert world.snapshots[-1]["frame_ref"] == "fake-002"
    assert all(s["frame_ref"] for s in world.snapshots)


def test_a_snapshot_is_a_moment_and_does_not_move_when_the_world_does():
    #Inverting the event log to rebuild a past state is the trap this avoids: positions
    #are resolved when they are true, so a parent moving later cannot rewrite history.
    world = runScript(3) #Mug hidden under the box at (150, 200)
    covering = world.snapshots[-1]
    hidden = next(e for e in covering["entities"] if e["status"] == Status.HIDDEN.value)
    was = list(hidden["pos"])

    desk = FakeDesk() #Carry on: slide the box, which drags the mug with it
    for row in SCRIPT[:3]:
        desk.step(*row)
    applyObservation(world, desk.step(9.0, "move", "box", (420, 300)))

    assert len(world.snapshots) == 4
    assert covering["entities"][0]["pos"] is not None
    assert list(next(e for e in world.snapshots[-2]["entities"]
                     if e["id"] == hidden["id"])["pos"]) == was


def test_a_snapshot_is_shaped_like_the_state_the_dashboard_already_draws():
    #The scrubber feeds these straight back into drawMap, so they have to carry the same
    #keys a live frame does.
    world = runScript(2)
    snap, entity = world.snapshots[-1], world.snapshots[-1]["entities"][0]

    assert set(snap) >= {"ts", "mat", "entities", "frame_ref", "seq"}
    assert snap["mat"] == [0.0, 0.0, 600.0, 450.0]
    assert set(entity) >= {"id", "label", "status", "parent", "relation", "pos",
                           "footprint", "confidence", "is_agent", "location"}


def test_the_snapshot_endpoint_pages_from_a_cursor():
    from fastapi.testclient import TestClient
    import service

    service.WORLD = runScript(4)
    client = TestClient(service.app)

    everything = client.get("/snapshots").json()
    assert everything["count"] == 4
    assert len(everything["snapshots"]) == 4

    tail = client.get("/snapshots?since=3").json()
    assert tail["count"] == 4 and len(tail["snapshots"]) == 1


def test_a_missing_keyframe_is_a_404_not_a_traversal():
    from fastapi.testclient import TestClient
    import service

    client = TestClient(service.app)
    assert client.get("/snaps/nothing-here.jpg").status_code == 404
    assert client.get("/snaps/..%2F..%2Fservice.py").status_code == 404
