from core import AGENT_ID, DESK, Entity, EventKind, Relation, Status
from world import World
from resolve import settle
from fake import SCRIPT, FakeDesk
from service import (history, observationFrom, phraseLocation, resolveReferent,
                     stateJson, whatsIn, whereIs, whoMoved)
import numpy as np
import time
import pytest


def labelledWorld(upto: int) -> tuple[World, dict[str, Entity]]:
    #The fake stream carries no labels, so name the entities the way the demo does:
    #find them by embedding, then tell the world what they are.
    world, desk = World(), FakeDesk()
    for row in SCRIPT[:upto]:
        settle(world, desk.step(*row))

    named = {}
    for name in ("mug", "box"):
        v = desk.embedding(name)
        e = max(world.entities.values(), key=lambda e: e.bank.best(v))
        e.label = name
        named[name] = e
    return world, named


def test_phrase_location_walks_the_chain():
    world, named = labelledWorld(3)
    mug, box = named["mug"], named["box"]

    assert phraseLocation(world, box) == "on the desk, toward the middle left"
    assert phraseLocation(world, mug) == "under the box, on the desk"


def test_phrase_location_for_hand_and_off_desk():
    world, named = labelledWorld(5)
    mug = named["mug"]

    assert mug.status is Status.OFF_DESK
    assert phraseLocation(world, mug) == "off the desk"

    world.reparent(mug, AGENT_ID, Relation.HELD, (0.0, 0.0))
    assert phraseLocation(world, mug) == "in your hand"


def test_resolve_referent_matches_a_substring():
    world, named = labelledWorld(3)

    assert resolveReferent(world, "where is my mug") is named["mug"]
    assert resolveReferent(world, "BOX") is named["box"]
    assert resolveReferent(world, named["mug"].id) is named["mug"] #Ids round-trip
    assert resolveReferent(world, "stapler") is None
    assert resolveReferent(world, "hand") is None #The agent is not a referent


def test_where_is_the_covered_mug():
    world, named = labelledWorld(3)
    out = whereIs(world, "mug")

    assert out["found"] is True
    assert out["entity"] == "mug"
    assert out["location"] == "under the box, on the desk"
    assert out["status"] == "HIDDEN"
    assert out["confidence"] == 0.8 #Stabilised: the box is right there and has not moved
    assert out["supporting_event"]["kind"] == EventKind.COVERED.value
    assert out["supporting_event"]["cause"] == f"covered_by:{named['box'].id}"


def test_where_is_reports_not_found():
    world, _ = labelledWorld(3)
    assert whereIs(world, "screwdriver") == {"found": False, "query": "screwdriver"}


def test_whats_in_lists_descendants_and_the_desk():
    world, named = labelledWorld(3)

    box = whatsIn(world, "the box")
    assert box["found"] is True and box["count"] == 1
    assert box["contents"][0]["entity"] == "mug"
    assert box["contents"][0]["relation"] == Relation.UNDER.value

    desk = whatsIn(world, "what is on the desk")
    assert desk["container"] == "desk"
    assert [c["entity"] for c in desk["contents"]] == ["box"] #The hand is not furniture


def test_who_moved_names_the_occluder():
    world, named = labelledWorld(3)
    out = whoMoved(world, "mug")

    assert out["moved"] is True
    assert out["kind"] == EventKind.COVERED.value
    assert out["cause"] == f"covered_by:{named['box'].id}"
    assert out["frame_ref"] == "fake-002"


def test_history_is_chronological_and_limited():
    world, _ = labelledWorld(5)
    out = history(world, "mug")

    assert out["found"] is True
    assert [ev["kind"] for ev in out["events"]] == ["APPEARED", "COVERED", "REVEALED", "LEFT_DESK"]
    assert [ev["ts"] for ev in out["events"]] == sorted(ev["ts"] for ev in out["events"])
    assert [ev["kind"] for ev in history(world, "mug", limit=2)["events"]] == ["REVEALED", "LEFT_DESK"]


def test_state_carries_decayed_confidence_and_a_phrase():
    world, named = labelledWorld(3)
    state = stateJson(world, time.time())
    byId = {e["id"]: e for e in state["entities"]}

    assert byId[named["box"].id]["confidence"] == 1.0 #VISIBLE never decays
    assert byId[named["mug"].id]["location"] == "under the box, on the desk"
    assert byId[AGENT_ID]["is_agent"] is True


def test_observation_decodes_from_json():
    #The wire form of an Observation: embeddings as lists, rects as lists.
    obs = observationFrom({"ts": 4.0, "frame_ref": "http-000",
                           "detections": [{"centroid": [10, 20], "size": [30, 40],
                                           "embedding": [1.0, 0.0], "crop_path": "c.png"}],
                           "changed": [[0, 0, 50, 50]], "agent_present": True})

    assert obs.ts == 4.0 and obs.agent_present is True
    assert obs.changed == [(0, 0, 50, 50)] and obs.agent_swept == []
    assert isinstance(obs.detections[0].embedding, np.ndarray)
    assert obs.detections[0].centroid == (10, 20)

    world = World()
    settle(world, obs)
    assert len(world.childrenOf(DESK)) == 2 #The hand, and the thing that just appeared
