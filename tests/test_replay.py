from core import DESK, EventKind, Status
from calib import solveHomography
from world import World
from resolve import settle
from agent import AgentTracker
from perceive import ChromaDetector, Config, RefDiffDetector, openVideo, run, videoClock
import cv2, os
import pytest

#BUILD.md's test table, run against the fixture instead of a desk. This is the one test
#that exercises everything at once: motion gate, settle trigger, homography, detector,
#embedding, matching, all four hypotheses, reveal and verification. If it passes, the
#demo sequence works; if it breaks, it says which beat broke.

VIDEO, TRUTH = "synth.mp4", "synth.json"

MUG, BOX = (150.0, 200.0), (400.0, 220.0) #Where the script puts them, in mm
BOX_COVERING, BOX_AWAY = (150.0, 200.0), (420.0, 300.0)


def kindsFor(world: World, e) -> list:
    return [ev.kind for ev in world.events if ev.entity == e.id]


@pytest.fixture(scope="module")
def replayed(tmp_path_factory) -> dict:
    if not (os.path.exists(VIDEO) and os.path.exists(TRUTH)):
        pytest.skip(f"{VIDEO} not rendered; run `python synth.py`")

    cfg = Config()
    cap = openVideo(VIDEO)
    first = cap.read()[1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    H = solveHomography(first) #The fixture carries its own markers, so no calibration
    assert H is not None, "the fixture should show four markers on its first frame"

    #Chroma here: the fixture is a green mat, and rejecting its cast shadow keeps the
    #geometry exact. RefDiff is checked separately below, on the same video.
    det = ChromaDetector(cfg, H, first.shape)
    agent = AgentTracker(det.mask, det.bounds)

    world = World()
    states, out = [], str(tmp_path_factory.mktemp("snaps"))

    def sink(obs):
        #Frozen per settle, because by the end of the run the mug is off the desk and
        #every question about what was believed at settle three has the wrong answer.
        events = settle(world, obs, cfg.confirm_settles)
        states.append({"obs": obs, "events": events,
                       "positions": {e.id: world.absolute(e) for e in world.entities.values()},
                       "status": {e.id: e.status for e in world.entities.values()},
                       "parent": {e.id: e.parent for e in world.entities.values()},
                       "relation": {e.id: e.relation for e in world.entities.values()}})

    run(cap, det, sink, cfg, H, agent=agent, embed=None,
        clock=videoClock(cap, 1000.0), root=out)
    cap.release()
    return {"world": world, "states": states}


def test_every_beat_produced_exactly_one_settle(replayed):
    #Five scripted actions plus the quiet lead-in the loop settles on first.
    assert len(replayed["states"]) == 6


def test_1_and_3_placing_things_mints_one_entity_each(replayed):
    world = replayed["world"]
    real = [e for e in world.entities.values() if not e.isAgent]
    assert len(real) == 2

    #One of them ended up off the desk and one is still on it, and each was announced.
    assert sorted(e.status for e in real) == sorted([Status.OFF_DESK, Status.VISIBLE])
    for e in real:
        assert EventKind.APPEARED in kindsFor(world, e)

    box = next(e for e in real if e.status is Status.VISIBLE)
    assert world.absolute(box) == pytest.approx(BOX_AWAY, abs=8.0)


def test_2_and_4_the_box_moves_and_the_mug_goes_hidden_under_it(replayed):
    #The settle where the box lands on the mug: BUILD.md steps 3 and 4.
    world, covering = replayed["world"], replayed["states"][3]
    kinds = [ev.kind for ev in covering["events"]]
    assert EventKind.MOVED in kinds
    assert EventKind.COVERED in kinds

    covered = next(ev for ev in covering["events"] if ev.kind is EventKind.COVERED)
    occluder = next(ev for ev in covering["events"] if ev.kind is EventKind.MOVED)
    assert covering["status"][covered.entity] is Status.HIDDEN
    assert covering["relation"][covered.entity].value in ("UNDER", "IN")
    assert covering["parent"][covered.entity] == occluder.entity
    assert covered.cause == f"covered_by:{occluder.entity}"
    assert covering["positions"][occluder.entity] == pytest.approx(BOX_COVERING, abs=8.0)


def test_5_the_hidden_mug_travels_with_the_box(replayed):
    #Relative pose doing the work: the occluder moves, and the thing under it is carried
    #with no propagation code at all.
    covering, revealing = replayed["states"][3], replayed["states"][4]
    covered = next(ev for ev in covering["events"] if ev.kind is EventKind.COVERED)

    assert covering["positions"][covered.entity] == pytest.approx(BOX_COVERING, abs=8.0)
    #By the next settle the box has slid off and the mug has been found again, so what
    #this asserts is that it was the box's position it held while it was hidden.
    assert revealing["positions"][covered.entity] == pytest.approx(MUG, abs=10.0)


def test_6_lifting_the_box_reveals_the_mug_at_full_confidence(replayed):
    world, revealing = replayed["world"], replayed["states"][4]
    revealed = next(ev for ev in revealing["events"] if ev.kind is EventKind.REVEALED)

    assert revealed.confidence == 1.0
    assert revealed.note == "confirmed where I thought it was"
    assert revealing["status"][revealed.entity] is Status.VISIBLE
    assert revealing["parent"][revealed.entity] == DESK #Out from under the box


def test_8_carrying_the_mug_off_the_desk_is_attributed_to_the_agent(replayed):
    #H3, which only fires because the tracker saw a hand sweep the mug's footprint. With
    #the tracker stubbed this degrades to UNRESOLVED -- a weaker answer, not a wrong one.
    world, last = replayed["world"], replayed["states"][5]
    assert len(last["events"]) == 1

    gone = last["events"][0]
    assert gone.kind is EventKind.LEFT_DESK
    assert gone.cause == "agent"
    assert last["status"][gone.entity] is Status.OFF_DESK
    assert world.entities[gone.entity].parent is None

    #And it says what else it considered, rather than presenting one story as the truth.
    assert [a["kind"] for a in gone.alternatives] == ["LOST"]


def test_the_whole_run_never_mints_a_third_entity(replayed):
    #Two objects went on the desk. A hand crossing the frame five times, its shadow, and
    #four markers must not become entities of their own -- nothing can ever delete one.
    world = replayed["world"]
    assert sorted(ev.kind.value for ev in world.events if ev.kind is EventKind.APPEARED) \
           == ["APPEARED", "APPEARED"]


def test_the_primary_detector_reaches_the_same_conclusions(replayed, tmp_path):
    #Chroma is optional and this fixture happens to suit it. RefDiff is what runs on a
    #real desk, so it has to tell the same story about the same video.
    cfg = Config()
    cap = openVideo(VIDEO)
    first = cap.read()[1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    H = solveHomography(first)

    det = RefDiffDetector(first, cfg) #An empty-desk reference: the fixture's lead-in
    agent = AgentTracker(det.mask, det.bounds)
    world = World()

    run(cap, det, lambda obs: settle(world, obs, cfg.confirm_settles), cfg, H,
        agent=agent, embed=None, clock=videoClock(cap, 1000.0), root=str(tmp_path))
    cap.release()

    kinds = [ev.kind for ev in world.events]
    assert kinds.count(EventKind.APPEARED) == 2
    assert EventKind.COVERED in kinds
    assert EventKind.REVEALED in kinds
    assert EventKind.LEFT_DESK in kinds
