from calib import overlapFraction, solveHomography
from agent import AgentTracker, MIN_AREA_PX, borderBlob, enteringEdge, farTip, touchesBorder
from perceive import ChromaDetector, Config, matMaskPx
from synth import AGENT_MIN_AREA_PX, SCRIPT, matBackground, render
import numpy as np
import cv2
import pytest

MUG_FP = (110.0, 160.0, 190.0, 240.0) #The mug's footprint in mm, from the script


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("agentclip")
    truth = render(str(d / "s.mp4"), str(d / "s.json"), script=SCRIPT[:1])
    cap = cv2.VideoCapture(str(d / "s.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return {"truth": truth, "frames": frames, "H": solveHomography(frames[0])}


def handFrame(clip) -> int:
    #A frame the answer key says has a hand well inside the shot.
    return max(range(len(clip["frames"])),
               key=lambda i: clip["truth"]["frame"][i]["agent"]["area_px"])


def test_the_fixture_and_the_tracker_agree_on_what_counts_as_an_arm():
    #synth pins its answer key to this number. If they drift, the key says "hand" on
    #frames the tracker calls empty and every agent test becomes a coin toss.
    assert AGENT_MIN_AREA_PX == MIN_AREA_PX


def test_touching_the_border_is_measured_against_what_the_detector_can_see():
    frame = (0, 0, 1280, 720)
    mat = (240, 60, 1040, 660)
    assert touchesBorder((0, 300, 200, 400), frame)
    assert not touchesBorder((300, 300, 400, 400), frame)
    assert touchesBorder((240, 300, 400, 400), mat) #Hard against the mat's left edge
    assert not touchesBorder((240, 300, 400, 400), frame) #But nowhere near the frame's


def test_an_arm_reaching_in_is_found_and_a_bare_desk_is_not(clip):
    cfg = Config()
    det = ChromaDetector(cfg, clip["H"], clip["frames"][0].shape)
    _mask, bounds = matMaskPx(clip["H"], clip["frames"][0].shape)

    assert borderBlob(det.mask(clip["frames"][handFrame(clip)]), bounds) is not None
    assert borderBlob(det.mask(clip["frames"][0]), bounds) is None #Empty mat, no markers big enough


def test_the_far_tip_is_the_hand_and_not_the_elbow():
    #A horizontal bar entering from the left: the hand is its right end.
    blob = np.zeros((100, 200), bool)
    blob[40:60, 0:150] = True
    assert farTip(blob, "left")[0] == 149
    assert farTip(blob, "right")[0] == 0

    stats = (0, 40, 150, 20, 3000)
    assert enteringEdge(stats, (0, 0, 200, 100)) == "left"


def test_the_swept_path_covers_what_the_hand_reached_for(clip):
    #H3 asks whether the agent passed over the place something used to be. If the path
    #does not cover the object the hand demonstrably picked up, H3 can never fire.
    cfg = Config()
    det = ChromaDetector(cfg, clip["H"], clip["frames"][0].shape)
    agent = AgentTracker(det.mask, det.bounds)

    for f in clip["frames"]:
        agent.update(f, None, clip["H"])

    swept = agent.swept_mm()
    assert len(swept) > 10 #The hand is in shot for most of a two second beat
    assert max(overlapFraction(MUG_FP, s) for s in swept) > 0.3 #resolve.SWEPT_MIN


def test_reset_forgets_the_burst(clip):
    cfg = Config()
    det = ChromaDetector(cfg, clip["H"], clip["frames"][0].shape)
    agent = AgentTracker(det.mask, det.bounds)

    agent.update(clip["frames"][handFrame(clip)], None, clip["H"])
    assert agent.swept_mm()
    agent.present = True

    agent.reset()
    assert agent.swept_mm() == [] and agent.present is False
