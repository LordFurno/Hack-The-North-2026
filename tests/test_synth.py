from core import AGENT_ID, DESK
from calib import MAT_BOUNDS
from synth import (FPS, MARKER_MM, MARKER_PX, MARKER_QUIET_PX, Thing, hiddenFraction,
                   matBackground, mmToPx, palmAt, parentOf, pxToMm, render, timeline)
import numpy as np
import cv2, json
import pytest

SETTLE_FRAMES = 15 #What the settle loop needs quiet before it will analyse anything
SHORT = [(0.0, "appear", "mug", (150, 200), (80, 80)),
         (2.0, "move", "mug", (300, 300))]


def solveFrom(frame) -> np.ndarray:
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = det.detectMarkers(frame)
    seen = {int(i): c[0].mean(axis=0) for c, i in zip(corners, ids.flatten())}
    src = np.array([seen[i] for i in sorted(seen)], np.float32)
    dst = np.array([MARKER_MM[i] for i in sorted(seen)], np.float32)
    return cv2.findHomography(src, dst)[0]


def pxToMmVia(H: np.ndarray, pt) -> tuple[float, float]:
    q = H @ np.array([pt[0], pt[1], 1.0])
    return (float(q[0] / q[2]), float(q[1] / q[2]))


def test_the_four_markers_are_where_the_mat_says_they_are():
    #If a marker centre is half a pixel off its corner, every mm in the answer key is a
    #guess. detectMarkers reports pixel centres, so the render has to place them there.
    frame = matBackground()
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = det.detectMarkers(frame)

    seen = {int(i): c[0].mean(axis=0) for c, i in zip(corners, ids.flatten())}
    assert sorted(seen) == [0, 1, 2, 3]
    for i, mm in MARKER_MM.items():
        assert tuple(seen[i]) == pytest.approx(mmToPx(mm))


def test_the_solved_homography_inverts_the_renderer():
    #The whole point of the fixture: what perception recovers from the video is exactly
    #what synth used to draw it, so a disagreement downstream is a real disagreement.
    H = solveFrom(matBackground())
    for mm in [(0.0, 0.0), (600.0, 450.0), (300.0, 225.0), (150.0, 200.0), (420.0, 300.0)]:
        assert pxToMmVia(H, mmToPx(mm)) == pytest.approx(mm, abs=0.01)


def test_mm_and_px_round_trip():
    for mm in [(0.0, 0.0), (600.0, 450.0), (150.0, 200.0), (-80.0, 520.0)]:
        assert pxToMm(mmToPx(mm)) == pytest.approx(mm)


def test_an_object_is_carried_from_where_it_was_to_where_the_script_puts_it():
    beats, _ = timeline()
    move = next(b for b in beats if b.kind == "move" and b.name == "box")

    assert palmAt(move, move.grab) == pytest.approx((400.0, 220.0)) #Where the box was
    assert palmAt(move, move.release) == pytest.approx((150.0, 200.0)) #Where it goes
    assert move.grab > move.path[0][0] #Reaches in empty-handed first
    assert move.release < move.t #And lets go before it withdraws


def test_a_removed_object_never_leaves_the_hand():
    beats, _ = timeline()
    gone = next(b for b in beats if b.kind == "remove")
    assert gone.release == float("inf")


def test_every_beat_is_followed_by_enough_quiet_to_settle():
    #An action that runs into the next one gives the settle loop nothing to analyse, and
    #a quiet gap shorter than SETTLE_FRAMES is the same thing with extra steps.
    beats, duration = timeline()
    ends = [b.t for b in beats]
    starts = [b.path[0][0] for b in beats] + [duration]

    for end, nextStart in zip(ends, starts[1:]):
        assert (nextStart - end) * FPS > SETTLE_FRAMES


def test_a_box_over_a_mug_hides_it_and_becomes_its_parent():
    mug = Thing(name="mug", pos=(150, 200), size=(80, 80), z=0)
    box = Thing(name="box", pos=(150, 200), size=(160, 140), z=1)
    things = {"mug": mug, "box": box}

    assert parentOf(mug, things, held=None) == "box"
    assert hiddenFraction(mug, things, None, None) == pytest.approx(1.0)
    assert parentOf(box, things, held=None) == DESK #The small thing under it covers nothing
    assert hiddenFraction(box, things, None, None) == pytest.approx(0.0)

    box.pos = (420, 300) #Slid away, and the mug is its own object again
    assert parentOf(mug, things, held=None) == DESK
    assert hiddenFraction(mug, things, None, None) == pytest.approx(0.0)


def test_a_held_object_is_parented_to_the_agent():
    mug = Thing(name="mug", pos=(150, 200), size=(80, 80), z=0)
    assert parentOf(mug, {"mug": mug}, held="mug") == AGENT_ID


def test_render_writes_a_video_and_an_answer_key_that_agree(tmp_path):
    video, key = tmp_path / "s.mp4", tmp_path / "s.json"
    truth = render(str(video), str(key), script=SHORT)

    assert json.loads(key.read_text()) == truth
    assert truth["mat"]["mm"] == list(MAT_BOUNDS)
    assert len(truth["frame"]) == truth["frames"]

    cap = cv2.VideoCapture(str(video))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    assert decoded == truth["frames"]

    #The mug is nowhere before the hand brings it in, on the mat once it is let go, and
    #at the second script position by the end.
    assert truth["frame"][0]["objects"]["mug"]["present"] is False
    assert truth["frame"][-1]["objects"]["mug"]["mm"] == pytest.approx([300.0, 300.0])
    assert truth["frame"][-1]["objects"]["mug"]["parent"] == DESK
    assert truth["frame"][-1]["agent"]["present"] is False


def test_the_answer_key_agrees_with_the_pixels_it_shipped(tmp_path):
    #Segment a settled frame the way perception will and check the objects land where the
    #key says. A fixture whose key and video disagree is worse than no fixture.
    video, key = tmp_path / "s.mp4", tmp_path / "s.json"
    truth = render(str(video), str(key), script=SHORT)

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    i = truth["beats"][-1]["quiet_frame"] + SETTLE_FRAMES
    hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
    h, s, _v = cv2.split(hsv) #Value is unused on purpose: that is what rejects the shadow
    dh = np.minimum(np.abs(h.astype(int) - truth["mat"]["hue"]),
                    180 - np.abs(h.astype(int) - truth["mat"]["hue"]))
    fg = (~((dh < 18) & (s > 60))).astype(np.uint8)

    inset = MARKER_PX // 2 + MARKER_QUIET_PX + 6 #Clear of the corner markers, which are
    x0, y0, x1, y1 = (round(v + 0.5) for v in truth["mat"]["px"]) #foreground themselves
    n, _lbl, stats, _c = cv2.connectedComponentsWithStats(
        fg[y0 + inset:y1 - inset, x0 + inset:x1 - inset])
    found = [stats[j] for j in range(1, n) if stats[j][4] > 1500]

    assert len(found) == 1 #One object, and no phantom beside it where the shadow fell
    x, y, w, h_, _a = found[0]
    truthPx = truth["frame"][i]["objects"]["mug"]["px"]
    assert (x0 + inset + x + w / 2, y0 + inset + y + h_ / 2) == pytest.approx(truthPx, abs=3.0)
