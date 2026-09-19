from calib import (MAT_BOUNDS, MAT_MM, findMarkers, level, loadCalib, pxToMm,
                   rectPxToMm, saveCalib, solveHomography)
from synth import MARKER_MM, matBackground, mmToPx, rectPx
import numpy as np
import cv2
import pytest


def test_the_markers_the_code_looks_for_are_the_ones_the_fixture_draws():
    #If these two drift apart the homography is solved against the wrong answer key and
    #every millimetre downstream is quietly wrong.
    assert MAT_MM == MARKER_MM


def test_the_solved_homography_inverts_the_renderer():
    H = solveHomography(matBackground())
    for mm in [(0.0, 0.0), (600.0, 450.0), (300.0, 225.0), (150.0, 200.0), (420.0, 300.0)]:
        assert pxToMm(H, mmToPx(mm)) == pytest.approx(mm, abs=0.01)


def test_a_rect_survives_the_round_trip():
    H = solveHomography(matBackground())
    got = rectPxToMm(H, rectPx((150.0, 200.0), (80.0, 80.0)))
    assert got == pytest.approx((110.0, 160.0, 190.0, 240.0), abs=1.0)


def test_rect_mapping_takes_all_four_corners():
    #Under any rotation the two corners of a pixel box are not the two corners of the mm
    #box, and using them alone understates the region -- which understates every overlap.
    H = cv2.getRotationMatrix2D((0.0, 0.0), 30.0, 1.0)
    H = np.vstack([H, [0.0, 0.0, 1.0]])
    x0, y0, x1, y1 = rectPxToMm(H, (0, 0, 100, 100))

    corners = [H @ np.array([x, y, 1.0]) for x, y in ((0, 0), (100, 0), (100, 100), (0, 100))]
    assert x0 == pytest.approx(min(c[0] for c in corners))
    assert x1 == pytest.approx(max(c[0] for c in corners))
    assert y1 == pytest.approx(max(c[1] for c in corners))


def test_fewer_than_four_markers_keeps_the_previous_homography():
    frame = matBackground()
    frame[:, :400] = 0 #Black out the left edge, taking markers 0 and 3 with it
    assert sorted(findMarkers(frame)) == [1, 2]
    assert solveHomography(frame) is None


def test_levelling_matches_the_references_brightness_to_the_frame():
    #Exposure drift is a slow leak into the detector, and this is the whole defence.
    ref = np.full((20, 20, 3), 100, np.uint8)
    now = np.full((20, 20, 3), 150, np.uint8)
    assert level(ref, now).mean() == pytest.approx(now.mean(), abs=1.0)
    assert level(ref, ref).mean() == pytest.approx(ref.mean(), abs=1.0)


def test_levelling_survives_a_black_reference():
    #max(ref.mean(), 1.0) is load-bearing: a black frame would otherwise divide by zero.
    ref = np.zeros((8, 8, 3), np.uint8)
    assert level(ref, np.full((8, 8, 3), 200, np.uint8)).max() == 0


def test_calibration_round_trips_through_disk(tmp_path):
    H = solveHomography(matBackground())
    reference = matBackground()
    calib, png = tmp_path / "calib.json", tmp_path / "reference.png"
    saveCalib(H, reference, str(calib), str(png))

    back, ref = loadCalib(str(calib))
    assert back == pytest.approx(H)
    assert ref.shape == reference.shape
    assert list(MAT_BOUNDS) == [0.0, 0.0, 600.0, 450.0] #The units every threshold is in
