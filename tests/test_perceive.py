from calib import solveHomography
from perceive import (ChromaDetector, Config, DetectorError, RefDiffDetector, analyse,
                      changedRegions, components, histEmbed, intersects, matMaskPx,
                      observationJson)
from synth import SCRIPT, matBackground, mmToPx, render
import numpy as np
import cv2, json
import pytest

MUG_MM, MUG_SIZE = (150.0, 200.0), (80.0, 80.0)
#Three seconds apart, not two: one beat is 2.1 s of hand movement, and beats that run
#into each other leave no settled frame between them to look at.
SHORT = [(0.0, "appear", "mug", (150, 200), (80, 80)),
         (3.0, "move", "mug", (300, 300))]


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> dict:
    #A two-beat render, decoded once. Everything here needs the same handful of frames.
    d = tmp_path_factory.mktemp("clip")
    truth = render(str(d / "s.mp4"), str(d / "s.json"), script=SHORT)

    cap = cv2.VideoCapture(str(d / "s.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return {"truth": truth, "frames": frames, "dir": d,
            "H": solveHomography(frames[0])}


def settledFrame(clip, beat: int) -> np.ndarray:
    return clip["frames"][clip["truth"]["beats"][beat]["quiet_frame"] + 15]


def test_config_falls_back_to_its_defaults_and_ignores_what_it_does_not_know(tmp_path, capsys):
    assert Config.load(str(tmp_path / "missing.json")) == Config()

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"pixel_delta": 12, "not_a_threshold": 3}))
    cfg = Config.load(str(path))
    assert cfg.pixel_delta == 12
    assert cfg.settle_frames == Config().settle_frames
    assert "not_a_threshold" in capsys.readouterr().out


def test_config_round_trips(tmp_path):
    path = str(tmp_path / "c.json")
    Config(pixel_delta=31, motion_frac=0.01).save(path)
    assert Config.load(path).pixel_delta == 31


def test_both_detectors_find_the_object_where_the_answer_key_says_it_is(clip):
    cfg = Config()
    frame = settledFrame(clip, 0)
    regions = changedRegions(clip["frames"][0], frame, cfg)
    truthPx = clip["truth"]["frame"][clip["truth"]["beats"][0]["quiet_frame"] + 15]["objects"]["mug"]["px"]

    for det in (RefDiffDetector(clip["frames"][0], cfg),
                ChromaDetector(cfg, clip["H"], frame.shape)):
        boxes = det.detect(frame, regions)
        assert len(boxes) == 1, f"{type(det).__name__} saw {len(boxes)} objects"
        x0, y0, x1, y1 = boxes[0]
        #Refdiff swallows the cast shadow, so it runs a few pixels large and a few off
        #centre. Chroma rejects the shadow outright. Both are well inside a mug.
        assert ((x0 + x1) / 2, (y0 + y1) / 2) == pytest.approx(truthPx, abs=12.0)


def test_a_detection_outside_a_changed_region_is_dropped(clip):
    #Segmenting outside changed regions is what makes every object on the desk appear to
    #disappear on every settle.
    cfg = Config()
    frame = settledFrame(clip, 0)
    det = RefDiffDetector(clip["frames"][0], cfg)

    assert det.detect(frame, changedRegions(clip["frames"][0], frame, cfg))
    assert det.detect(frame, [(0, 0, 40, 40)]) == [] #A corner the mug is nowhere near


def test_an_empty_desk_is_not_a_detector_failure(clip):
    #Nothing on the desk is a fact about the desk. Only a broken detector raises.
    cfg = Config()
    bare = clip["frames"][0]
    assert RefDiffDetector(bare, cfg).detect(bare, [(300, 300, 400, 400)]) == []


def test_a_frame_wide_change_raises_rather_than_reporting_an_empty_desk(clip):
    #A moved camera reads as everything changing at once. An empty Observation here would
    #mark the whole desk LOST in a single tick, so the detector has to refuse.
    cfg = Config()
    frame = settledFrame(clip, 0)
    whole = [(0, 0, frame.shape[1], frame.shape[0])]

    with pytest.raises(DetectorError):
        RefDiffDetector(clip["frames"][0], cfg).detect(frame, whole)
    with pytest.raises(DetectorError):
        ChromaDetector(cfg, clip["H"], frame.shape).detect(frame, whole)


def test_a_reference_of_the_wrong_shape_raises(clip):
    cfg = Config()
    det = RefDiffDetector(cv2.resize(clip["frames"][0], (320, 180)), cfg)
    with pytest.raises(DetectorError):
        det.mask(clip["frames"][0])


def test_no_reference_at_all_raises():
    with pytest.raises(DetectorError):
        RefDiffDetector(None, Config())


def test_the_mat_mask_covers_the_mat_and_nothing_else(clip):
    frame = clip["frames"][0]
    mask, bounds = matMaskPx(clip["H"], frame.shape)

    assert mask[tuple(reversed([int(v) for v in mmToPx((300.0, 225.0))]))] == 1 #Middle
    assert mask[4, 4] == 0 #Frame corner, well off the mat
    assert bounds[0] == pytest.approx(round(mmToPx((0.0, 0.0))[0]), abs=2)


def test_chroma_ignores_everything_off_the_mat(clip):
    #Without the mat mask the whole neutral surround is foreground, and being a single
    #component touching every changed region it swallows the entire frame.
    cfg = Config()
    frame = settledFrame(clip, 0)
    fg = ChromaDetector(cfg, clip["H"], frame.shape).mask(frame)
    _mask, bounds = matMaskPx(clip["H"], frame.shape)

    assert fg[4, 4] == 0
    for box in components(fg, cfg.min_object_px):
        assert box[0] >= bounds[0] - 1 and box[2] <= bounds[2] + 1
        assert box[1] >= bounds[1] - 1 and box[3] <= bounds[3] + 1

    #The markers are foreground too -- they are printed paper, not mat. They never sit in
    #a changed region, so detect() never sees them, which is the thing that matters.
    boxes = ChromaDetector(cfg, clip["H"], frame.shape).detect(
        frame, changedRegions(clip["frames"][0], frame, cfg))
    assert len(boxes) == 1


def test_analyse_puts_the_object_in_millimetres_where_the_script_put_it(clip, tmp_path):
    cfg = Config()
    frame = settledFrame(clip, 0)
    det = ChromaDetector(cfg, clip["H"], frame.shape)
    obs = analyse(clip["frames"][0], frame, det, clip["H"], None, histEmbed, cfg,
                  ts=12.0, root=str(tmp_path))

    assert obs.ts == 12.0
    assert obs.agent_swept == [] and obs.agent_present is False #No tracker passed in
    assert len(obs.detections) == 1

    d = obs.detections[0]
    assert d.centroid == pytest.approx(MUG_MM, abs=6.0)
    assert d.size == pytest.approx(MUG_SIZE, abs=8.0)
    assert d.embedding.shape[0] > 1 and np.linalg.norm(d.embedding) == pytest.approx(1.0)
    assert obs.changed and any(r[0] < MUG_MM[0] < r[2] for r in obs.changed)


def test_analyse_writes_the_keyframe_and_crop_it_promised(clip, tmp_path):
    cfg = Config()
    frame = settledFrame(clip, 0)
    det = ChromaDetector(cfg, clip["H"], frame.shape)
    obs = analyse(clip["frames"][0], frame, det, clip["H"], None, histEmbed, cfg,
                  ts=12.0, root=str(tmp_path))

    #Every Event carries a frame_ref, and the scrubber shows the image it was believed
    #from. A path with no file behind it makes that pane an empty box.
    assert cv2.imread(obs.frame_ref) is not None
    assert cv2.imread(obs.detections[0].crop_path) is not None


def test_the_observation_encoder_matches_the_services_decoder(clip, tmp_path):
    from service import observationFrom
    cfg = Config()
    frame = settledFrame(clip, 0)
    det = ChromaDetector(cfg, clip["H"], frame.shape)
    obs = analyse(clip["frames"][0], frame, det, clip["H"], None, histEmbed, cfg,
                  ts=12.0, root=str(tmp_path))

    back = observationFrom(json.loads(json.dumps(observationJson(obs))))
    assert back.ts == obs.ts and back.frame_ref == obs.frame_ref
    assert back.changed == pytest.approx(obs.changed)
    assert back.detections[0].centroid == pytest.approx(obs.detections[0].centroid)
    assert back.detections[0].embedding == pytest.approx(obs.detections[0].embedding)


def test_histograms_tell_the_two_fixture_objects_apart():
    #Not re-identification, but if it cannot separate terracotta from blue it is no use
    #as the offline stand-in either.
    from synth import objectPatch
    mug = histEmbed(objectPatch("mug", 100, 100), (0, 0, 100, 100))
    mug2 = histEmbed(objectPatch("mug", 110, 105), (0, 0, 110, 105))
    box = histEmbed(objectPatch("box", 100, 100), (0, 0, 100, 100))

    assert float(mug @ mug2) > 0.9
    assert float(mug @ box) < 0.3


def test_region_intersection_is_exclusive_at_the_edge():
    assert intersects((0, 0, 10, 10), (5, 5, 15, 15))
    assert not intersects((0, 0, 10, 10), (10, 0, 20, 10)) #Touching is not overlapping
