# BUILD.md v2 — get to live testing

You've finished v1 steps 1–5. The rig requirement collapsed to *a webcam that doesn't move*,
so the plan is now: **get live in about 90 minutes, then fix what actually breaks** rather than
hardening against problems you're guessing at.

Read `spec.md` first. Sections marked **[BUILT]** already exist.

---

## Already built

`core.py` · `world.py` · `resolve.py` · `verify.py` · `fake.py` · `calib.py` ·
`identity.py` (matcher + Embedder) · `service.py` · dashboard · `synth.py` · tests

**Phases A and C are now written**, and the whole of A plus C1–C6 runs green against
`synth.mp4`: `pytest -q` covers steps 1–6 and 8 of the test table below end to end, on both
detectors. Step 7 (the palmed falsification) is covered at the world-model level by
`tests/test_script.py`, because the fixture's script has no palm-out beat.

Two things the fixture cannot answer, and a desk can:

- **Phase B is still open.** Nothing below has been run against real objects under a real
  webcam, which is the only place the thresholds in `config.json` get their real values.
- **The flicker guard is off.** `confirm_settles: 0` in `config.json`. Set it to `2` the
  first time phantom entities appear; it is built, tested and one number away.

---

# Phase A — minimum live loop (~90 min) — **written**

Three steps, then you're testing on real objects. Deliberately skips the agent tracker,
labelling, flicker guard and ArUco — all of those degrade gracefully or are cosmetic.

## A1. Calibration + capture — 25 min — built

> Read spec.md §3 and §4. Add to calib.py: openCamera (locks exposure, white balance and
> autofocus via CAP_PROP_*), clickCorners for the homography, level() for brightness matching,
> and captureReference which grabs an empty-desk frame. Write calibrate.py: opens the webcam,
> shows a live preview, `r` captures the empty-desk reference, `c` runs the corner clicks, `s`
> saves H, the reference frame and MAT_BOUNDS to calib.json + reference.png. Keep cv2 out of
> calib.py's existing pure functions.

**Test it:** clear your desk, run it, click four corners, save. Verify `pxToMm` on a known
point lands roughly where you'd expect.

## A2. RefDiff detector + settle loop — 45 min — built

The whole of perception, minus everything optional.

> Read spec.md §7. Write perceive.py with the Detector protocol, RefDiffDetector (level the
> stored reference, absdiff, threshold, connected components, intersect with changed regions),
> and the settle loop from §7 — frame diff, MOTION_FRAC, SETTLE_FRAMES quiet, then analyse().
> analyse() builds Detections by mapping boxes through H and embedding each crop with the
> Embedder from identity.py. A detector failure must raise DetectorError and cause the settle
> loop to skip that settle and keep the previous settled frame — never emit an empty
> Observation. All thresholds load from config.json. Stub the agent: agent_swept=[] and
> agent_present=False for now. Add a --video flag so it runs on a file as well as a camera —
> same code path, no live-only branch.

## A3. Wire it live — 20 min — built

> Write live.py: loads calib.json, opens the camera, runs perceive.run() with a sink that POSTs
> each Observation to the service. Add a `--local` flag that skips HTTP and calls settle()
> in-process for faster debugging. Print each returned Event to stdout as one line so you can
> see what it's deciding without the dashboard.

---

# ⟶ TEST NOW

Prop the webcam over your desk. Run `calibrate.py`, then `live.py`, then open the dashboard.

Work through these in order and write down what breaks:

| # | Action | Expect |
| --- | --- | --- |
| 1 | Place a mug on the empty desk | one entity appears, dot on the map |
| 2 | Move the mug across the desk | same entity, dot follows, `MOVED` |
| 3 | Place a box elsewhere | second entity |
| 4 | Cover the mug with the box | mug goes `HIDDEN`, nested inside the box's shape |
| 5 | Slide the box, mug still under | mug's dot travels with the box |
| 6 | Lift the box away | mug `REVEALED`, confidence back to 1.0 |
| 7 | Palm the mug out while covering, then lift | `BELIEF_FALSIFIED`, red row |
| 8 | Take the mug off the desk entirely | `LEFT_DESK`, cause `agent` (C2 is built, so H3 fires) |

Steps 4–7 are the demo. If those work, everything after this is polish.

---

# Phase B — fix what actually broke

Don't pre-empt these. Come back to this list *after* testing, and only do the ones you hit.

| Symptom | Fix |
| --- | --- |
| Phantom entities appearing and vanishing | **Flicker guard** (C1) — most likely first problem |
| Two objects merge into one detection | Move them apart for now; chroma or SAM later |
| Covering not detected | Lower `COVER_MIN` to 0.4, tiebreak on centroid distance |
| Mug not re-identified after reveal | Check DINO scores in the log; raise `MATCH_LO` or widen `PRIOR_SIGMA` |
| Nothing ever settles | Lower `SETTLE_FRAMES` to 8, raise `MOTION_FRAC` |
| Detections drift over minutes | Exposure didn't lock — confirm `level()` is being applied |
| Everything disappears at once | Camera moved, or the detector returned empty without raising |

---

# Phase C — hardening, in likely-need order — **written**

## C1. Flicker guard — 20 min — built, off by default

Almost certainly your first real problem once live.

> Wire World._provisional. A detection must appear at roughly the same position across two
> consecutive settles before settle() mints an entity from it; key on a coarse position bucket.
> Test: a detection present in one settle and gone the next mints nothing.

## C2. Agent tracker — 30 min — built

Unlocks H3, which turns "I lost track of it" into "you took it off the desk".

> Read spec.md §9. Write agent.py: AgentTracker with reset/update/swept_mm/present using the
> edge-touching blob heuristic and far-tip extraction. Wire into perceive.py's motion branch,
> replacing the stub.

## C3. Async labelling — 30 min — built

Turns every entity from "unknown" into something the voice layer can say.

> One vision call per new entity returning label and isContainer together, fired as a
> background task from World.mint, patching the entity when it returns. The entity is created
> and tracked immediately with label="unknown" — tracking never blocks on the network. Cache
> forever, never re-call. Push a state delta on /stream when the label lands.

## C4. Rewind — 25 min — built

> Snapshot the full world state after every settle into a list alongside events. Add GET
> /snapshots and a scrubber to the dashboard that indexes it directly — do not invert the event
> log. Scrubbing shows both the map at that moment and the frame_ref keyframe it was believed
> from.

## C5. ArUco + per-settle re-solve — 30 min — built

> Add findMarkers and solveHomography to calib.py per spec.md §4. Re-solve on every settle,
> keeping the previous H when fewer than 4 markers are visible. Makes a bumped camera free.

## C6. replay.py and tune.py — 40 min — built

> replay.py: run the settle loop over a recorded video, writing an annotated JPEG per settle
> (boxes, entity ids, match scores, status) and an events.jsonl.
> tune.py: live preview with trackbars for PIXEL_DELTA, MOTION_FRAC, MIN_OBJECT_PX and the
> refdiff threshold; `s` dumps to config.json.

**Record a video of your demo sequence as soon as A3 works.** From then on you can iterate on
`replay.py` without re-performing it every time.

---

# Phase D — demo

1. Objects that don't touch each other, and visually distinct (refdiff merges touching blobs).
2. Rehearse the eight test steps five times.
3. The two beats that land: slide the box with the mug under it, and the palmed-object
   falsification.
4. Write the submission.

---

## Cut order

ArUco → rewind → labelling → agent tracker. **Never cut steps 4–7 of the test table.**

## Non-negotiable

Stagger sleep with your teammate. Two people who both work straight through ship less than two
people who each get four hours.
