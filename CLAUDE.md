# Spatial memory system

Read spec.md before any task. It is the source of truth for types,
thresholds, and module boundaries.

## Layout
Flat, at repo root. No packages, no src/ dir.
  core.py      types and enums — FROZEN, see below
  world.py     World: geometry, structure, reparent, decay, flicker guard
  resolve.py   H1-H4 hypotheses + resolve()
  verify.py    reveal path, falsified beliefs
  calib.py     mat geometry + homography, camera, reference frame, ArUco
  perceive.py  detectors, settle loop, analysis pass, Config
  identity.py  embedder, matcher, exemplar bank
  agent.py     AgentTracker
  omni.py      async labelling (label + isContainer), one call per new entity
  voice.py     wake phrase -> world state as context -> Omni -> spoken answer
  fake.py      scripted observation generator
  synth.py     synthetic fixture video renderer
  start.py     the access point: calibrate, then service + camera + voice
  calibrate.py one-off: clicked corners + empty-desk reference -> calib.json
  live.py      camera -> perceive.run -> POST /observation (or --local)
  replay.py    run the pipeline over a video file, annotated JPEG per settle
  tune.py      trackbars for the thresholds, s dumps config.json
  service.py   FastAPI: /observation /state /events /stream /snapshots /snaps
  ui.html      dashboard: desk map, event timeline, rewind scrubber
  config.json  thresholds. calib.json + reference.png are per-machine, untracked
  tests/

## Hard rules
- core.py is a frozen contract. Never change Entity, Event, Detection,
  Observation, or the enums unless I explicitly say so.
- MATCH THE STYLE OF core.py EXACTLY: naming, spacing, docstring form,
  type-hint style, comment placement, dataclass conventions. Read it
  before writing any new file and conform to it. Do not impose your
  own conventions.
- Perception writes to the world ONLY via Observation.
- No new dependencies without asking. Current: opencv-python, numpy,
  torch, fastapi, uvicorn, websockets, pytest.
- Prefer plain functions and dataclasses. No abstraction layers.
- After every change run `pytest -q` and paste the real output.
  Never claim something works without running it.