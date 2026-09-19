# Spatial Memory System — Technical Spec v2

A fixed overhead camera watches a desk. The system maintains a persistent model of every
object on it — what it is, where it is, and what happened to it — and answers spoken
questions about that model.

**v2 changes:** no coloured mat, no boom, no lamp. Detection is reference-frame subtraction,
not chroma. Omni labels and speaks; it does not detect. Sections marked **[BUILT]** describe
code that already exists — they're here for reference, not to be rewritten.

---

## 1. Core claim

> **Absence of detection is not evidence of absence.**

Conventional trackers delete a track when the object stops being visible. This system never
deletes an entity. A disappearance is an *event requiring an explanation*; the system commits
to the best available explanation, records it with its alternatives, and later tests it.

| Property | Meaning |
| --- | --- |
| **Persistence** | Entities have a *status*, not an existence flag. Confidence decays; the entity remains. |
| **Causal provenance** | Every position change has an attributed cause, a timestamp, and a keyframe receipt. |
| **Relative pose** | Position is stored relative to a parent. Move the parent, the children follow for free. |

### The division of labour

The project is best described as **giving Qwen 3.5 Omni spatial memory**. Omni perceives and
speaks; it has no persistence, and a context window — however large — is a buffer, not a store.
Nothing in a window decays, gets verified, or can be told that a belief formed four minutes ago
was just falsified.

| Layer | Role | Timescale |
| --- | --- | --- |
| Omni | perceives, classifies, hears, speaks | transient, in-context |
| World model | persists, relates, decays, verifies | durable, structured |
| DINO | binds a record to a physical object across time | per-observation |

---

## 2. Scope

One desk, one camera pointing down, one human.

> **From directly overhead, occlusion has exactly two causes.** Something is on top of the
> object, or the object left the desk. No "behind", no viewpoint-dependent depth ordering.
> The hypothesis space for a disappearance is small and closed.

This is the only reason the camera must be overhead. A homography handles tilt fine — that's
what it's for — but past roughly 15° off vertical, tall objects start leaning into their
neighbours and "on top of" blurs into "beside". Roughly overhead is enough; perpendicular is
not required.

**Out of scope:** SLAM or a moving camera; metric depth; multi-camera fusion; analysis at frame
rate; deformable objects; multi-person disambiguation; recovering identities across a cold start
with a non-empty desk.

---

## 3. Physical setup

Everything the system needs:

| Requirement | Why | Acceptable |
| --- | --- | --- |
| Camera roughly overhead | The two-causes argument | Laptop webcam, phone, USB cam |
| Camera does not move during a run | The **motion gate**, not the homography | Propped on books, taped to a shelf, monitor arm |
| Locked exposure, WB, autofocus | Reference subtraction and embeddings both drift otherwise | See below |
| An empty-desk reference frame | It *is* the detector | Clear desk, press a key |

No mat. No boom. No lamp. The coloured mat in v1 existed solely because chroma segmentation
needed a known background hue; reference subtraction doesn't care what the surface is.

### Locking the camera

```python
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual on most backends
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
```

These are backend-dependent and often silently ignored. If exposure won't lock, brightness
drift will slowly poison reference subtraction. Cheap insurance, three lines, do it regardless:

```python
def level(ref, now):
    """Match the reference's brightness to the current frame before diffing."""
    return np.clip(ref.astype(np.float32) * (now.mean() / max(ref.mean(), 1.0)), 0, 255).astype(np.uint8)
```

### Optional upgrade: ArUco markers

Four printed DICT_4X4_50 markers taped to the desk in a rectangle give you the homography *and*
the world bounds, re-solvable on every settle so a bumped camera costs nothing. Worth doing if
there's a printer. Without one, clicked corners (§4) are equivalent for a static camera.

---

## 4. Coordinates

One 3×3 homography maps pixels to millimetres on the desk plane. No depth model, no pose
estimation.

### Primary: clicked corners

Fastest path to a working system, needs no printer.

```python
# calib.py

def clickCorners(frame) -> np.ndarray:
    """Click 4 desk corners clockwise from top-left. Declares them MAT_BOUNDS."""
    pts = []
    cv2.imshow("calib", frame)
    cv2.setMouseCallback("calib", lambda ev, x, y, *_:
                         pts.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while len(pts) < 4:
        cv2.waitKey(50)
    x0, y0, x1, y1 = MAT_BOUNDS
    dst = np.float32([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    H, _ = cv2.findHomography(np.float32(pts), dst)
    return H
```

`MAT_BOUNDS` stays `(0, 0, 600, 450)` — you're declaring the clicked quad to be a 600×450 mm
region. It needn't be exactly true; it only needs to be consistent, because every threshold in
the system is expressed in those units.

### Upgrade: markers, re-solved per settle

```python
def solveHomography(frame) -> np.ndarray | None:
    seen = findMarkers(frame)                    # aruco, id -> centre px
    pts = [(seen[i], MAT_MM[i]) for i in MAT_MM if i in seen]
    if len(pts) < 4:
        return None                              # keep the previous H
    return cv2.findHomography(np.float32([p for p, _ in pts]),
                              np.float32([m for _, m in pts]))[0]
```

Called on every settle, this makes camera drift a non-issue.

### Footprints **[BUILT]**

`overlapFraction(a, b)` in `calib.py` — fraction of `a` that `b` covers, asymmetric on purpose.
A large box fully covering a small mug gives 1.0 one way and ~0.05 the other; you always want
*how much of the missing thing is covered*.

---

## 5. Data model **[BUILT]**

`core.py` holds `Entity`, `Event`, `Detection`, `Observation`, `ExampleBank`, and the enums
`Relation` (ON/IN/UNDER/HELD), `Status` (VISIBLE/HIDDEN/OFF_DESK/UNRESOLVED) and `EventKind`.

The contract that matters: **`Observation` is the only write into the world.** Everything
upstream of it is replaceable — that's what lets the detector change without touching any
reasoning code.

```python
@dataclass
class Observation:
    ts: float
    frame_ref: str
    detections: list[Detection]     # centroid mm, size mm, embedding, crop_path
    changed: list[Rect]             # mm regions differing from the last settle
    agent_swept: list[Rect]         # footprints the agent passed over (may be empty)
    agent_present: bool
```

---

## 6. The world **[BUILT]**

`world.py`: `World.absolute` walks the parent chain and sums poses; `footprintRect`;
`childrenOf`; `descendants`; `reparent` (the only mutation path, rejects cycles); `mint`;
`decayed`. Plus `surfaceUnder`, `placeOn`, `covers`, `snapshot`, `missingEntities`.

Moving a parent is one write — set `box.pose` and every descendant's absolute position follows
with no propagation code.

Confidence decays by status half-life (HIDDEN 3600s, OFF_DESK 1800s, UNRESOLVED 300s), with a
stabiliser: a hidden child whose occluder is still visible and unmoved floors at 0.8, because
that evidence genuinely hasn't degraded.

---

## 7. Perception

### Latency budget

This drives the whole design. Omni detection costs 3–4 s per call, which cannot sit in the
per-settle path — a five-action demo would spend twenty seconds waiting and settles would queue
behind each other on stale frames.

| Path | Cost | What runs |
| --- | --- | --- |
| Every frame | ~1 ms | frame diff, motion gate |
| Every settle | ~50 ms | refdiff → boxes → DINO → Observation → settle() |
| Per **new** entity | 3–4 s, **async** | Omni → label + isContainer, cached forever |
| Per query | 3–4 s | Omni voice loop |

An entity is minted, tracked and displayed immediately with `label="unknown"`; the label
patches in when the call returns. Tracking never waits on the network.

### The settle loop

The trigger is **motion stopping**, not hand detection. Frame difference over a threshold, then
quiet for ~500 ms. No model, no assumptions, trivially reliable.

```python
SETTLE_FRAMES = 15          # ~500 ms at 30 fps; drop to 8 if people work fast
MOTION_FRAC   = 0.002
PIXEL_DELTA   = 25

def run(cap, det, world, sink):
    prev, quiet, settled = None, 0, None
    while True:
        frame = grab(cap)
        g = gray(frame)
        if prev is not None:
            diff = cv2.absdiff(g, prev)
            if float((diff > PIXEL_DELTA).mean()) > MOTION_FRAC:
                quiet = 0
                agent.update(frame, diff, H)        # optional; feeds H3 only
            else:
                quiet += 1
                if quiet == SETTLE_FRAMES and settled is not None:
                    try:
                        sink(analyse(settled, frame, det, H, agent))
                        settled = frame
                    except DetectorError:
                        pass                        # keep the old settled frame
                    agent.reset()
                elif settled is None:
                    settled = frame
        prev = g
```

**A detector failure must abort the settle, never emit an empty `Observation`.** An empty one
makes `missingEntities` fire for everything in the changed region and marks the whole desk
`LOST` in a single tick.

### Analysis pass

Only changed regions are examined. Everything else is unchanged by definition.

```python
def analyse(prevSettled, now, det, H, agent) -> Observation:
    d = cv2.absdiff(gray(now), gray(prevSettled))
    changed = components(morph(d > PIXEL_DELTA), minArea=MIN_REGION_PX)

    dets = []
    for box in det.detect(now, regions=changed):
        (x0, y0), (x1, y1) = pxToMm(H, box.tl), pxToMm(H, box.br)
        dets.append(Detection(centroid=((x0 + x1) / 2, (y0 + y1) / 2),
                              size=(abs(x1 - x0), abs(y1 - y0)),
                              embedding=embedder(now, box),
                              crop_path=saveCrop(now, box)))
    return Observation(ts=time.time(), frame_ref=saveFrame(now), detections=dets,
                       changed=[rectPxToMm(H, r) for r in changed],
                       agent_swept=agent.swept_mm(), agent_present=agent.present)
```

**The subtlety that costs an hour if missed.** Most entities produce no detection on most
settles — normal, and it means nothing. An entity is *missing* only if its footprint intersects
a changed region and it still produced no match. `missingEntities` in `world.py` handles this.

### Detector protocol

```python
class Detector(Protocol):
    def detect(self, frame, regions) -> list[BoxPx]: ...
```

**`RefDiffDetector` — primary.** Subtract the stored empty-desk frame (brightness-levelled),
threshold, connected components, intersected with changed regions. ~2 ms, entirely local, works
on any surface. Weakness: touching objects merge into one blob, and it drifts if lighting or the
camera changes.

**`ChromaDetector` — optional.** Hue-distance mask against a coloured mat, ignoring the value
channel so shadows are rejected for free. Only worth it if you end up with a green or blue
surface. Strictly better than refdiff when available.

**`OmniDetector` — not in the settle path.** Too slow. Keep the class if you want it for a
one-off cold-start scan of a non-empty desk; it raises on timeout like any other detector.

---

## 8. Identity **[BUILT, matcher]**

DINO does **instance re-identification**, never location. Each box answers two independent
questions:

```
box px ──┬── homography ───────────────► WHERE  (mm on the plane)
         └── crop → DINO → cosine ─────► WHICH  (which known entity)
```

`identity.py` has `match()`: cosine over exemplar banks, plus a **spatial prior**
(`0.10 · exp(−d/150mm)`) which is the single highest-value line in the file. Two identical pens
are indistinguishable by appearance; the one that was here 200 ms ago is overwhelmingly likely
to be the one here now.

`MATCH_LO = 0.55` below which it's a new entity, `MARGIN_MIN = 0.05` the required gap to the
runner-up. Below that gap the match is `ambiguous` and the system says so rather than guessing.

The `Embedder` (torch, imported lazily) crops, square-pads — never stretches — resizes to 224
and returns the normalised CLS token.

What identity buys that a label cannot: re-acquiring a specific object after minutes of
occlusion when several share a label; separating two objects the VLM describes identically; and
noticing a **substitution** — lift the box, find a *different* mug, and get a
`BELIEF_FALSIFIED` where label-matching would have silently agreed.

---

## 9. The agent — optional

The agent is an ordinary entity whose children are `HELD`. Pick up and put down are reparents;
the containment graph does the work.

The detector is heuristic: from overhead over a bounded surface, an arm **must** enter from the
frame border, so the largest edge-touching motion blob is the agent, and the point furthest from
the entering edge is the hand rather than the elbow.

**This feeds H3 and nothing else.** H1, H2 and H4 use no agent evidence at all. With the tracker
absent or broken, the system degrades from *"you carried it off"* to *"it left the desk, cause
unknown"* — a weaker answer, not a wrong one, and nothing stops updating. A heuristic feeding
one optional hypothesis is a normal engineering choice; a heuristic gating the update loop would
not be.

Upgrade if it misbehaves: MediaPipe Hands, ~5 ms CPU, and grasp aperture separates a full hand
from an empty one.

---

## 10. Disappearance resolution **[BUILT]**

`resolve.py`. Entity was `VISIBLE`, its footprint intersects a changed region, it produced no
match. Evaluate in order, take the first that fires, record the rest as `alternatives`.

| | Hypothesis | Condition | Result |
| --- | --- | --- | --- |
| H1 | Moved | matched a detection elsewhere | `MOVED`, 0.95 |
| H2 | Covered | a detection overlaps its old footprint ≥ `COVER_MIN` | `HIDDEN`, `IN`/`UNDER` |
| H3 | Carried | agent swept its footprint, matched nowhere | `PICKED_UP` / `LEFT_DESK` |
| H4 | Lost | nothing fits | `UNRESOLVED`, 0.40 |

`missing.sort(key=lambda e: e.id not in matches)` in `settle()` is load-bearing: resolving
matched entities first is what lets H2 see the occluder at its *new* position.

If detector boxes run loose, drop `COVER_MIN` from 0.5 toward 0.4 and use centroid distance as
a tiebreak.

---

## 11. Verification and falsified beliefs **[BUILT]**

`verify.py`. On reveal, any match is a reveal — the note distinguishes "confirmed where I
thought it was" from "found it, though not where I expected". Absence alone is falsification.

Guard: a still-covering occluder means the belief is untested, not wrong.

**The IN/UNDER split**, which must be present:

```python
if occluder.status in (Status.OFF_DESK, Status.HIDDEN) and child.relation == Relation.IN:
    child.status = occluder.status      # it's in there, wherever there is now
    continue
```

Keys `IN` a tin travel with the tin. A mug `UNDER` a lifted box stays on the desk and *is*
falsified if it didn't turn up. Without this, the container demo — put keys in a tin, carry the
tin away, ask where the keys are — gives the wrong answer.

**Announce falsification unprompted.** *"I was wrong — the mug isn't under the box. Last time I
actually saw it was 03:41."* A system that reports its own errors reads as far more
sophisticated than one that is silently right, and it converts the worst failure mode into the
best demo beat. Palm the object out while covering it, then lift the box.

---

## 12. Query layer

```
POST /observation        GET /state          GET /events?since=
WS   /stream             GET /snapshots      # for rewind
```

Four tools, exposed as HTTP and MCP: `where_is`, `whats_in`, `who_moved`, `history`. Each
returns compact JSON, never prose. **Phrasing is the model's job; grounding is the world
model's.** That line is what stops the system hallucinating positions.

Confidence is rendered, not hidden:

| Confidence | Phrasing |
| --- | --- |
| > 0.9 | "It's under the box." |
| 0.7–0.9 | "It should be under the box — I haven't seen it since 03:41, but the box hasn't moved." |
| 0.4–0.7 | "Probably under the box, though I'm not certain." |
| < 0.4 | "I've lost track of it. Last confirmed on the desk at 03:41." |

### Omni

Two jobs, both outside the settle path:

1. **Labelling** — one async call per new entity returning `label` and `isContainer`.
2. **Voice** — Thinker takes streaming audio and video, Talker speaks, tools registered. Feed it
   the camera stream as well as audio so referent resolution works: *"where's the one I just
   had"*, or pointing and saying *"what about this one"*.

System prompt, blunt:

```
Every claim you make about where something is MUST come from a tool result. Never
invent a location, never infer one from the camera image alone, and never smooth over
a low confidence score. If a tool returns found=false, say you don't know.
```

---

## 13. Dashboard

Two panes plus a scrubber.

**Left — desk map.** Entities at absolute positions. Solid = `VISIBLE`; hollow dashed and
nested inside the parent's shape = `HIDDEN`; faded with a question mark = `UNRESOLVED`; greyed
into a margin strip = `OFF_DESK`. **Opacity tracks confidence**, so a decaying belief is
something you can watch fade.

**Right — event timeline.** Newest first: timestamp, label, kind, cause, confidence.
`BELIEF_FALSIFIED` in red.

**Rewind.** Snapshot the entire world state after every settle into a list alongside `events` —
a few hundred bytes each at this scale. The scrubber indexes that list. **Do not invert the
event log**; it's tempting and it's a trap.

Because every `Event` carries `frame_ref`, scrubbing shows both the map state *and* the keyframe
it was believed from. "Here's what it thought at 3:41, and here's the image it thought it from"
beats a timeline alone.

---

## 14. Failure modes

| Failure | Symptom | Mitigation |
| --- | --- | --- |
| **Exposure drift** | refdiff slowly fills with noise; embeddings shift | Lock exposure; brightness-level the reference before diffing |
| **Camera moved** | Everything reads as changed at once | ArUco re-solve per settle, or recalibrate hotkey |
| **Empty Observation on detector failure** | Whole desk marked `LOST` in one tick | Detector raises; settle loop skips and keeps the old settled frame |
| **Segmenting outside changed regions** | Every object "disappears" each settle | `missingEntities` gating |
| **Mask flicker** | One object becomes two entities | Two-settle confirmation before minting |
| **Touching objects** | refdiff merges them into one blob | Move them apart for the demo, or switch to chroma/SAM |
| **Identical objects** | Two pens swap identity | Spatial prior; on ambiguity keep both and say so |
| **Loose boxes vs `COVER_MIN`** | Covering not detected | Lower to 0.4, tiebreak on centroid distance |
| **Label arrives late** | `IN` misread as `UNDER` for ~4 s | Default `isContainer=False`; cosmetic only |
| **Omni invents a location** | Confident answer with no grounding | Constraining prompt; log every tool call beside its answer |

### Limitations worth stating plainly

- Objects moved while a body occludes the whole surface are unresolvable.
- Stacking beyond two or three levels is untested; footprint overlap gets noisier with depth.
- One agent only, so "who moved it" means "a person moved it".
- A cold start with a non-empty desk mints new entities for everything; prior identities are not
  recoverable.
