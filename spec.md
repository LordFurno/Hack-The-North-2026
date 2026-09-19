# Spatial Memory System — Technical Spec

A fixed overhead camera watches a desk. The system maintains a persistent model of every
object on it — what it is, where it is, and what happened to it — and answers spoken
questions about that model.

---

## 1. Core claim

> **Absence of detection is not evidence of absence.**

Conventional trackers delete a track when the object stops being visible. This system never
deletes an entity. A disappearance is an *event requiring an explanation*; the system commits
to the best available explanation, records it with its alternatives, and later tests it.

Three properties follow, and together they are the system:

| Property | Meaning |
| --- | --- |
| **Persistence** | Entities have a *status*, not an existence flag. Confidence decays; the entity remains. |
| **Causal provenance** | Every position change has an attributed cause, a timestamp, and a keyframe receipt. |
| **Relative pose** | Position is stored relative to a parent. Move the parent, the children follow for free. |

End to end:

```
camera ─► motion gate ─► settle ─► segment changed regions ─► embed ─► match
                                                                        │
                                          ┌─────────────────────────────┘
                                          ▼
                             resolve disappearances ─► world model ─► events
                                                            │
                                    voice query ─► tools ────┘─► spoken answer
```

---

## 2. Scope

The demo world is one desk under one camera pointing straight down. This restriction buys a
real simplification, and every design decision below leans on it:

> **From directly overhead, occlusion has exactly two causes.** Something is on top of the
> object, or the object left the desk. There is no "behind", no viewpoint-dependent depth
> ordering, no oblique partial occlusion. The hypothesis space for a disappearance is small
> and closed.

**In scope:** rigid objects placed, moved, stacked, covered, contained and removed by hand;
containment to arbitrary nesting depth; spoken queries about location, history and causation;
explicit uncertainty in every answer.

**Out of scope** — each is a day of work and none is the interesting part: SLAM or a moving
camera; metric depth (the desk is a plane, a homography suffices); multi-camera fusion;
analysis at frame rate; deformable objects and liquids; multi-person disambiguation; entity
identity across a cold start with a non-empty desk.

---

## 3. Physical setup

### Bill of materials

| Item | Purpose | Notes |
| --- | --- | --- |
| USB webcam | The only sensor | 1080p ideal, 720p fine |
| Overhead boom | Camera looking straight down | Ring-light stand with overhead arm; tripod with horizontal extension arm; clamp + gooseneck |
| USB extension cable | Reaching the boom | The stock cable will not reach. Most commonly forgotten item |
| Matte mat, saturated colour | Defines the world; makes segmentation trivial | **See note below** — not black |
| One diffuse lamp | Softens shadows | Off-axis, so a hand casts no hard edge onto objects |
| 4 printed ArUco markers | Homography | Taped at mat corners, inside frame |

**On the mat colour.** Use a **matte, saturated mid-value colour** — green or blue cloth, or
poster board. Not black, and not white.

The reason is shadow rejection. A shadow on the mat keeps the mat's *hue* and loses *value*.
If the mat is chromatic, you segment by hue distance and shadows are rejected for free,
regardless of how dark they get:

```python
def foreground_mask(frame_bgr, mat_hue, hue_tol=18, sat_min=60):
    """True where the pixel is NOT mat. Shadows keep mat hue -> rejected."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, _v = cv2.split(hsv)                      # note: value is unused
    dh = np.minimum(np.abs(h.astype(int) - mat_hue),
                    180 - np.abs(h.astype(int) - mat_hue))
    is_mat = (dh < hue_tol) & (s > sat_min)
    return (~is_mat).astype(np.uint8)
```

Deliberately not using `v` is the whole trick. A black mat would make shadows invisible too,
but it would also swallow every dark object; a chromatic mat keeps dark objects visible while
still discarding their shadows.

### Geometry

Horizontal FOV is typically 65–70°, so visible width ≈ `2h·tan(FOV/2)`:

| Height | Field of view |
| --- | --- |
| 50 cm | ~65 cm |
| 60 cm | ~80 cm |
| 70 cm | ~95 cm |

Mount at **60–70 cm**. Size the mat to sit comfortably inside the frame with all four markers
visible. Objects should occupy ≥ 40×40 px; at 1080p over a 60 cm mat that is about 1.2 cm of
real object, fine for mugs, boxes, keys and phones.

### Camera settings — do this before writing any code

```bash
v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1            # 1 = manual
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_time_absolute=250
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_automatic=0
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature=4600
v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=0
```

macOS: use OpenCV's `CAP_PROP_AUTO_EXPOSURE` / `CAP_PROP_AUTO_WB` or the vendor utility.

Auto-exposure shifts the instant a hand enters the frame. That changes the apparent colour and
brightness of *every* object, which shifts their embeddings, which breaks instance matching —
and it presents as a model problem. This is the single most expensive thing to discover late.

---

## 4. Coordinates

The entire spatial stack is one 3×3 homography. No depth model, no pose estimation, no SLAM.

```python
# perception/calib.py
import cv2, numpy as np, json

MAT_MM = {                              # marker id -> (x, y) mm, measured once with a ruler
    0: (0.0,   0.0),
    1: (600.0, 0.0),
    2: (600.0, 450.0),
    3: (0.0,   450.0),
}
MAT_BOUNDS = (0.0, 0.0, 600.0, 450.0)   # x0, y0, x1, y1

_DET = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def find_markers(frame) -> dict[int, np.ndarray]:
    corners, ids, _ = _DET.detectMarkers(frame)
    if ids is None:
        return {}
    return {int(i): c[0].mean(axis=0) for c, i in zip(corners, ids.flatten())}


def solve_homography(frame) -> np.ndarray | None:
    seen = find_markers(frame)
    pts = [(seen[i], MAT_MM[i]) for i in MAT_MM if i in seen]
    if len(pts) < 4:
        return None
    src = np.array([p for p, _ in pts], dtype=np.float32)
    dst = np.array([m for _, m in pts], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def px_to_mm(H: np.ndarray, pt) -> tuple[float, float]:
    q = H @ np.array([pt[0], pt[1], 1.0])
    return (float(q[0] / q[2]), float(q[1] / q[2]))


def mm_to_px(H: np.ndarray, pt) -> tuple[float, float]:
    q = np.linalg.inv(H) @ np.array([pt[0], pt[1], 1.0])
    return (float(q[0] / q[2]), float(q[1] / q[2]))


def on_mat(pt_mm) -> bool:
    x0, y0, x1, y1 = MAT_BOUNDS
    return x0 <= pt_mm[0] <= x1 and y0 <= pt_mm[1] <= y1
```

**Drift detection.** Persist `H` to disk and bind a recalibrate hotkey, because the boom will
get bumped. Detect it automatically: if every marker centre shifts by a similar vector between
two settles, re-solve rather than reporting every object as moved.

```python
def boom_moved(prev: dict[int, np.ndarray], now: dict[int, np.ndarray]) -> bool:
    common = set(prev) & set(now)
    if len(common) < 3:
        return False
    deltas = np.array([now[i] - prev[i] for i in common])
    return bool(np.linalg.norm(deltas.mean(axis=0)) > 4.0      # coherent shift
                and deltas.std(axis=0).max() < 3.0)            # not object motion
```

### Footprints, not points

Store an axis-aligned bounding box in mm alongside the centroid. Occlusion reasoning needs
*area overlap*, not point containment: a box covering a mug overlaps the mug's footprint even
when their centroids are 4 cm apart.

```python
Rect = tuple[float, float, float, float]    # x0, y0, x1, y1 in mm

def overlap_fraction(a: Rect, b: Rect) -> float:
    """Fraction of `a` that `b` covers. Asymmetric on purpose."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    return (ix * iy) / area_a if area_a > 0 else 0.0
```

Asymmetry matters: a large box fully covering a small mug gives `overlap_fraction(mug, box) =
1.0` but `overlap_fraction(box, mug) ≈ 0.05`. You always want "how much of the *missing* thing
is covered".

---

## 5. Data model

```python
# worldmodel/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np, time, uuid

DESK = "DESK"


class Relation(str, Enum):
    ON    = "ON"       # resting on a surface or another object
    IN    = "IN"       # inside a container
    UNDER = "UNDER"    # covered by something that is not a container
    HELD  = "HELD"     # in an agent's grasp


class Status(str, Enum):
    VISIBLE    = "VISIBLE"
    HIDDEN     = "HIDDEN"        # believed present, occluded, with a named occluder
    OFF_DESK   = "OFF_DESK"
    UNRESOLVED = "UNRESOLVED"    # gone, no explanation


class EventKind(str, Enum):
    APPEARED         = "APPEARED"
    MOVED            = "MOVED"
    COVERED          = "COVERED"
    REVEALED         = "REVEALED"
    PICKED_UP        = "PICKED_UP"
    PUT_DOWN         = "PUT_DOWN"
    LEFT_DESK        = "LEFT_DESK"
    LOST             = "LOST"
    CONFIRMED        = "CONFIRMED"
    BELIEF_FALSIFIED = "BELIEF_FALSIFIED"
```

### Exemplar bank

Instance identity is a set of embeddings per entity, not a single vector. Kept as its own
small type so the update policy lives in one place.

```python
@dataclass
class ExemplarBank:
    vectors: list[np.ndarray] = field(default_factory=list)
    cap: int = 8

    ADD_MIN_MATCH = 0.85    # only learn from confident matches
    ADD_MAX_SIM   = 0.95    # only learn if it adds information

    def best(self, q: np.ndarray) -> float:
        return max((float(q @ v) for v in self.vectors), default=0.0)

    def offer(self, v: np.ndarray, match_score: float) -> None:
        """Conditionally absorb a new view of this object."""
        if match_score < self.ADD_MIN_MATCH:
            return                                   # not confident it's the same thing
        if self.best(v) > self.ADD_MAX_SIM:
            return                                   # redundant with what we have
        self.vectors.append(v)
        if len(self.vectors) > self.cap:
            self._evict_most_redundant()

    def _evict_most_redundant(self) -> None:
        sims = [sum(float(a @ b) for b in self.vectors if b is not a)
                for a in self.vectors]
        self.vectors.pop(int(np.argmax(sims)))
```

Both thresholds matter. Learning from unconfident matches is how an entity's bank slowly
becomes a different object; learning redundant views is how the bank fills with eight copies of
one pose and then fails the moment the object is rotated.

### Entity

```python
@dataclass
class Entity:
    id: str
    label: str = "unknown object"
    is_container: bool = False
    is_agent: bool = False

    bank: ExemplarBank = field(default_factory=ExemplarBank)

    parent: str | None = DESK
    relation: Relation = Relation.ON
    pose: tuple[float, float] = (0.0, 0.0)      # mm, RELATIVE TO PARENT
    size: tuple[float, float] = (0.0, 0.0)      # w, h in mm

    status: Status = Status.VISIBLE
    confidence: float = 1.0
    last_seen: float = 0.0
    last_confirmed: float = 0.0

    @staticmethod
    def new(**kw) -> "Entity":
        now = time.time()
        return Entity(id=uuid.uuid4().hex[:8], last_seen=now, last_confirmed=now, **kw)
```

### Event

Append-only. This is the system's memory of causation.

```python
@dataclass
class Event:
    ts: float
    entity: str
    kind: EventKind
    cause: str                                   # "agent" | "covered_by:<id>" | "unexplained"
    confidence: float
    from_state: dict | None = None               # {parent, relation, pose, status}
    to_state:   dict | None = None
    alternatives: list[dict] = field(default_factory=list)
    frame_ref: str = ""                          # keyframe that produced this
    note: str = ""                               # human-readable, for the timeline
```

`alternatives` costs nothing to record and is what lets the system explain itself rather than
merely assert: *"it's under the box — I also considered you'd carried it off, but your hand
left the frame empty."*

`frame_ref` means every claim has a visual receipt. Showing that keyframe beside the answer is
a strong demo beat.

### The observation contract

This is the interface between the two halves of the team. Agree it in the first thirty minutes
and do not change it after.

```python
@dataclass
class Detection:
    """One segmented region in a settled frame."""
    centroid: tuple[float, float]       # mm
    size: tuple[float, float]           # w, h in mm
    embedding: np.ndarray               # (D,), L2-normalised
    crop_path: str


@dataclass
class Observation:
    """Everything perception learned from one settle. The only write to the world."""
    ts: float
    frame_ref: str
    detections: list[Detection]
    changed: list[Rect]                         # mm regions that differ from last settle
    agent_swept: list[Rect]                     # footprints the agent passed over
    agent_present: bool                         # agent still in frame at settle time
```

---

## 6. The world

```python
# worldmodel/world.py

class World:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.events: list[Event] = []
        self._provisional: dict[int, tuple[Detection, int]] = {}   # flicker guard

    # ---- geometry -------------------------------------------------------

    def absolute(self, e: Entity | str) -> tuple[float, float]:
        e = self.entities[e] if isinstance(e, str) else e
        x, y = e.pose
        p = e.parent
        seen = {e.id}
        while p and p != DESK:
            if p in seen:                       # defensive; reparent() prevents this
                break
            seen.add(p)
            par = self.entities[p]
            x += par.pose[0]
            y += par.pose[1]
            p = par.parent
        return (x, y)

    def footprint(self, e: Entity | str) -> Rect:
        e = self.entities[e] if isinstance(e, str) else e
        cx, cy = self.absolute(e)
        w, h = e.size
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    # ---- structure ------------------------------------------------------

    def children_of(self, pid: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.parent == pid]

    def descendants(self, pid: str) -> list[Entity]:
        out, stack = [], [pid]
        while stack:
            for c in self.children_of(stack.pop()):
                out.append(c)
                stack.append(c.id)
        return out

    def reparent(self, e: Entity, parent: str | None,
                 relation: Relation, abs_pos: tuple[float, float]) -> None:
        """Set parent and store pose relative to it. Rejects cycles."""
        if parent and parent != DESK:
            if e.id == parent or any(d.id == parent for d in self.descendants(e.id)):
                raise ValueError(f"reparent {e.id} under {parent} would cycle")
            px, py = self.absolute(parent)
        else:
            px, py = (0.0, 0.0)
        e.parent = parent
        e.relation = relation
        e.pose = (abs_pos[0] - px, abs_pos[1] - py)
```

**Moving a parent is a single write.** Set `box.pose`; every descendant's absolute position
updates automatically. No propagation pass, no bookkeeping to get wrong — and the demo case
"slide the box across the desk, then ask where the mug is" works without a line of code
written for it.

### Confidence decay

```python
HALF_LIFE = {                       # seconds
    Status.HIDDEN:     3600.0,      # well-founded: the occluder is right there
    Status.OFF_DESK:   1800.0,
    Status.UNRESOLVED:  300.0,      # nothing is keeping this belief true
}

def decayed(self, e: Entity, now: float) -> float:
    if e.status == Status.VISIBLE:
        return 1.0
    dt = now - e.last_confirmed
    c = e.confidence * 0.5 ** (dt / HALF_LIFE[e.status])

    # Stabiliser: if the occluder is still visible and hasn't moved, the
    # evidence genuinely hasn't degraded, so don't pretend it has.
    if e.status == Status.HIDDEN and e.parent in self.entities:
        occ = self.entities[e.parent]
        if occ.status == Status.VISIBLE and occ.last_confirmed >= e.last_confirmed:
            c = max(c, 0.8)
    return c
```

The stabiliser is the difference between a decay model that is merely present and one that is
*right*. A mug under a box that nobody has touched should not become uncertain just because
time passed.

---

## 7. Perception

### The settle loop

Tiered by cost. This is what lets it run on a laptop, and it is also what makes it correct —
analysing mid-motion yields blurred masks and embeddings of half-occluded objects, which is
worse than not analysing.

| Tier | Rate | Work |
| --- | --- | --- |
| 0 | 30 fps | Frame difference → motion mask. Nothing else |
| 1 | on settle | Analysis pass |
| 2 | new entity only | One VLM call for a label. Cached forever |
| 3 | on query | Voice → tool → world → speech |

```python
# perception/camera.py

SETTLE_FRAMES = 15          # ~500 ms at 30 fps
MOTION_FRAC   = 0.002       # fraction of pixels differing
PIXEL_DELTA   = 25


def run(cap, H, sink, agent_tracker):
    prev = None
    quiet = 0
    settled = to_gray(grab(cap))
    settled_bgr = None
    agent_tracker.reset()

    while True:
        frame = grab(cap)
        g = to_gray(frame)

        if prev is not None:
            diff = cv2.absdiff(g, prev)
            moving = float((diff > PIXEL_DELTA).mean()) > MOTION_FRAC

            if moving:
                quiet = 0
                agent_tracker.update(frame, diff, H)     # accumulate swept footprints
            else:
                quiet += 1
                if quiet == SETTLE_FRAMES:
                    obs = analyse(settled, settled_bgr, g, frame, H, agent_tracker)
                    sink(obs)
                    settled, settled_bgr = g, frame
                    agent_tracker.reset()
        prev = g
```

The system is deliberately blind while a hand is in frame. It reasons about what changed
between two stable states, which is exactly the information the world model needs.

### The analysis pass

```python
# perception/analyse.py

def analyse(prev_gray, prev_bgr, now_gray, now_bgr, H, agent) -> Observation:
    # 1. Where did anything change between the two settled frames?
    d = cv2.absdiff(now_gray, prev_gray)
    changed_px = cv2.morphologyEx((d > PIXEL_DELTA).astype(np.uint8),
                                  cv2.MORPH_CLOSE, KERNEL_9)
    regions_px = components(changed_px, min_area=MIN_REGION_PX)

    # 2. Segment ONLY inside those regions. Everything else is unchanged by
    #    definition and needs no work.
    fg = foreground_mask(now_bgr, MAT_HUE)
    detections = []
    for r in regions_px:
        for comp in components(fg[r.slice], min_area=MIN_OBJECT_PX):
            box_px = comp.bbox_in(r)
            emb    = embed(now_bgr, box_px)
            (x0, y0), (x1, y1) = px_to_mm(H, box_px.tl), px_to_mm(H, box_px.br)
            detections.append(Detection(
                centroid=((x0 + x1) / 2, (y0 + y1) / 2),
                size=(abs(x1 - x0), abs(y1 - y0)),
                embedding=emb,
                crop_path=save_crop(now_bgr, box_px),
            ))

    return Observation(
        ts=time.time(),
        frame_ref=save_frame(now_bgr),
        detections=detections,
        changed=[rect_px_to_mm(H, r) for r in regions_px],
        agent_swept=agent.swept_mm(),
        agent_present=agent.present,
    )
```

**The subtlety that will otherwise cost you an hour.** Because you segment only inside changed
regions, most entities produce no detection on any given settle — that is normal and means
nothing. An entity counts as *missing* only if its footprint intersects a changed region and it
still produced no match. Getting this wrong makes every object on the desk disappear on every
frame.

```python
def missing_entities(world, obs, matched: set[str]) -> list[Entity]:
    out = []
    for e in world.entities.values():
        if e.id in matched or e.status != Status.VISIBLE:
            continue
        fp = world.footprint(e)
        if any(overlap_fraction(fp, region) > 0.2 for region in obs.changed):
            out.append(e)          # it was where something changed, and it's not there now
    return out
```

### Segmentation: which path

Two options; pick by what is working at hour three.

**Chroma + connected components** (recommended to start): the `foreground_mask` above plus
`cv2.connectedComponentsWithStats`. Zero model load, ~2 ms, and the chromatic mat makes it
robust. Its weakness is touching objects merging into one blob.

**SAM 2** (upgrade if merging bites): prompt with each changed region's bounding box, take the
returned masks. Handles touching objects properly, costs ~80 ms on a laptop GPU and a model
download. Swap it in behind the same `components()` interface so nothing downstream changes.

---

## 8. Instance identity

DINOv3's job is **instance re-identification**, not classification. The question is never "is
this a mug" — the VLM answers that once, at birth. The question is "is this *the same mug*",
and self-supervised dense features are unusually good at exactly that.

```python
# perception/identity.py

class Embedder:
    def __init__(self, name="dinov3_vits16", device="cuda"):
        self.model = torch.hub.load("facebookresearch/dinov3", name).to(device).eval()
        self.device = device
        self.tf = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def __call__(self, frame_bgr, bbox_px) -> np.ndarray:
        crop = tight_crop(frame_bgr, bbox_px, pad=0.10)
        crop = square_pad(crop)                      # preserve aspect ratio, do not stretch
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        x = self.tf(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))[None].to(self.device)
        f = self.model(x)[0]
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()
```

Square-padding rather than stretching matters more than it looks. A stretched crop of a rotated
object embeds differently from the same object unrotated, and that is a false negative you will
spend an hour chasing.

### Matching

```python
MATCH_HI     = 0.75     # accept
MATCH_LO     = 0.55     # below this: it is a new entity
MARGIN_MIN   = 0.05     # winner must beat runner-up by this
PRIOR_WEIGHT = 0.10
PRIOR_SIGMA  = 150.0    # mm


@dataclass
class MatchResult:
    entity: Entity | None
    score: float
    ambiguous: bool = False
    runner_up: Entity | None = None


def match(world, det: Detection) -> MatchResult:
    scored = []
    for e in world.entities.values():
        if e.status == Status.OFF_DESK or e.is_agent:
            continue
        s = e.bank.best(det.embedding)
        d = dist(det.centroid, world.absolute(e))
        s += PRIOR_WEIGHT * math.exp(-d / PRIOR_SIGMA)      # spatial prior
        scored.append((s, e))

    if not scored:
        return MatchResult(None, 0.0)
    scored.sort(key=lambda t: -t[0])
    (s1, e1) = scored[0]
    (s2, e2) = scored[1] if len(scored) > 1 else (0.0, None)

    if s1 < MATCH_LO:
        return MatchResult(None, s1)
    if s1 - s2 < MARGIN_MIN:
        return MatchResult(e1, s1, ambiguous=True, runner_up=e2)
    return MatchResult(e1, s1)
```

The **spatial prior** is the highest-value line in this file. Two identical black pens are
near-indistinguishable by appearance, but the one that was at this spot 200 ms ago is
overwhelmingly likely to be the one here now. Appearance alone coin-flips; appearance plus
proximity almost never does.

**On `ambiguous`:** do not guess silently. Keep both candidates alive, leave the entity
`UNRESOLVED`, and let the query layer say so — *"one of your two black pens; I can't tell them
apart, but the one on the left hasn't moved since you put it there."* Admitting this reads as
more sophisticated than guessing right by luck.

### Flicker guard

A mask must persist across two consecutive settles before it mints an entity. This removes most
spurious entities on its own.

```python
def confirm_new(world, det, key: int, now: float) -> Entity | None:
    prior = world._provisional.pop(key, None)
    if prior is None:
        world._provisional[key] = (det, 1)
        return None                              # wait one more settle
    return world.mint(det, now)
```

---

## 9. The agent

> **Design note.** An earlier draft treated hand detection as a side channel feeding one
> hypothesis. That was the hack — not the heuristic itself, but the special-casing. The agent
> is modelled here as an ordinary entity whose children are `HELD`. Picking up is a reparent;
> putting down is a reparent; the containment graph does all the work. Only the *detector*
> stays heuristic, which is an ordinary engineering position.

### Why the heuristic is defensible

From overhead, over a bounded mat, an arm **must** enter from the frame border. That is a
geometric fact of the rig, not an arbitrary rule.

```python
# perception/agent.py

class AgentTracker:
    MIN_AREA_PX = 4000

    def __init__(self):
        self.reset()

    def reset(self):
        self._swept: list[Rect] = []
        self.present = False

    def update(self, frame, diff, H) -> tuple[float, float] | None:
        mask = cv2.dilate((diff > PIXEL_DELTA).astype(np.uint8), KERNEL_9)
        h, w = mask.shape
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)

        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            edge = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
            if not (edge and area > self.MIN_AREA_PX):
                continue

            tip_px = self._far_tip(lbl == i, entered_from=(x, y, bw, bh, w, h))
            self.present = True
            self._swept.append(footprint_mm_at(H, tip_px, radius_mm=45))
            return px_to_mm(H, tip_px)
        self.present = False
        return None

    @staticmethod
    def _far_tip(component_mask, entered_from) -> tuple[int, int]:
        """The hand is the point furthest from the edge the arm came through."""
        ys, xs = np.nonzero(component_mask)
        x, y, bw, bh, w, h = entered_from
        if x <= 2:      return (int(xs.max()), int(ys[xs.argmax()]))
        if y <= 2:      return (int(xs[ys.argmax()]), int(ys.max()))
        if x + bw >= w - 2: return (int(xs.min()), int(ys[xs.argmin()]))
        return (int(xs[ys.argmin()]), int(ys.min()))

    def swept_mm(self) -> list[Rect]:
        return self._swept
```

Taking the point furthest from the entering edge gives you the hand rather than the elbow.

### What it cannot do, and the upgrade

| Limitation | Consequence |
| --- | --- |
| Cannot tell hand from sleeve or cable | Occasional spurious sweep; harmless, H2 is checked first |
| Cannot distinguish two hands | "Who moved it" means "a person moved it" |
| Cannot tell a full hand from an empty one | H3 cannot confirm the object actually left with the hand |

The third is the only one that costs accuracy. The fix, if it bites, is MediaPipe Hands — wrist
and fingertip keypoints, ~5 ms on CPU, works acceptably from overhead, and grasp aperture
separates full from empty:

```python
def grasping(landmarks) -> bool:
    """Thumb tip to index tip, normalised by hand span."""
    thumb, index, wrist = landmarks[4], landmarks[8], landmarks[0]
    span = np.linalg.norm(np.array(landmarks[9]) - np.array(wrist))
    return np.linalg.norm(np.array(thumb) - np.array(index)) / span < 0.45
```

Forty minutes and one dependency. Do it only if the heuristic misbehaves.

### Graceful degradation

**The permanence logic does not depend on the agent at all.** H1, H2 and H4 use no agent
evidence. If the detector fails completely, the system degrades from *"you carried it off"* to
*"it left the desk, cause unknown"* — a weaker answer, not a wrong one. That property is what
makes the heuristic acceptable rather than load-bearing.

---

## 10. Disappearance resolution

Entity `E` was `VISIBLE` at pose `P`, its footprint intersects a changed region, and it
produced no match. Something must account for that.

```python
# worldmodel/resolve.py

@dataclass
class Hypothesis:
    kind: EventKind
    confidence: float
    parent: str | None
    relation: Relation
    status: Status
    cause: str
    note: str
```

Evaluate in order; take the first that fires; record the rest as `alternatives`.

### H1 — Moved

`E`'s embedding matched a detection elsewhere on the mat. Not really a disappearance; checked
first because it is the common case and it is free (the match already happened).

```python
def h1_moved(world, e, matches) -> Hypothesis | None:
    hit = matches.get(e.id)
    if hit is None:
        return None
    return Hypothesis(EventKind.MOVED, 0.95,
                      parent=surface_under(world, hit.centroid),
                      relation=Relation.ON, status=Status.VISIBLE,
                      cause="agent", note=f"moved to {phrase_pos(hit.centroid)}")
```

### H2 — Covered

A detection now overlaps `E`'s last footprint that was not there before.

```python
COVER_MIN = 0.5

def h2_covered(world, e, new_entities) -> Hypothesis | None:
    fp = world.footprint(e)
    best, best_ov = None, 0.0
    for c in new_entities:                       # entities detected at this settle
        ov = overlap_fraction(fp, world.footprint(c))
        if ov > best_ov:
            best, best_ov = c, ov
    if best is None or best_ov < COVER_MIN:
        return None

    rel = Relation.IN if best.is_container else Relation.UNDER
    return Hypothesis(EventKind.COVERED,
                      confidence=min(0.98, 0.6 + 0.4 * best_ov),
                      parent=best.id, relation=rel, status=Status.HIDDEN,
                      cause=f"covered_by:{best.id}",
                      note=f"{rel.value.lower()} the {best.label}")
```

`IN` versus `UNDER` comes from `is_container`, which the VLM sets at labelling time. The
distinction only affects phrasing — *"in the tin"* against *"under the box"* — but that phrasing
is what makes the demo land.

### H3 — Carried off

The agent swept `E`'s footprint during the motion burst and `E` matched nothing anywhere.

```python
def h3_carried(world, e, obs) -> Hypothesis | None:
    fp = world.footprint(e)
    if not any(overlap_fraction(fp, s) > 0.3 for s in obs.agent_swept):
        return None
    if obs.agent_present:
        return Hypothesis(EventKind.PICKED_UP, 0.80,
                          parent=AGENT_ID, relation=Relation.HELD,
                          status=Status.HIDDEN, cause="agent",
                          note="in your hand")
    return Hypothesis(EventKind.LEFT_DESK, 0.85,
                      parent=None, relation=Relation.ON, status=Status.OFF_DESK,
                      cause="agent", note="taken off the desk")
```

### H4 — Unexplained

Nothing fits. Keep the last known pose and parent; lower confidence sharply.

```python
def h4_lost(world, e, obs) -> Hypothesis:
    return Hypothesis(EventKind.LOST, 0.40,
                      parent=e.parent, relation=e.relation,
                      status=Status.UNRESOLVED, cause="unexplained",
                      note="lost track of it")
```

This is not an embarrassing state; it is the honest one, and the query layer reports it as
such.

### The resolver

```python
def resolve(world, e, obs, matches, new_entities) -> Event:
    cands = [h for h in (h1_moved(world, e, matches),
                         h2_covered(world, e, new_entities),
                         h3_carried(world, e, obs),
                         h4_lost(world, e, obs)) if h is not None]
    winner, rest = cands[0], cands[1:]
    before = snapshot(world, e)

    world.reparent(e, winner.parent, winner.relation,
                   abs_pos=world.absolute(e) if winner.status != Status.VISIBLE
                           else matches[e.id].centroid)
    e.status, e.confidence = winner.status, winner.confidence
    e.last_seen = obs.ts
    if winner.status == Status.VISIBLE:
        e.last_confirmed = obs.ts

    return Event(ts=obs.ts, entity=e.id, kind=winner.kind, cause=winner.cause,
                 confidence=winner.confidence, from_state=before,
                 to_state=snapshot(world, e), frame_ref=obs.frame_ref,
                 note=winner.note,
                 alternatives=[{"kind": h.kind, "confidence": h.confidence,
                                "note": h.note} for h in rest])
```

---

## 11. Verification and falsified beliefs

A belief that is never tested is an assertion. The reveal path makes the system's claims
falsifiable, and it produces the best moment in the demo.

```python
# worldmodel/verify.py

REVEAL_RADIUS_MM = 60.0

def verify_children(world, occluder: Entity, obs, matches, now: float) -> list[Event]:
    """Called when `occluder` moves, shrinks, or leaves the desk."""
    events = []
    for child in world.children_of(occluder.id):
        if child.status != Status.HIDDEN:
            continue
        expected = world.absolute(child)
        hit = matches.get(child.id)
        found = hit is not None and dist(hit.centroid, expected) < REVEAL_RADIUS_MM

        if found:
            world.reparent(child, surface_under(world, hit.centroid),
                           Relation.ON, hit.centroid)
            child.status, child.confidence = Status.VISIBLE, 1.0
            child.last_confirmed = now
            events.append(Event(now, child.id, EventKind.REVEALED,
                                cause=f"revealed_by:{occluder.id}", confidence=1.0,
                                frame_ref=obs.frame_ref,
                                note=f"confirmed where I thought it was"))
        else:
            child.status, child.confidence = Status.UNRESOLVED, 0.2
            events.append(Event(now, child.id, EventKind.BELIEF_FALSIFIED,
                                cause=f"expected under {occluder.label}, absent on reveal",
                                confidence=0.2, frame_ref=obs.frame_ref,
                                note=f"I was wrong — not under the {occluder.label}"))
    return events
```

### Announce the falsification

When `BELIEF_FALSIFIED` fires, say so unprompted through the voice layer:

> *"I was wrong — the mug isn't under the box. The last time I actually saw it was 03:41,
> before you covered it."*

This is counterintuitive and correct: **a system that announces its own errors reads as far
more sophisticated than one that is silently right.** It demonstrates that the beliefs are
real, held with confidence, and tested — and it converts the worst failure mode, a wrong
answer, into a feature, because the error is caught and reported rather than asserted.

Budget ten minutes at the end to make this trigger cleanly. In the demo, palm an object out
from under the box while covering it, then lift the box. That moment is worth more than any
successful lookup.

### Free re-confirmation

Every `VISIBLE` entity that matched a detection gets `last_confirmed = now` and `confidence =
1.0`. The matching already happened, so this costs nothing — and it keeps the decay model
honest: confidence measures *time since the world last agreed with the model*, not time since
the entity was created.

---

## 12. Query layer

### Service boundary

The world model runs as an HTTP service. Perception is its only writer; everything else reads.

```
POST /observation        # perception -> world (internal, the only mutation)
GET  /state              # full snapshot: entities, relations, decayed confidence
GET  /events?since=<ts>  # event log tail
WS   /stream             # push: state deltas + events, for the UI
```

### Tools

The same four functions are exposed as HTTP endpoints and as MCP tools, so the voice layer, the
UI, and any external agent all query one model.

| Tool | Arguments | Returns |
| --- | --- | --- |
| `where_is` | `query: str` | entity, location phrase, confidence, last_confirmed, supporting event |
| `whats_in` | `container: str` | entities with that container as an ancestor, each with confidence |
| `who_moved` | `query: str` | most recent movement event: cause, timestamp, frame_ref |
| `history` | `query: str, limit: int` | chronological events for that entity |

Each returns compact JSON, never prose. **Phrasing is the model's job; grounding is the world
model's job.** Keeping that line clean is what stops the system hallucinating positions.

```python
@app.get("/tools/where_is")
def where_is(query: str):
    e = resolve_referent(world, query)
    if e is None:
        return {"found": False, "query": query}
    now = time.time()
    return {
        "found": True,
        "entity": e.label,
        "location": phrase_location(world, e),
        "status": e.status,
        "confidence": round(world.decayed(e, now), 2),
        "last_confirmed": e.last_confirmed,
        "seconds_since_confirmed": round(now - e.last_confirmed),
        "supporting_event": last_event_for(world, e.id),
    }
```

### Location phrasing

Walk the parent chain, deepest first.

```python
PREP = {Relation.IN: "in", Relation.UNDER: "under",
        Relation.ON: "on", Relation.HELD: "held by"}


def phrase_location(world, e) -> str:
    parts, node = [], e
    while node.parent and node.parent != DESK:
        par = world.entities[node.parent]
        parts.append(f"{PREP[node.relation]} the {par.label}")
        node = par
    if not parts:
        return f"on the desk, {quadrant(world.absolute(e))}"
    return ", ".join(parts) + " on the desk"


def quadrant(pos) -> str:
    x0, y0, x1, y1 = MAT_BOUNDS
    fx, fy = (pos[0] - x0) / (x1 - x0), (pos[1] - y0) / (y1 - y0)
    v = "top" if fy < 0.33 else "bottom" if fy > 0.67 else "middle"
    h = "left" if fx < 0.33 else "right" if fx > 0.67 else "centre"
    return "in the centre" if (v, h) == ("middle", "centre") else f"toward the {v} {h}"
```

Nobody wants to hear millimetres.

### Uncertainty in the answer

Confidence is rendered, not hidden.

| Confidence | Phrasing |
| --- | --- |
| > 0.9 | "It's under the box." |
| 0.7 – 0.9 | "It should be under the box — I haven't seen it directly since 03:41, but the box hasn't moved." |
| 0.4 – 0.7 | "Probably under the box, though I'm not certain." |
| < 0.4 | "I've lost track of it. Last confirmed on the desk at 03:41." |

### Voice loop

Qwen3.5-Omni handles the conversational turn end to end. Its Thinker takes streaming audio and
video; its Talker synthesises speech in parallel, so no separate TTS is needed.

1. Mic audio streams in.
2. The four tools are registered; Omni decides when to call them.
3. Tool result returns as JSON.
4. Omni phrases the spoken answer under a constraining system prompt.

Feed it the **live camera stream as well as audio**. That is what makes referent resolution
work: when the user says *"where's the one I just had"* or points and says *"what about this
one"*, Omni sees the gesture and the scene and can map the utterance to an entity id. Text-only
would force the user to name objects exactly — a worse experience and a worse demo.

The system prompt must be blunt:

```
You have a tool-backed world model of the desk. Every claim you make about where
something is MUST come from a tool result. Never invent a location, never infer one
from the camera image alone, and never smooth over a low confidence score. If a tool
returns found=false, say you don't know. If it returns a low confidence, say so in
the words given to you.
```

Log every tool call and its result next to the spoken answer. This is how you catch
hallucination in testing rather than on stage.

---

## 13. Visualisation

Not decoration. It is how a spectator understands the system in three seconds, and it is the
debugging tool that will save hours. Build it early, against the fake stream, before perception
works.

One page, two panes.

**Left — the desk map.** A rectangle at the mat's aspect ratio, entities at absolute positions.

- Solid dot — `VISIBLE`
- Hollow dashed outline, nested *inside its parent's shape* — `HIDDEN`, so containment is
  visible at a glance
- Faded with a question mark — `UNRESOLVED`
- Greyed into a margin strip — `OFF_DESK`
- **Opacity tracks confidence.** A belief decaying over time is something you can *watch* fade,
  which communicates the whole confidence model without a word of explanation.

**Right — the event timeline.** Newest at top. Each row: timestamp, label, kind, cause,
confidence. `BELIEF_FALSIFIED` in red. Clicking a row shows its `frame_ref` keyframe — the
visual receipt.

Optionally overlay the live feed at low opacity behind the map. Seeing the real mug and the
model's dot sitting on top of each other is immediately convincing, and when they drift apart
you know instantly that calibration slipped.

Plain HTML plus a websocket to `/stream`. No framework — the state payload is small enough to
re-render the whole scene on every delta.

---

## 14. Build order

Two people. The first thirty minutes decide whether you spend the rest of the time building or
blocking each other.

### Hour zero — the contract

Before either of you writes real code, commit:

- `types.py`, verbatim — `Entity`, `Event`, `Detection`, `Observation`, the enums.
- The four endpoints and their exact JSON shapes.
- `fake_perception.py`, emitting a scripted sequence on a timer.

```python
# fake/fake_perception.py — the most valuable file in the repo
SCRIPT = [
    (0.0,  "appear", "mug",  (150, 200), (80, 80)),
    (3.0,  "appear", "box",  (400, 220), (160, 140)),
    (6.0,  "move",   "box",  (150, 200)),          # box now covers mug
    (9.0,  "move",   "box",  (420, 300)),          # slid away; mug should be revealed
    (12.0, "remove", "mug"),                       # palmed off -> LEFT_DESK
]
```

That script exercises every path: appearance, covering, transitive motion, reveal-confirm, and
carried-off. The query layer, the tool surface and the UI can all be finished against it while
the camera is still on the bench.

### Split

**Person A — perception.** Camera loop, motion gating, settle detection, segmentation, DINOv3
matching, agent tracker. This half has the unknown unknowns; it goes to whoever has used DINO.

**Person B — everything downstream.** World model service, resolver, tools, Omni voice loop,
referent resolution, UI, demo script. Entirely against the fake stream until integration.

Note the resolver sits with B, not A. It is pure logic over the `Observation` contract, it is
fully testable against the fake stream, and keeping it out of the perception process means A
can rewrite segmentation without touching it.

### Schedule

| Hours | A | B |
| --- | --- | --- |
| 0–1 | Contract, repo, camera mounted, homography solved | Contract, service skeleton, fake stream |
| 1–4 | Real detections flowing as `Observation` | World model + resolver + UI on fake data |
| 4–8 | Agent tracker, segmentation hardening | Voice → tool → spoken answer |
| 8–10 | **Integrate.** Target: place mug, cover, ask, correct answer | |
| 10–15 | Sleep | Polish, second demo path |
| 15–20 | Polish, harden | Sleep |
| 16–20 | Pointer hardware, if the core is solid | |
| 20–22 | Rehearse five times. Write the submission | |

**Stagger the sleep.** Two people who both work straight through ship less than two people who
each get four hours.

### Cut order

Pointer hardware → event timeline UI → referent resolution → `who_moved` and `history`.
**Never cut the cover-and-ask path.** That is the demo.

### The pointer, if time allows

A pan/tilt laser (2× SG90, laser diode, ESP32) makes the answer physical, which beats any
amount of UI. Self-calibrate it with the same camera:

```python
def calibrate_pointer(cap, servo, H):
    """Sweep, find the dot, fit servo angles -> mm. No manual geometry."""
    samples = []
    for pan in range(-40, 41, 8):
        for tilt in range(-40, 41, 8):
            servo.goto(pan, tilt); time.sleep(0.15)
            dot_px = brightest_spot(grab(cap))
            if dot_px is not None:
                samples.append((pan, tilt, *px_to_mm(H, dot_px)))
    return fit_biquadratic(samples)     # (x_mm, y_mm) -> (pan, tilt)
```

Forty-five minutes, and the fact that it calibrated itself against the same camera is worth
saying out loud.

For a hidden object, point at the **covering** object: *"it's under this."* Pointing at what
you cannot see is the whole idea expressed in one gesture.

---

## 15. Failure modes

Ranked by how likely each is to cost you an hour.

| Failure | Symptom | Mitigation |
| --- | --- | --- |
| **Auto-exposure shift** | Matching degrades the instant a hand enters frame; looks like a model problem | Lock exposure, WB and focus before any code. Listed first because it is both the most likely and the most misleading |
| **Segmenting outside changed regions** | Every object "disappears" every settle | Only entities whose footprint intersects a changed region can be missing (§7) |
| **Hard shadows** | Phantom entities beside real objects | Chromatic mat + hue-only foreground mask. Diffuse off-axis lamp |
| **Mask flicker** | One object splits into two entities | Two-settle confirmation before minting |
| **Identical objects** | Two black pens swap identity | Spatial prior; on ambiguity keep both and say so |
| **Embedding drift** | An entity stops matching itself | Conservative exemplar policy: `> 0.85` to learn, `< 0.95` to add, cap 8, evict most redundant |
| **Boom bumped** | Every position off by a constant | Coherent-marker-shift detection + auto-recalibrate |
| **Partial covering** | 30% overlap, H2 does not fire, H4 marks it lost | `COVER_MIN` is tunable; if the visible remainder still matches, H1 catches it and it was never a disappearance |
| **Placing onto a hidden object's spot** | New object lands on a `HIDDEN` entity's footprint | Resolve containment against topmost *visible* entity only; the hidden entity keeps its parent |
| **Omni invents a location** | Confident answer with no grounding | Constraining system prompt; log every tool call beside its spoken answer |

### Limitations worth stating plainly

Naming these in the write-up reads as better engineering than presenting the system as
universal:

- Objects moved while a body occludes the whole mat are unresolvable.
- Stacking beyond two or three levels is untested; footprint overlap gets noisier with depth.
- The system cannot distinguish two agents, so "who moved it" means "a person moved it".
- A cold start with a non-empty desk mints new entities for everything already there; prior
  identities are not recoverable.