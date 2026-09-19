from core import Detection, Observation, Rect
from calib import level, pxToMm, rectPxToMm, solveHomography
from agent import touchesBorder
from dataclasses import asdict, dataclass
from typing import Protocol
import numpy as np
import cv2, json, os, time

#Everything between a camera and an Observation. Perception writes to the world through
#exactly one object, so nothing here knows what an Entity is, and the detector can be
#swapped for a better one without a line of reasoning code changing.

CONFIG_PATH = "config.json"
SNAP_DIR = "snaps"

BoxPx = tuple[int, int, int, int] #(x0,y0,x1,y1) px, axis-aligned


@dataclass
class Config: #Every threshold in perception, so tune.py has one file to write
    pixel_delta: int = 25 #Per-pixel grey difference that counts as changed
    motion_frac: float = 0.002 #Fraction of the frame moving that counts as motion
    settle_frames: int = 15 #~500 ms at 30 fps. Drop to 8 if people work fast
    min_region_px: int = 400 #Smallest changed region worth looking at
    min_object_px: int = 900 #Smallest blob a detector will call an object
    ref_delta: int = 30 #Refdiff threshold against the empty-desk frame
    morph_px: int = 5
    max_changed_frac: float = 0.6 #More than this at once is a moved camera, not an action
    confirm_settles: int = 0 #Flicker guard: settles a detection must survive before minting
    chroma_hue: int = 60 #ChromaDetector only, OpenCV's 0-179 scale
    chroma_tol: int = 18
    chroma_sat: int = 60

    @staticmethod
    def load(path: str = CONFIG_PATH) -> "Config":
        if not os.path.exists(path):
            return Config()
        with open(path) as f:
            data = json.load(f)
        known = {f for f in Config.__dataclass_fields__}
        for k in set(data) - known:
            print(f"warning: {path} sets unknown threshold {k!r}, ignoring it")
        return Config(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str = CONFIG_PATH):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)


class DetectorError(Exception):
    #A detector that cannot see raises. It must never quietly return nothing: an empty
    #Observation makes missingEntities fire for the whole changed region and marks the
    #entire desk LOST in a single tick.
    pass


class Detector(Protocol):
    def detect(self, frame: np.ndarray, regions: list[BoxPx]) -> list[BoxPx]: ...


# ---- pixels -------------------------------------------------------------

def gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def morph(mask: np.ndarray, px: int) -> np.ndarray:
    #Open then close: drop the speckle first, then heal what the opening tore.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def components(mask: np.ndarray, minArea: int) -> list[BoxPx]:
    n, _lbl, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= minArea:
            out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def intersects(a: BoxPx, b: BoxPx) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def changedRegions(prevSettled: np.ndarray, now: np.ndarray, cfg: Config) -> list[BoxPx]:
    d = cv2.absdiff(gray(now), gray(prevSettled))
    mask = morph((d > cfg.pixel_delta).astype(np.uint8), cfg.morph_px)
    return components(mask, cfg.min_region_px)


def changedFraction(regions: list[BoxPx], shape) -> float:
    #Boxes overlap, so sum of areas overstates -- which is the safe direction for a guard
    #that only ever has to notice "suspiciously much of the frame at once".
    h, w = shape[:2]
    return min(1.0, sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in regions) / float(w * h))


# ---- detectors ----------------------------------------------------------

class RefDiffDetector: #Primary. Whatever is not the empty desk is an object
    def __init__(self, reference: np.ndarray, cfg: Config):
        if reference is None:
            raise DetectorError("no reference frame: run calibrate.py and press r")
        self.reference = gray(reference)
        self.cfg = cfg

    def detect(self, frame: np.ndarray, regions: list[BoxPx]) -> list[BoxPx]:
        g = gray(frame)
        if g.shape != self.reference.shape:
            raise DetectorError(f"frame {g.shape} does not match reference {self.reference.shape}")
        if changedFraction(regions, g.shape) > self.cfg.max_changed_frac:
            raise DetectorError("most of the frame changed at once, camera probably moved")

        fg = morph((cv2.absdiff(g, level(self.reference, g)) > self.cfg.ref_delta).astype(np.uint8),
                   self.cfg.morph_px)
        return [b for b in components(fg, self.cfg.min_object_px)
                if any(intersects(b, r) for r in regions)]


class ChromaDetector: #Optional. Only worth it on a known-hue surface, and then it is better
    #Hue distance, ignoring the value channel, so a cast shadow keeps the surface's hue and
    #is rejected for free. Refdiff has no way to tell a shadow from a thin dark object.
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def mask(self, frame: np.ndarray) -> np.ndarray:
        h, s, _v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        dh = np.minimum(np.abs(h.astype(int) - self.cfg.chroma_hue),
                        180 - np.abs(h.astype(int) - self.cfg.chroma_hue))
        return morph((~((dh < self.cfg.chroma_tol) & (s > self.cfg.chroma_sat))).astype(np.uint8),
                     self.cfg.morph_px)

    def detect(self, frame: np.ndarray, regions: list[BoxPx]) -> list[BoxPx]:
        if changedFraction(regions, frame.shape) > self.cfg.max_changed_frac:
            raise DetectorError("most of the frame changed at once, camera probably moved")
        return [b for b in components(self.mask(frame), self.cfg.min_object_px)
                if any(intersects(b, r) for r in regions)]


# ---- keyframes ----------------------------------------------------------

def snapDir(root: str = SNAP_DIR) -> str:
    os.makedirs(root, exist_ok=True)
    return root


def saveFrame(frame: np.ndarray, stamp: str, root: str = SNAP_DIR) -> str:
    #Every Event carries one of these. Scrubbing shows the map state AND the image it was
    #believed from, which beats a timeline alone.
    path = os.path.join(snapDir(root), f"{stamp}.jpg")
    cv2.imwrite(path, frame)
    return path


def saveCrop(frame: np.ndarray, box: BoxPx, stamp: str, i: int, root: str = SNAP_DIR) -> str:
    path = os.path.join(snapDir(root), f"{stamp}-{i:02d}.jpg")
    cv2.imwrite(path, frame[box[1]:box[3], box[0]:box[2]])
    return path


# ---- the analysis pass --------------------------------------------------

def analyse(prevSettled: np.ndarray, now: np.ndarray, det: Detector, H: np.ndarray,
            agent, embed, cfg: Config, ts: float = None, root: str = SNAP_DIR) -> Observation:
    #Only changed regions are examined. Everything else is unchanged by definition, and
    #segmenting outside them is what makes every object on the desk "disappear" each settle.
    ts = time.time() if ts is None else ts
    changed = changedRegions(prevSettled, now, cfg)
    boxes = det.detect(now, changed)

    #A box still touching the frame border once everything has stopped is an arm reaching
    #in, not an object on the desk. Dropping it answers agent_present -- the question H3
    #actually asks, which is whether the hand is there NOW -- and stops the arm minting an
    #entity of its own. Nothing else in the system would ever get rid of that entity.
    reaching = [b for b in boxes if touchesBorder(b, now.shape)]
    if agent is not None:
        agent.present = bool(reaching)
        boxes = [b for b in boxes if b not in reaching]

    stamp = f"{ts:.3f}".replace(".", "_")
    dets = []
    for i, box in enumerate(boxes):
        (x0, y0), (x1, y1) = pxToMm(H, (box[0], box[1])), pxToMm(H, (box[2], box[3]))
        dets.append(Detection(centroid=((x0 + x1) / 2, (y0 + y1) / 2),
                              size=(abs(x1 - x0), abs(y1 - y0)),
                              embedding=embed(now, box),
                              crop_path=saveCrop(now, box, stamp, i, root)))

    return Observation(ts=ts, frame_ref=saveFrame(now, stamp, root), detections=dets,
                       changed=[rectPxToMm(H, r) for r in changed],
                       agent_swept=agent.swept_mm() if agent else [],
                       agent_present=agent.present if agent else False)


def observationJson(obs: Observation) -> dict:
    #The encoder paired with service.observationFrom, because perception may be out of
    #process. core.py stays the only contract: this is an encode, not a second schema.
    return {"ts": obs.ts, "frame_ref": obs.frame_ref,
            "detections": [{"centroid": list(d.centroid), "size": list(d.size),
                            "embedding": [float(v) for v in d.embedding],
                            "crop_path": d.crop_path} for d in obs.detections],
            "changed": [list(r) for r in obs.changed],
            "agent_swept": [list(r) for r in obs.agent_swept],
            "agent_present": obs.agent_present}


# ---- the settle loop ----------------------------------------------------

def run(cap, det: Detector, sink, cfg: Config, H: np.ndarray, agent=None,
        embed=None, clock=time.time, resolveH: bool = True, root: str = SNAP_DIR):
    #The trigger is motion STOPPING, not hand detection: frame difference over a threshold,
    #then quiet for ~500 ms. No model, no assumptions, trivially reliable.
    embed = embed or (lambda frame, box: np.zeros(1, dtype=float))
    prev, quiet, settled = None, 0, None

    while True:
        ok, frame = cap.read()
        if not ok:
            return #A file ran out, or the camera went away. Either way there is no next frame
        g = gray(frame)

        if prev is not None:
            diff = cv2.absdiff(g, prev)
            if float((diff > cfg.pixel_delta).mean()) > cfg.motion_frac:
                quiet = 0
                if agent is not None:
                    agent.update(frame, diff, H) #Feeds H3 and nothing else
            else:
                quiet += 1
                if quiet == cfg.settle_frames and settled is not None:
                    if resolveH: #A bumped camera costs nothing if the markers are visible
                        solved = solveHomography(frame) #None means a hand is over one
                        H = H if solved is None else solved
                    try:
                        sink(analyse(settled, frame, det, H, agent, embed, cfg,
                                     ts=clock(), root=root))
                        settled = frame
                    except DetectorError as e:
                        print(f"detector failed, skipping settle: {e}")
                    if agent is not None:
                        agent.reset()
                elif settled is None:
                    settled = frame
        prev = g


def openVideo(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    return cap


def videoClock(cap: cv2.VideoCapture, start: float = 0.0):
    #A recorded run has its own timebase, and decay measures time since the world last
    #agreed with the model. Wall clock on a replay makes a 17 second video look instant.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return lambda: start + cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
