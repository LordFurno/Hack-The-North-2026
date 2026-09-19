from core import Detection, Observation, Rect
from calib import MAT_BOUNDS, level, pxToMm, rectPxToMm, solveHomography
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
    #`mask` and `bounds` are not part of the contract the world model depends on -- that
    #is `detect` alone -- but the agent tracker needs a foreground mask and the region the
    #detector can actually see, and the detector is the only thing that knows either.
    bounds: BoxPx

    def detect(self, frame: np.ndarray, regions: list[BoxPx]) -> list[BoxPx]: ...
    def mask(self, frame: np.ndarray) -> np.ndarray: ...


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


def matMaskPx(H: np.ndarray, shape) -> tuple[np.ndarray, BoxPx]: #(mask, its bounding box)
    #MAT_BOUNDS pushed back through H. Anything outside it is not the desk, so it is not
    #an object however different from the surface it looks.
    x0, y0, x1, y1 = MAT_BOUNDS
    inv = np.linalg.inv(H)
    quad = []
    for mm in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        q = inv @ np.array([mm[0], mm[1], 1.0])
        quad.append((q[0] / q[2], q[1] / q[2]))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(quad)], 1)
    bx, by, bw, bh = cv2.boundingRect(np.int32(quad))
    return mask, (bx, by, bx + bw, by + bh)


def changedFraction(regions: list[BoxPx], shape) -> float:
    #Boxes overlap, so sum of areas overstates -- which is the safe direction for a guard
    #that only ever has to notice "suspiciously much of the frame at once".
    h, w = shape[:2]
    return min(1.0, sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in regions) / float(w * h))


# ---- detectors ----------------------------------------------------------

class RefDiffDetector: #Primary. Whatever is not the empty desk is an object
    #Differenced per channel, not on grey. A terracotta mug and a green mat can sit four
    #grey levels apart while being 150 apart in blue, and thresholding their luminance
    #shreds a patterned object into its own light and dark patches.
    def __init__(self, reference: np.ndarray, cfg: Config):
        if reference is None:
            raise DetectorError("no reference frame: run calibrate.py and press r")
        self.reference = reference
        self.cfg = cfg
        h, w = reference.shape[:2]
        self.bounds = (0, 0, w, h) #The reference covers the whole frame, surround included

    def mask(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape != self.reference.shape:
            raise DetectorError(f"frame {frame.shape} does not match "
                                f"reference {self.reference.shape}")
        d = cv2.absdiff(frame, level(self.reference, frame))
        return morph((d.max(axis=2) > self.cfg.ref_delta).astype(np.uint8), self.cfg.morph_px)

    def detect(self, frame: np.ndarray, regions: list[BoxPx]) -> list[BoxPx]:
        if changedFraction(regions, frame.shape) > self.cfg.max_changed_frac:
            raise DetectorError("most of the frame changed at once, camera probably moved")
        return [b for b in components(self.mask(frame), self.cfg.min_object_px)
                if any(intersects(b, r) for r in regions)]


class ChromaDetector: #Optional. Only worth it on a known-hue surface, and then it is better
    #Hue distance, ignoring the value channel, so a cast shadow keeps the surface's hue and
    #is rejected for free. Refdiff has no way to tell a shadow from a thin dark object.
    #Everything off the surface reads as foreground here, so this one needs the mat mask:
    #without it the whole surround is one enormous object touching every changed region.
    def __init__(self, cfg: Config, H: np.ndarray, shape):
        self.cfg = cfg
        self.matMask, self.bounds = matMaskPx(H, shape)

    def mask(self, frame: np.ndarray) -> np.ndarray:
        h, s, _v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        dh = np.minimum(np.abs(h.astype(int) - self.cfg.chroma_hue),
                        180 - np.abs(h.astype(int) - self.cfg.chroma_hue))
        fg = (~((dh < self.cfg.chroma_tol) & (s > self.cfg.chroma_sat))).astype(np.uint8)
        return morph(fg * self.matMask, self.cfg.morph_px)

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

    #A box still touching the edge of what the detector can see, once everything has
    #stopped, is an arm reaching in rather than an object on the desk. Dropped either way,
    #tracker or no tracker, because nothing in the system could ever remove the entity it
    #would otherwise mint. With a tracker it also answers agent_present -- the question H3
    #actually asks, which is whether the hand is there NOW.
    reaching = [b for b in boxes if touchesBorder(b, det.bounds)]
    boxes = [b for b in boxes if b not in reaching]
    if agent is not None:
        agent.present = bool(reaching)

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


def histEmbed(frame: np.ndarray, box: BoxPx) -> np.ndarray:
    #NOT re-identification, and no substitute for DINO: a hue-saturation histogram tells
    #two differently coloured objects apart and nothing more. It exists so the loop can be
    #run and tuned on a machine with no torch on it, where the alternative is a constant
    #vector that scores 0.0 against everything and mints a new entity every settle.
    hsv = cv2.cvtColor(frame[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [24, 4], [0, 180, 0, 256]).flatten()
    n = float(np.linalg.norm(h))
    return h / n if n else h #L2, so a dot is a cosine, same as the real embedder


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
        embed=None, clock=time.time, resolveH: bool = True, root: str = SNAP_DIR,
        preview=None):
    #The trigger is motion STOPPING, not hand detection: frame difference over a threshold,
    #then quiet for ~500 ms. No model, no assumptions, trivially reliable.
    #
    #`preview` is shown to a human and read by nothing: Observation stays the only write
    #into the world. Whatever is passed here must return immediately -- anything that
    #blocks on a socket in this loop is dropped frames and a settle that never fires.
    embed = embed or histEmbed
    prev, quiet, settled = None, 0, None

    while True:
        ok, frame = cap.read()
        if not ok:
            return #A file ran out, or the camera went away. Either way there is no next frame
        if preview is not None:
            preview(frame)
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
