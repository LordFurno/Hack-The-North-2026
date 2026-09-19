from dataclasses import dataclass
from core import Detection, Entity, Status
from calib import dist
from world import World
import numpy as np
import cv2, math

#The embedder is the only thing here that needs torch, and it imports it inside
#__init__. Matching is pure arithmetic over the exemplar banks, so this module stays
#importable, and the tests stay fast, on a machine with no torch installed at all.

MODEL_REPO = "facebookresearch/dinov3"
MODEL_NAME = "dinov3_vits16"
FALLBACK_REPO = "facebookresearch/dinov2" #dinov3 weights are gated. One fallback, never a
FALLBACK_NAME = "dinov2_vits14"           #retry: a download that needs auth fails the same way twice

CROP_PAD = 0.10 #Context around the box, an object rarely fills its own bbox
CROP_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

#0.60 seems to work better
#used to 0.55
MATCH_LO = 0.60 #Below this it is a new entity, not a returning one
MARGIN_MIN = 0.05 #Winner must beat the runner-up by this
PRIOR_WEIGHT = 0.10
PRIOR_SIGMA = 150.0 #mm


# ---- embedding ----------------------------------------------------------

def tightCrop(frame_bgr: np.ndarray, bbox_px: tuple[int, int, int, int],
              pad: float = CROP_PAD) -> np.ndarray:
    x0, y0, x1, y1 = bbox_px
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    h, w = frame_bgr.shape[:2]
    x0, x1 = max(0, int(x0 - dx)), min(w, int(x1 + dx))
    y0, y1 = max(0, int(y0 - dy)), min(h, int(y1 + dy))
    return frame_bgr[y0:y1, x0:x1]


def squarePad(crop: np.ndarray) -> np.ndarray:
    #Pad, do not stretch. A stretched crop of a rotated object embeds differently from
    #the same object unrotated, and that false negative costs an hour to find.
    h, w = crop.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, crop.shape[2]), crop.dtype)
    y, x = (side - h) // 2, (side - w) // 2
    out[y:y + h, x:x + w] = crop
    return out


def loadBackbone(torch, name: str) -> tuple[str, object]: #(name actually loaded, model)
    try:
        return name, torch.hub.load(MODEL_REPO, name)
    except Exception as e:
        print(f"{MODEL_REPO}:{name} unavailable ({type(e).__name__}: {e}), "
              f"using {FALLBACK_REPO}:{FALLBACK_NAME}")
        return FALLBACK_NAME, torch.hub.load(FALLBACK_REPO, FALLBACK_NAME)


class Embedder: #Instance re-identification, not classification. The VLM names a thing once, at birth.
    def __init__(self, name: str = MODEL_NAME, device: str | None = None):
        import torch #Lazily, so importing identity costs nothing
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name, self.model = loadBackbone(torch, name)
        self.model = self.model.to(self.device).eval()

    def __call__(self, frame_bgr: np.ndarray, bbox_px: tuple[int, int, int, int]) -> np.ndarray:
        crop = squarePad(tightCrop(frame_bgr, bbox_px))
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = self.torch.from_numpy(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
        with self.torch.no_grad():
            f = self.model(x[None].to(self.device))[0]
        return self.torch.nn.functional.normalize(f, dim=-1).cpu().numpy() #L2, so a dot is a cosine


# ---- matching -----------------------------------------------------------

@dataclass
class MatchResult:
    entity: Entity | None
    score: float
    ambiguous: bool = False
    runner_up: Entity | None = None


def match(world: World, det: Detection) -> MatchResult:
    scored = []
    for e in world.entities.values():
        if e.status == Status.OFF_DESK or e.isAgent:
            continue
        s = e.bank.best(det.embedding)
        d = dist(det.centroid, world.absolute(e))
        s += PRIOR_WEIGHT * math.exp(-d / PRIOR_SIGMA) #The thing that was here 200ms ago is the thing here now
        scored.append((s, e))

    if not scored:
        return MatchResult(None, 0.0)
    scored.sort(key=lambda t: -t[0])
    s1, e1 = scored[0]
    s2, e2 = scored[1] if len(scored) > 1 else (0.0, None)

    if s1 < MATCH_LO:
        return MatchResult(None, s1)
    if s1 - s2 < MARGIN_MIN:
        return MatchResult(e1, s1, ambiguous=True, runner_up=e2) #Do not guess silently
    return MatchResult(e1, s1)
