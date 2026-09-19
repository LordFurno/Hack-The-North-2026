from core import Detection, Observation, Rect
from calib import overlapFraction
from world import COVER_MIN
import numpy as np
import time

#The scripted sequence exercises every path: appearance, covering, transitive
#motion, reveal-confirm and carried-off.
SCRIPT = [
    (0.0,  "appear", "mug",  (150, 200), (80, 80)),
    (3.0,  "appear", "box",  (400, 220), (160, 140)),
    (6.0,  "move",   "box",  (150, 200)),          #box now covers mug
    (9.0,  "move",   "box",  (420, 300)),          #slid away, mug should be revealed
    (12.0, "remove", "mug"),                       #palmed off -> LEFT_DESK
]

EMBED_DIM = 64


class FakeDesk:
    #Ground truth for a desk, and the perception it would have produced. Objects are
    #only segmented inside changed regions, and only when nothing is sitting on them.
    def __init__(self):
        self.objects: dict[str, list] = {} #name -> [pos, size], in mm
        self.vectors: dict[str, np.ndarray] = {}
        self.n = 0 #Settle counter, stands in for a keyframe name

    def embedding(self, name: str) -> np.ndarray:
        v = self.vectors.get(name)
        if v is None:
            rng = np.random.default_rng(list(name.encode())) #Same object, same vector, every run
            v = rng.normal(size=EMBED_DIM)
            v = v / np.linalg.norm(v)
            self.vectors[name] = v
        return v

    def rect(self, name: str) -> Rect:
        (x, y), (w, h) = self.objects[name]
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def visible(self, name: str) -> bool:
        me = self.rect(name)
        return not any(overlapFraction(me, self.rect(o)) >= COVER_MIN
                       for o in self.objects if o != name)

    def detection(self, name: str) -> Detection:
        (x, y), size = self.objects[name]
        return Detection(centroid=(float(x), float(y)), size=(float(size[0]), float(size[1])),
                         embedding=self.embedding(name), crop_path=f"fake/{name}-{self.n:03d}.png")

    def palm(self, name: str):
        self.objects.pop(name, None) #Taken while covered, so nothing changed on camera

    def step(self, ts: float, kind: str, name: str,
             pos: tuple[float, float] = None, size: tuple[float, float] = None) -> Observation:
        changed: list[Rect] = []
        if name in self.objects:
            changed.append(self.rect(name))

        if kind == "appear":
            self.objects[name] = [pos, size]
            changed.append(self.rect(name))
        elif kind == "move":
            self.objects[name][0] = pos
            changed.append(self.rect(name))
        elif kind == "remove":
            self.objects.pop(name)
        else:
            raise ValueError(f"unknown script step {kind}")

        dets = [self.detection(o) for o in self.objects
                if self.visible(o) and any(overlapFraction(self.rect(o), c) > 0.0 for c in changed)]
        obs = Observation(ts=ts, frame_ref=f"fake-{self.n:03d}", detections=dets,
                          changed=changed, agent_swept=list(changed), agent_present=False)
        self.n += 1
        return obs


def scriptObservations(desk: FakeDesk = None) -> list[Observation]:
    desk = desk or FakeDesk()
    return [desk.step(*row) for row in SCRIPT]


def stream(speed: float = 1.0):
    #The same script on a timer, for anything downstream that wants a live feed.
    #Script time is stamped onto the wall clock here: decay measures time since the
    #world last agreed with the model, and 1970 is a long time to have not agreed.
    desk = FakeDesk()
    start = time.time()
    for row in SCRIPT:
        wait = row[0] / speed - (time.time() - start)
        if wait > 0:
            time.sleep(wait)
        yield desk.step(start + row[0] / speed, *row[1:])
