from core import AGENT_ID, DESK
from calib import MAT_BOUNDS, overlapFraction
from world import COVER_MIN
from fake import SCRIPT
from dataclasses import dataclass
import numpy as np
import cv2, json

#A synthetic overhead camera, so perception has something to run against before the boom
#is built. It renders the fake.py SCRIPT as a hand physically doing the work: the hand
#comes in over a frame border, an object moves only while the hand is on top of it, and
#nothing on the mat moves on its own. The JSON written beside the video is the answer key.

VIDEO_PATH = "synth.mp4"
TRUTH_PATH = "synth.json"

WIDTH, HEIGHT = 1280, 720
FPS = 30

#The mat is the pixel block rows 60:660, cols 240:1040. detectMarkers reports pixel
#CENTRES, so that block's corner centres land on the half pixel, and so does the centre
#of an even-sided marker drawn into it. Lining those two up is what makes the homography
#solved from this video the exact inverse of mmToPx, which is what lets the JSON below
#claim to be ground truth. 800x600 px over 600x450 mm, so 4/3 px per mm on both axes.
MAT_PX = (239.5, 59.5, 1039.5, 659.5) #x0,y0,x1,y1 px, the corners MAT_BOUNDS mm maps to
MARKER_PX = 72 #6 cells of 12 px: a 4x4 marker plus its one-cell black border
MARKER_QUIET_PX = 10 #White paper around the marker, as a printed one has
MARKER_MM = { #Marker id -> mm. Perception's MAT_MM has to agree with this
    0: (MAT_BOUNDS[0], MAT_BOUNDS[1]),
    1: (MAT_BOUNDS[2], MAT_BOUNDS[1]),
    2: (MAT_BOUNDS[2], MAT_BOUNDS[3]),
    3: (MAT_BOUNDS[0], MAT_BOUNDS[3]),
}

MAT_HUE = 60 #OpenCV's 0-179 scale, so 60 is green
MAT_SAT = 180
MAT_VAL = 120
MAT_GRAIN = 7 #Cloth weave, in V alone: hue is the one channel the foreground mask trusts
DESK_BGR = (78, 84, 96) #Off-mat surround, neutral and nowhere near the mat hue

TEXTURE = { #Base BGR and weave, kept far off the mat hue or the foreground mask drops it
    "mug": ((70, 92, 204), "stripes"), #Terracotta, hue ~5
    "box": ((190, 130, 90), "checker"), #Blue, hue ~108. Cardboard tan lands at hue ~16,
}                                       #a few degrees off skin, and a box the same colour
                                        #as the hand carrying it is a confound this fixture
                                        #has no reason to introduce.
TEXTURE_NOISE = 6
RIM_PX = 3 #Dark edge, so two objects that touch still segment apart

SKIN_BGR = (148, 176, 212)
HAND_RADIUS_MM = 55.0 #Palm. Wide enough to swallow the 80 mm mug whole, not the 160 mm box
ARM_WIDTH_MM = 70.0
ARM_EDGE_INSET = 0.2 #The arm crosses the middle 60% of an edge, so it never sweeps a marker
AGENT_MIN_AREA_PX = 4000 #AgentTracker.MIN_AREA_PX: under this the detector ignores a blob
PALM_OUT_MM = 90.0 #Where the palm waits between beats, far enough out to be fully off shot
SHOULDER_MM = 200.0 #The arm pivots here, further out again, so it enters WITH the palm and
                    #not before it: an arm hinged on the border pops into frame all at once,
                    #and that one-frame jump reads as motion long after the hand has gone.

LEAD_S = 1.5 #Quiet mat first, so the settle loop has a baseline to diff against
REACH_S = 0.7 #Border to the object
CARRY_S = 0.8 #Object to where it is going
RETREAT_S = 0.6 #Back out over the border
TAIL_S = 1.2 #Quiet after the last beat, long enough for one more settle
ACTION_S = REACH_S + CARRY_S + RETREAT_S
EASE_BLEND = 0.35 #How much of the hand's motion is eased rather than constant speed

SHADOW_SCALE = 0.62 #Scales BGR, which is a pure value change: hue and saturation survive it
SHADOW_OFFSET_PX = (8, 10)
SHADOW_BLUR_PX = 21 #One diffuse off-axis lamp, so no hard edge lands on an object
LENS_BLUR_PX = 3
SENSOR_NOISE = 2.5 #Far under PIXEL_DELTA=25, or the motion gate would never settle
SENSOR_SEED = 20260919

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_PATCHES: dict[tuple[str, int, int], np.ndarray] = {}


# ---- geometry -----------------------------------------------------------

PX_PER_MM_X = (MAT_PX[2] - MAT_PX[0]) / (MAT_BOUNDS[2] - MAT_BOUNDS[0])
PX_PER_MM_Y = (MAT_PX[3] - MAT_PX[1]) / (MAT_BOUNDS[3] - MAT_BOUNDS[1])


def mmToPx(pt: tuple[float, float]) -> tuple[float, float]:
    return (MAT_PX[0] + (pt[0] - MAT_BOUNDS[0]) * PX_PER_MM_X,
            MAT_PX[1] + (pt[1] - MAT_BOUNDS[1]) * PX_PER_MM_Y)


def pxToMm(pt: tuple[float, float]) -> tuple[float, float]:
    return (MAT_BOUNDS[0] + (pt[0] - MAT_PX[0]) / PX_PER_MM_X,
            MAT_BOUNDS[1] + (pt[1] - MAT_PX[1]) / PX_PER_MM_Y)


def rectMm(pos: tuple[float, float], size: tuple[float, float]):
    return (pos[0] - size[0] / 2, pos[1] - size[1] / 2,
            pos[0] + size[0] / 2, pos[1] + size[1] / 2)


def rectPx(pos: tuple[float, float], size: tuple[float, float]) -> tuple[int, int, int, int]:
    cx, cy = mmToPx(pos)
    w, h = size[0] * PX_PER_MM_X, size[1] * PX_PER_MM_Y
    return (round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2))


def iround(pt: tuple[float, float]) -> tuple[int, int]:
    return (int(round(pt[0])), int(round(pt[1])))


#The whole frame in mm. A pixel block [0:N] covers [-0.5, N-0.5] in pixel-centre coords.
FRAME_MM = (pxToMm((-0.5, -0.5)) + pxToMm((WIDTH - 0.5, HEIGHT - 0.5)))
HAND_RADIUS_PX = round(HAND_RADIUS_MM * PX_PER_MM_X)
ARM_WIDTH_PX = round(ARM_WIDTH_MM * PX_PER_MM_X)


# ---- the mat ------------------------------------------------------------

def drawMarkers(frame: np.ndarray):
    for i, mm in MARKER_MM.items():
        cx, cy = mmToPx(mm)
        x, y = round(cx - MARKER_PX / 2 + 0.5), round(cy - MARKER_PX / 2 + 0.5)
        q = MARKER_QUIET_PX
        frame[y - q:y + MARKER_PX + q, x - q:x + MARKER_PX + q] = 255
        marker = cv2.aruco.generateImageMarker(_DICT, i, MARKER_PX)
        frame[y:y + MARKER_PX, x:x + MARKER_PX] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def matBackground() -> np.ndarray:
    rng = np.random.default_rng(SENSOR_SEED)
    frame = np.full((HEIGHT, WIDTH, 3), DESK_BGR, np.uint8)
    x0, y0 = round(MAT_PX[0] + 0.5), round(MAT_PX[1] + 0.5)
    x1, y1 = round(MAT_PX[2] + 0.5), round(MAT_PX[3] + 0.5)

    h = np.full((y1 - y0, x1 - x0), MAT_HUE, np.uint8)
    s = np.full_like(h, MAT_SAT)
    v = np.clip(rng.normal(MAT_VAL, MAT_GRAIN, h.shape), 0, 255).astype(np.uint8)
    frame[y0:y1, x0:x1] = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    drawMarkers(frame)
    return frame


# ---- objects ------------------------------------------------------------

def fallbackTexture(name: str): #Anything the SCRIPT grows that TEXTURE has no entry for
    rng = np.random.default_rng(list(name.encode()))
    hue = (MAT_HUE + 30 + int(rng.integers(0, 120))) % 180 #At least 30 off the mat hue
    bgr = cv2.cvtColor(np.uint8([[[hue, 190, 190]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return (tuple(int(c) for c in bgr), "stripes" if hue % 2 else "checker")


def objectPatch(name: str, w: int, h: int) -> np.ndarray:
    #Textured, not flat: a flat rectangle gives the embedder nothing to tell two
    #same-sized objects apart with, which is the one thing it is here to do.
    key = (name, w, h)
    if key in _PATCHES:
        return _PATCHES[key]

    base, weave = TEXTURE.get(name) or fallbackTexture(name)
    rng = np.random.default_rng(list(name.encode())) #Same object, same weave, every run
    dark = tuple(int(c * 0.72) for c in base)
    patch = np.full((h, w, 3), base, np.uint8)

    if weave == "stripes":
        for x in range(0, w, 18):
            patch[:, x:x + 9] = dark
    elif weave == "checker":
        for y in range(0, h, 26):
            for x in range((y // 26 % 2) * 26, w, 52):
                patch[y:y + 26, x:x + 26] = dark

    patch = np.clip(patch + rng.normal(0, TEXTURE_NOISE, patch.shape), 0, 255).astype(np.uint8)
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), dark, RIM_PX)
    _PATCHES[key] = patch
    return patch


def paste(frame: np.ndarray, patch: np.ndarray, rect: tuple[int, int, int, int]):
    x0, y0, x1, y1 = rect
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(WIDTH, x1), min(HEIGHT, y1)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    frame[fy0:fy1, fx0:fx1] = patch[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]


# ---- the hand -----------------------------------------------------------

def drawHand(canvas: np.ndarray, palmPx, shoulderPx, colour, offset=(0, 0)):
    #A palm disc plus the arm back to the shoulder off frame. One connected component
    #touching a frame edge, which is exactly what AgentTracker looks for, and the palm
    #is the end furthest from that edge, which is what its _far_tip picks out.
    p = iround((palmPx[0] - offset[0], palmPx[1] - offset[1]))
    s = iround((shoulderPx[0] - offset[0], shoulderPx[1] - offset[1]))
    cv2.line(canvas, s, p, colour, ARM_WIDTH_PX)
    cv2.circle(canvas, p, HAND_RADIUS_PX, colour, -1)


def entryFor(target: tuple[float, float]):
    #An arm has to come in over the frame border. The far edge is behind the boom, so
    #the hand reaches from the near edge or the sides, whichever sits closest.
    fx0, fy0, fx1, fy1 = FRAME_MM
    lo, hi = ARM_EDGE_INSET, 1.0 - ARM_EDGE_INSET
    ax = min(max(target[0], fx0 + (fx1 - fx0) * lo), fx0 + (fx1 - fx0) * hi)
    ay = min(max(target[1], fy0 + (fy1 - fy0) * lo), fy0 + (fy1 - fy0) * hi)

    edges = {"bottom": ((ax, fy1), (0.0, 1.0), fy1 - target[1]),
             "left": ((fx0, ay), (-1.0, 0.0), target[0] - fx0),
             "right": ((fx1, ay), (1.0, 0.0), fx1 - target[0])}
    border, away, _ = min(edges.values(), key=lambda e: e[2])
    shoulder = (border[0] + away[0] * SHOULDER_MM, border[1] + away[1] * SHOULDER_MM)
    out = (border[0] + away[0] * PALM_OUT_MM, border[1] + away[1] * PALM_OUT_MM)
    return shoulder, out #Both outside, so the whole arm sits off frame between beats


# ---- the script as a timeline -------------------------------------------

@dataclass
class Beat:
    t: float #Video time the hand is clear and the mat goes quiet again
    kind: str
    name: str
    size: tuple[float, float] | None #Only "appear" brings a new object, and its size
    dest: tuple[float, float] | None #Where the script says it ends up. None if it leaves
    shoulder: tuple[float, float] #Where the arm pivots, off frame, mm
    path: list[tuple[float, tuple[float, float]]] #(video time, palm centre mm)
    grab: float #Video time the object starts following the palm
    release: float #...and stops. inf when it leaves the desk with the hand


def ease(u: float) -> float:
    #Blended, not a full smoothstep. A smoothstep parks the hand at walking pace either
    #side of every waypoint, and a dozen near-still frames mid-action is enough for the
    #settle loop to call the scene quiet and analyse a blur. This never drops below
    #1 - EASE_BLEND of the average speed, so an action reads as one unbroken motion burst.
    u = min(max(u, 0.0), 1.0)
    return (1.0 - EASE_BLEND) * u + EASE_BLEND * u * u * (3.0 - 2.0 * u)


def palmAt(beat: Beat, t: float):
    if not (beat.path[0][0] <= t <= beat.path[-1][0]):
        return None
    for (t0, p0), (t1, p1) in zip(beat.path, beat.path[1:]):
        if t <= t1:
            u = ease((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            return (p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u)
    return beat.path[-1][1]


def timeline(script=SCRIPT) -> tuple[list[Beat], float]:
    #Script time is a logical clock; here it becomes the moment the hand finishes. Each
    #beat is shifted by LEAD_S, so the gap the script leaves between rows becomes the
    #quiet window the settle loop needs.
    beats, pos = [], {}
    for row in script:
        t, kind, name = row[0], row[1], row[2]
        start = t + LEAD_S

        if kind == "appear":
            target, size = row[3], row[4]
            shoulder, out = entryFor(target)
            path = [(start, out), (start + REACH_S + CARRY_S, target), (start + ACTION_S, out)]
            grab, release = start, start + REACH_S + CARRY_S #It rides in with the hand
            pos[name] = target
        elif kind == "move":
            target, size = row[3], None
            shoulder, out = entryFor(pos[name])
            path = [(start, out), (start + REACH_S, pos[name]),
                    (start + REACH_S + CARRY_S, target), (start + ACTION_S, out)]
            grab, release = start + REACH_S, start + REACH_S + CARRY_S
            pos[name] = target
        elif kind == "remove":
            target, size = None, None
            shoulder, out = entryFor(pos[name])
            path = [(start, out), (start + REACH_S, pos.pop(name)), (start + ACTION_S, out)]
            grab, release = start + REACH_S, float("inf") #Palmed off, never let go of
        else:
            raise ValueError(f"unknown script step {kind}")

        beats.append(Beat(t=start + ACTION_S, kind=kind, name=name, size=size, dest=target,
                          shoulder=shoulder, path=path, grab=grab, release=release))
    return beats, beats[-1].t + TAIL_S


# ---- ground truth -------------------------------------------------------

@dataclass
class Thing:
    name: str
    pos: tuple[float, float] #mm
    size: tuple[float, float] #w,h mm
    z: int #Draw order. The last thing the hand touched sits on top of everything under it
    gone: bool = False #Carried off the desk


def hiddenFraction(me: Thing, things: dict[str, Thing], palmPx, anchorPx) -> float:
    #How much of this object the camera cannot see, measured on the geometry the renderer
    #actually draws rather than inferred from it. Off the frame counts as hidden too.
    x0, y0, x1, y1 = rectPx(me.pos, me.size)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return 1.0

    mask = np.zeros((h, w), np.uint8)
    mask[:max(0, min(h, -y0)), :] = 1
    mask[max(0, min(h, HEIGHT - y0)):, :] = 1
    mask[:, :max(0, min(w, -x0))] = 1
    mask[:, max(0, min(w, WIDTH - x0)):] = 1

    for th in things.values():
        if th.name == me.name or th.gone or th.z <= me.z:
            continue
        ox0, oy0, ox1, oy1 = rectPx(th.pos, th.size)
        ix0, iy0 = max(x0, ox0) - x0, max(y0, oy0) - y0
        ix1, iy1 = min(x1, ox1) - x0, min(y1, oy1) - y0
        if ix0 < ix1 and iy0 < iy1:
            mask[iy0:iy1, ix0:ix1] = 1

    if palmPx is not None:
        drawHand(mask, palmPx, anchorPx, 1, offset=(x0, y0))
    return float(mask.mean())


def parentOf(me: Thing, things: dict[str, Thing], held: str | None) -> str:
    #What the world model should conclude this object's parent is. Measured the way
    #world.covers measures it, so the JSON and /state are comparable without a fudge
    #factor. The hand hides things without ever becoming their parent -- unless it is
    #holding them, which is the one case that reparents.
    if me.name == held:
        return AGENT_ID
    mine = rectMm(me.pos, me.size)
    best, bestZ = DESK, me.z
    for th in things.values():
        if th.name == me.name or th.gone or th.z <= bestZ:
            continue
        if overlapFraction(mine, rectMm(th.pos, th.size)) >= COVER_MIN:
            best, bestZ = th.name, th.z
    return best


def handAreaPx(palmPx, shoulderPx) -> int:
    #How much hand is actually in shot. A palm poking one sliver over the border is not
    #yet an agent, and saying it is would make the answer key disagree with any detector
    #honest enough to have a minimum area.
    if palmPx is None:
        return 0
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    drawHand(mask, palmPx, shoulderPx, 1)
    return int(mask.sum())


def truthFor(i: int, t: float, names: list[str], things: dict[str, Thing],
             held: str | None, palm, palmPx, shoulderPx) -> dict:
    objects = {}
    for name in names:
        th = things.get(name)
        if th is None or th.gone:
            objects[name] = {"present": False, "px": None, "mm": None, "parent": None,
                             "occluded": False, "hidden_fraction": 1.0}
            continue
        hidden = hiddenFraction(th, things, palmPx, shoulderPx)
        objects[name] = {
            "present": True,
            "px": [round(v, 2) for v in mmToPx(th.pos)],
            "mm": [round(v, 2) for v in th.pos],
            "size_mm": [round(v, 2) for v in th.size],
            "parent": parentOf(th, things, held),
            "occluded": hidden >= COVER_MIN, #The threshold the world model covers at
            "hidden_fraction": round(hidden, 4),
        }
    area = handAreaPx(palmPx, shoulderPx)
    return {"i": i, "t": round(t, 4),
            "agent": {"present": area > AGENT_MIN_AREA_PX, #In shot, not merely mid-beat
                      "area_px": area,
                      "px": None if palm is None else [round(v, 2) for v in palmPx],
                      "mm": None if palm is None else [round(v, 2) for v in palm]},
            "objects": objects}


# ---- render -------------------------------------------------------------

def applyShadow(frame: np.ndarray, mask: np.ndarray):
    #A shadow keeps the mat's hue and loses only value, which is what scaling BGR by a
    #constant does -- and what the hue-only foreground mask is built to ignore. The mat
    #earns its keep here, so the fixture had better put it to the test.
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return #Bare mat, nothing to cast one

    dx, dy = SHADOW_OFFSET_PX
    pad = SHADOW_BLUR_PX + max(dx, dy) #Room for the blur and the offset, so neither clips
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(WIDTH, x + w + pad), min(HEIGHT, y + h + pad)

    soft = cv2.GaussianBlur(mask[y0:y1, x0:x1] * 255, (SHADOW_BLUR_PX, SHADOW_BLUR_PX), 0)
    cast = np.zeros_like(soft)
    cast[dy:, dx:] = soft[:-dy, :-dx] #Slice, not roll: no wrap at the edges
    a = (cast.astype(np.float32) / 255.0)[..., None] * (1.0 - SHADOW_SCALE)
    sub = frame[y0:y1, x0:x1]
    sub[:] = (sub.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def drawFrame(background: np.ndarray, things: dict[str, Thing],
              palmPx, shoulderPx, noise: np.ndarray) -> np.ndarray:
    frame = background.copy()
    order = sorted((th for th in things.values() if not th.gone), key=lambda th: th.z)

    shadow = np.zeros((HEIGHT, WIDTH), np.uint8)
    for th in order:
        x0, y0, x1, y1 = rectPx(th.pos, th.size)
        cv2.rectangle(shadow, (x0, y0), (x1, y1), 1, -1)
    if palmPx is not None:
        drawHand(shadow, palmPx, shoulderPx, 1)
    applyShadow(frame, shadow)

    for th in order:
        rect = rectPx(th.pos, th.size)
        paste(frame, objectPatch(th.name, rect[2] - rect[0], rect[3] - rect[1]), rect)
    if palmPx is not None:
        drawHand(frame, palmPx, shoulderPx, SKIN_BGR)

    frame = cv2.GaussianBlur(frame, (LENS_BLUR_PX, LENS_BLUR_PX), 0)
    cv2.randn(noise, 0, SENSOR_NOISE) #Into a buffer we own: a fresh float array per frame
    return cv2.add(frame, noise, dtype=cv2.CV_8U) #costs more than everything else combined


def render(videoPath: str = VIDEO_PATH, truthPath: str = TRUTH_PATH,
           script=SCRIPT) -> dict:
    beats, duration = timeline(script)
    names = list(dict.fromkeys(row[2] for row in script))
    background = matBackground()
    noise = np.empty((HEIGHT, WIDTH, 3), np.int16)
    cv2.setRNGSeed(SENSOR_SEED) #The grain is OpenCV's RNG, the weave is numpy's. Both fixed

    writer = cv2.VideoWriter(videoPath, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open {videoPath} for writing")

    things: dict[str, Thing] = {}
    records, total = [], int(round(duration * FPS))

    for i in range(total):
        t = i / FPS
        for b in beats:
            if b.dest is not None and b.release <= t:
                #Let go exactly where the script says. No frame lands on the release
                #instant, so without this an object is left a few mm short of its mark
                #and the answer key faithfully records the wrong number.
                things[b.name].pos = b.dest
            if b.kind == "remove" and t > b.path[-1][0]:
                things[b.name].gone = True #Off the desk, and the hand went with it
        beat = next(((bi, b) for bi, b in enumerate(beats)
                     if b.path[0][0] <= t <= b.path[-1][0]), None)

        palm = palmAt(beat[1], t) if beat else None
        held = None
        if beat and beat[1].grab <= t < beat[1].release:
            bi, b = beat
            held = b.name
            th = things.get(held) or Thing(name=held, pos=palm, size=b.size, z=bi)
            th.pos, th.z, things[held] = palm, bi, th #Objects move only under the hand

        shoulderPx = mmToPx(beat[1].shoulder) if beat else None
        palmPx = mmToPx(palm) if palm is not None else None
        records.append(truthFor(i, t, names, things, held, palm, palmPx, shoulderPx))
        writer.write(drawFrame(background, things, palmPx, shoulderPx, noise))

    writer.release()

    truth = {
        "video": videoPath,
        "fps": FPS, "width": WIDTH, "height": HEIGHT, "frames": total,
        "mat": {"px": list(MAT_PX), "mm": list(MAT_BOUNDS), "hue": MAT_HUE,
                "px_per_mm": [PX_PER_MM_X, PX_PER_MM_Y]},
        "markers": {str(i): {"mm": list(mm), "px": list(mmToPx(mm))}
                    for i, mm in MARKER_MM.items()},
        "cover_min": COVER_MIN,
        #Rows all the way down, so what this returns and what lands on disk are one thing
        "script": [[*row[:3], *(list(v) for v in row[3:])] for row in script],
        "beats": [{"script_t": row[0], "kind": b.kind, "name": b.name,
                   "start_frame": round(b.path[0][0] * FPS),
                   "quiet_frame": round(b.t * FPS)}
                  for row, b in zip(script, beats)],
        "frame": records,
    }
    with open(truthPath, "w") as f:
        json.dump(truth, f, indent=1)
    return truth


def main():
    truth = render()
    print(f"{truth['video']} {truth['width']}x{truth['height']} "
          f"{truth['frames']} frames at {FPS} fps "
          f"({truth['frames'] / FPS:.1f}s), truth in {TRUTH_PATH}")
    for b in truth["beats"]:
        print(f"  t={b['script_t']:>5.1f}  {b['kind']:<6} {b['name']:<4} "
              f"frames {b['start_frame']}-{b['quiet_frame']}, quiet from {b['quiet_frame']}")


if __name__ == "__main__":
    main()
