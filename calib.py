from core import Rect
import numpy as np
import cv2, json, math, os, time

#Two halves. Above the rule: the pure desk geometry the world model reasons in, no
#camera anywhere near it. Below it: the homography, the camera and the reference frame,
#which is everything needed to turn pixels into the millimetres the top half expects.

MAT_BOUNDS = (0.0, 0.0, 600.0, 450.0) #x0,y0,x1,y1 mm


def overlapFraction(a: Rect, b: Rect) -> float: #Fraction of a that b covers, asymmetric on purpose
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    return (ix * iy) / areaA if areaA > 0 else 0.0


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def onMat(pt: tuple[float, float]) -> bool:
    x0, y0, x1, y1 = MAT_BOUNDS
    return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


def quadrant(pt: tuple[float, float]) -> str: #Nobody wants to hear millimetres
    x0, y0, x1, y1 = MAT_BOUNDS
    fx, fy = (pt[0] - x0) / (x1 - x0), (pt[1] - y0) / (y1 - y0)
    v = "top" if fy < 0.33 else "bottom" if fy > 0.67 else "middle"
    h = "left" if fx < 0.33 else "right" if fx > 0.67 else "centre"
    return "in the centre" if (v, h) == ("middle", "centre") else f"toward the {v} {h}"


# ---- the camera ---------------------------------------------------------

CALIB_PATH = "calib.json"
REFERENCE_PATH = "reference.png"

CAM_WIDTH, CAM_HEIGHT = 1280, 720
EXPOSURE = -6 #A starting point only. What is right depends entirely on the room: on one
              #camera here -6 lands at mean 21 of 255 and -4 at 69. calibrate.py lets you
              #pick, and saves the choice, because a reference frame taken at one exposure
              #is worthless against frames captured at another.
SETTLE_READS = 15 #Frames to throw away while whatever is still automatic finds its level
AUTOFOCUS_S = 2.0 #Long enough for the lens to hunt and land before we freeze it


def sharpness(frame: np.ndarray) -> float:
    #Variance of the Laplacian. Only comparable between shots of the same scene, which is
    #all it is ever used for: is this frame crisper than the last one.
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def focusThenLock(cap: cv2.VideoCapture, seconds: float = AUTOFOCUS_S):
    #Turning autofocus off does NOT focus the lens, it freezes it wherever it was parked
    #-- which on this camera was a setting eight times blurrier than the sharp one. Let it
    #hunt first, then lock it there, so the picture is both sharp AND stable.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    t0 = time.time()
    while time.time() - t0 < seconds:
        cap.read()
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) #Freeze it where the hunt ended


def openCamera(index: int = 0, width: int = CAM_WIDTH, height: int = CAM_HEIGHT,
               exposure: float | None = EXPOSURE,
               focus: float | None = None) -> cv2.VideoCapture:
    #Auto anything is a slow leak into the detector: exposure drift fills refdiff with
    #noise, and a lens that hunts shifts every embedding. These are silently ignored by
    #plenty of backends -- CAP_PROP_AUTO_EXPOSURE reads back -1 on both cameras here --
    #which is exactly why level() exists as well. `exposure=None` leaves the camera's own
    #choice alone; `focus=None` means autofocus once, then lock.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) #DSHOW opens in milliseconds on Windows
    if not cap.isOpened():
        raise SystemExit(f"no camera at index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) #0.25 = manual on most backends
    if exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    if focus is None:
        focusThenLock(cap)
    else:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus)
    for _ in range(SETTLE_READS):
        cap.read()
    return cap


def cameraSettings(cap: cv2.VideoCapture, index: int) -> dict:
    #What live.py has to reproduce for the saved reference frame to mean anything.
    return {"index": index,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "focus": cap.get(cv2.CAP_PROP_FOCUS)}


def level(ref: np.ndarray, now: np.ndarray) -> np.ndarray:
    #Match the reference's brightness to the current frame before diffing.
    return np.clip(ref.astype(np.float32) * (now.mean() / max(ref.mean(), 1.0)),
                   0, 255).astype(np.uint8)


def captureReference(cap: cv2.VideoCapture, n: int = 10) -> np.ndarray:
    #The median of a few frames, because this one image IS the detector and a single
    #noisy grab would be baked into every settle for the rest of the run.
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


# ---- the homography -----------------------------------------------------

def clickCorners(frame: np.ndarray, window: str = "calib") -> np.ndarray:
    #Click 4 desk corners clockwise from top-left. Declares them MAT_BOUNDS: the quad
    #need not really be 600x450, it only has to be consistent, because every threshold
    #in the system is expressed in those units.
    pts = []
    cv2.imshow(window, frame)
    cv2.setMouseCallback(window, lambda ev, x, y, *_:
                         pts.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while len(pts) < 4:
        view = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 6, (0, 255, 0), -1)
            cv2.putText(view, str(i), (p[0] + 9, p[1] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window, view)
        if cv2.waitKey(50) == 27:
            raise KeyboardInterrupt("corner clicking cancelled")
    cv2.setMouseCallback(window, lambda *_: None)

    x0, y0, x1, y1 = MAT_BOUNDS
    dst = np.float32([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    H, _ = cv2.findHomography(np.float32(pts), dst)
    return H


def pxToMm(H: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    q = H @ np.array([pt[0], pt[1], 1.0])
    return (float(q[0] / q[2]), float(q[1] / q[2]))


def rectPxToMm(H: np.ndarray, box: tuple[int, int, int, int]) -> Rect:
    #A rotated homography turns a pixel rectangle into a quad, and the world model only
    #speaks axis-aligned mm, so take the bounding box of all four corners rather than
    #two of them: with any tilt at all, opposite corners alone understate the region.
    xs, ys = zip(*(pxToMm(H, p) for p in ((box[0], box[1]), (box[2], box[1]),
                                          (box[2], box[3]), (box[0], box[3]))))
    return (min(xs), min(ys), max(xs), max(ys))


# ---- markers, the upgrade -----------------------------------------------

MAT_MM = { #Marker id -> mm, clockwise from the origin corner. synth.MARKER_MM agrees
    0: (MAT_BOUNDS[0], MAT_BOUNDS[1]),
    1: (MAT_BOUNDS[2], MAT_BOUNDS[1]),
    2: (MAT_BOUNDS[2], MAT_BOUNDS[3]),
    3: (MAT_BOUNDS[0], MAT_BOUNDS[3]),
}

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_DETECTOR = cv2.aruco.ArucoDetector(_DICT)


def findMarkers(frame: np.ndarray) -> dict[int, tuple[float, float]]: #id -> centre px
    corners, ids, _ = _DETECTOR.detectMarkers(frame)
    if ids is None:
        return {}
    return {int(i): tuple(c[0].mean(axis=0)) for c, i in zip(corners, ids.flatten())}


def solveHomography(frame: np.ndarray) -> np.ndarray | None:
    #Called on every settle, this makes a bumped camera cost nothing. Fewer than four
    #markers is not an error, it is a frame with a hand over one: keep the previous H.
    seen = findMarkers(frame)
    pts = [(seen[i], MAT_MM[i]) for i in MAT_MM if i in seen]
    if len(pts) < 4:
        return None
    return cv2.findHomography(np.float32([p for p, _ in pts]),
                              np.float32([m for _, m in pts]))[0]


# ---- saved calibration --------------------------------------------------

def saveCalib(H: np.ndarray, reference: np.ndarray, path: str = CALIB_PATH,
              refPath: str = REFERENCE_PATH, camera: dict | None = None):
    cv2.imwrite(refPath, reference)
    with open(path, "w") as f:
        json.dump({"H": np.asarray(H).tolist(), "mat_bounds": list(MAT_BOUNDS),
                   "reference": refPath, "camera": camera or {}}, f, indent=1)


def loadCalib(path: str = CALIB_PATH) -> tuple[np.ndarray, np.ndarray]: #(H, reference)
    with open(path) as f:
        data = json.load(f)
    refPath = data.get("reference", REFERENCE_PATH)
    reference = cv2.imread(refPath)
    if reference is None:
        raise SystemExit(f"{path} points at {refPath}, which is missing. Run calibrate.py")
    if list(data.get("mat_bounds", MAT_BOUNDS)) != list(MAT_BOUNDS):
        print(f"warning: {path} was saved against mat bounds {data['mat_bounds']}, "
              f"code says {list(MAT_BOUNDS)}. Every threshold is in those units")
    return np.array(data["H"], dtype=float), reference


def loadCamera(path: str = CALIB_PATH) -> dict:
    #The camera the reference frame was taken with. Reopening at a different exposure
    #makes the whole desk read as changed, which is the worst failure the system has.
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get("camera") or {}


def haveCalib(path: str = CALIB_PATH) -> bool:
    return os.path.exists(path)
