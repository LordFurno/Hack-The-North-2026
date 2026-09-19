from core import Rect
from calib import rectPxToMm
import numpy as np
import cv2

#From overhead over a bounded surface an arm MUST enter from the frame border, so the
#largest edge-touching blob is the agent and the point furthest from the entering edge is
#the hand rather than the elbow. This feeds H3 and nothing else: with the tracker absent or
#broken the system degrades from "you carried it off" to "it left the desk, cause unknown",
#which is a weaker answer, not a wrong one.

MIN_AREA_PX = 4000 #Under this a border blob is a shadow or a sleeve, not an arm
MOTION_DELTA = 25 #Grey difference that counts as moving, same units as Config.pixel_delta
EDGE_PX = 6 #How close to the border counts as touching it
HAND_PX = 70 #Half-width of the footprint a hand sweeps, around the far tip


def touchesBorder(box: tuple[int, int, int, int], shape) -> bool:
    h, w = shape[:2]
    return (box[0] <= EDGE_PX or box[1] <= EDGE_PX
            or box[2] >= w - EDGE_PX or box[3] >= h - EDGE_PX)


def borderBlob(mask: np.ndarray, minArea: int = MIN_AREA_PX):
    #(blob mask, stats row) for the largest edge-touching component, or None. Anything
    #wholly interior is an object that moved, not something reaching in over the edge.
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    best, bestArea = 0, minArea
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= bestArea and touchesBorder((x, y, x + bw, y + bh), mask.shape):
            best, bestArea = i, area
    return (lbl == best, stats[best]) if best else None


def enteringEdge(stats, shape) -> str:
    #Which border the arm crossed. The one the blob is hard against is the shoulder side.
    x, y, w, h = stats[0], stats[1], stats[2], stats[3]
    frameH, frameW = shape[:2]
    gaps = {"left": x, "top": y, "right": frameW - (x + w), "bottom": frameH - (y + h)}
    return min(gaps, key=gaps.get)


def farTip(blob: np.ndarray, edge: str) -> tuple[int, int]:
    #The point of the arm furthest from the edge it came in over: the hand, not the elbow.
    ys, xs = np.nonzero(blob)
    if edge == "left":
        i = int(np.argmax(xs))
    elif edge == "right":
        i = int(np.argmin(xs))
    elif edge == "top":
        i = int(np.argmax(ys))
    else:
        i = int(np.argmin(ys))
    return (int(xs[i]), int(ys[i]))


class AgentTracker:
    #One burst of motion's worth of hand path, reset after every settle. `present` is a
    #different question from anything update() can answer -- it asks whether the hand is
    #STILL there once everything has stopped -- so analyse() sets it from the settled frame.
    def __init__(self, minArea: int = MIN_AREA_PX, handPx: int = HAND_PX,
                 delta: int = MOTION_DELTA):
        self.minArea = minArea
        self.handPx = handPx
        self.delta = delta
        self.path: list[Rect] = []
        self.present = False

    def reset(self):
        self.path, self.present = [], False

    def update(self, frame: np.ndarray, diff: np.ndarray, H: np.ndarray):
        #Called on motion frames only, with the frame-to-frame difference the gate built.
        found = borderBlob(diff > self.delta, self.minArea)
        if found is None:
            return
        blob, stats = found
        x, y = farTip(blob, enteringEdge(stats, diff.shape))
        self.path.append(rectPxToMm(H, (x - self.handPx, y - self.handPx,
                                        x + self.handPx, y + self.handPx)))

    def swept_mm(self) -> list[Rect]:
        return list(self.path)
