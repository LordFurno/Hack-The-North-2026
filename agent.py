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
EDGE_PX = 6 #How close to the border counts as touching it
HAND_PX = 70 #Half-width of the footprint a hand sweeps, around the far tip


def touchesBorder(box: tuple[int, int, int, int], bounds: tuple[int, int, int, int]) -> bool:
    #`bounds` is the region the detector can see: the whole frame for refdiff, the mat for
    #anything that only understands the surface.
    return (box[0] <= bounds[0] + EDGE_PX or box[1] <= bounds[1] + EDGE_PX
            or box[2] >= bounds[2] - EDGE_PX or box[3] >= bounds[3] - EDGE_PX)


def borderBlob(mask: np.ndarray, bounds: tuple[int, int, int, int],
               minArea: int = MIN_AREA_PX):
    #(blob mask, stats row) for the largest edge-touching component, or None. Anything
    #wholly interior is an object that moved, not something reaching in over the edge.
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    best, bestArea = 0, minArea
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= bestArea and touchesBorder((x, y, x + bw, y + bh), bounds):
            best, bestArea = i, area
    return (lbl == best, stats[best]) if best else None


def enteringEdge(stats, bounds: tuple[int, int, int, int]) -> str:
    #Which border the arm crossed. The one the blob is hard against is the shoulder side.
    x, y, w, h = stats[0], stats[1], stats[2], stats[3]
    gaps = {"left": x - bounds[0], "top": y - bounds[1],
            "right": bounds[2] - (x + w), "bottom": bounds[3] - (y + h)}
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
    #
    #The blob comes from the detector's FOREGROUND, not from the frame-to-frame difference
    #the motion gate built. An arm reaching in mostly translates along its own length, so
    #from one frame to the next it barely differs except at the fingertips, and those
    #crescents touch no border and are too small to be an arm. Against the empty desk the
    #whole limb is one large region crossing the edge, every frame it is in shot.
    def __init__(self, maskOf, bounds: tuple[int, int, int, int],
                 minArea: int = MIN_AREA_PX, handPx: int = HAND_PX):
        self.maskOf = maskOf
        self.bounds = bounds
        self.minArea = minArea
        self.handPx = handPx
        self.path: list[Rect] = []
        self.present = False

    def reset(self):
        self.path, self.present = [], False

    def update(self, frame: np.ndarray, diff: np.ndarray, H: np.ndarray):
        #Called on motion frames only. `diff` is what the gate already computed; the
        #foreground is what actually shows an arm.
        found = borderBlob(self.maskOf(frame), self.bounds, self.minArea)
        if found is None:
            return
        blob, stats = found
        x, y = farTip(blob, enteringEdge(stats, self.bounds))
        self.path.append(rectPxToMm(H, (x - self.handPx, y - self.handPx,
                                        x + self.handPx, y + self.handPx)))

    def swept_mm(self) -> list[Rect]:
        return list(self.path)
