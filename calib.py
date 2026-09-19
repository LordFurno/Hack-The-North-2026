from core import Rect
import math

#The homography half of this file (aruco, px_to_mm, drift) belongs to perception.
#What is here is the pure mat geometry the world model reasons in.

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
