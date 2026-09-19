from calib import (CALIB_PATH, MAT_BOUNDS, REFERENCE_PATH, captureReference, clickCorners,
                   findMarkers, openCamera, pxToMm, saveCalib, solveHomography)
import numpy as np
import cv2, argparse

#Everything the system needs from the physical world, in one screen: a homography and a
#frame of the empty desk. The reference frame IS the detector, so it is worth taking care
#over -- clear the desk properly, and take it after the exposure has settled, not before.

WINDOW = "calibrate"
HINT = "r reference  c corners  a aruco  s save  esc quit"


def drawMarkers(view: np.ndarray, seen: dict[int, tuple[float, float]]):
    for i, (x, y) in seen.items():
        cv2.circle(view, (int(x), int(y)), 7, (0, 200, 255), 2)
        cv2.putText(view, str(i), (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def drawMat(view: np.ndarray, H: np.ndarray):
    #The mat bounds pushed back through H. If this quad does not sit on your desk edges,
    #nothing downstream will land where you expect it to.
    x0, y0, x1, y1 = MAT_BOUNDS
    inv = np.linalg.inv(H)
    quad = []
    for mm in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        q = inv @ np.array([mm[0], mm[1], 1.0])
        quad.append((q[0] / q[2], q[1] / q[2]))
    cv2.polylines(view, [np.int32(quad)], True, (0, 255, 0), 2)


def drawStatus(view: np.ndarray, H, reference, cursor):
    ready = "H " + ("ok" if H is not None else "--") + \
            "   reference " + ("ok" if reference is not None else "--")
    cv2.putText(view, HINT, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(view, ready, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (120, 255, 120) if H is not None and reference is not None else (120, 190, 255), 2)
    if H is not None and cursor is not None:
        mm = pxToMm(H, cursor)
        cv2.putText(view, f"{mm[0]:7.1f}, {mm[1]:7.1f} mm", (12, view.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--out", default=CALIB_PATH)
    ap.add_argument("--reference", default=REFERENCE_PATH)
    args = ap.parse_args()

    cap = openCamera(args.camera)
    H, reference = None, None
    pos = [None] #Boxed, because the callback has nowhere else to put it

    cv2.namedWindow(WINDOW)
    #Live mm readout under the pointer: the sanity check is then just moving the mouse.
    cv2.setMouseCallback(WINDOW, lambda ev, x, y, *_: pos.__setitem__(0, (x, y)))

    print(HINT)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("dropped frame")
            continue

        view = frame.copy()
        seen = findMarkers(frame)
        drawMarkers(view, seen)
        if H is not None:
            drawMat(view, H)
        drawStatus(view, H, reference, pos[0])
        cv2.imshow(WINDOW, view)

        k = cv2.waitKey(1) & 0xFF
        ch = chr(k).lower() if 32 <= k < 127 else ""
        if k == 27:
            break
        elif ch == "r":
            reference = captureReference(cap)
            print(f"reference captured, mean brightness {reference.mean():.1f}")
        elif ch == "c":
            try:
                H = clickCorners(frame, WINDOW)
            except KeyboardInterrupt:
                print("cancelled")
                continue
            print("homography from clicked corners")
        elif ch == "a":
            solved = solveHomography(frame)
            if solved is None:
                print(f"saw markers {sorted(seen)}, need all four of 0 1 2 3")
                continue
            H = solved
            print("homography from markers")
        elif ch == "s":
            if H is None or reference is None:
                print("need both an H (c or a) and a reference (r) before saving")
                continue
            saveCalib(H, reference, args.out, args.reference)
            print(f"saved {args.out} and {args.reference}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
