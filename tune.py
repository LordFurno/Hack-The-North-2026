from calib import CALIB_PATH, haveCalib, loadCalib, openCamera, solveHomography
from perceive import (ChromaDetector, Config, DetectorError, RefDiffDetector,
                      changedRegions, components, gray, openVideo)
import numpy as np
import cv2, argparse

#Thresholds are the whole difference between a detector that works on your desk and one
#that does not, and guessing at them from event logs is miserable. Watch the mask move
#while you drag the slider, then press s once.

WINDOW = "tune"
HINT = "s saves config.json   r re-takes the reference   esc quits"

#(trackbar label, Config field, maximum, scale). Trackbars are integers, so the one
#fractional threshold is stored per mille and divided back.
BARS = [("pixel_delta", "pixel_delta", 100, 1.0),
        ("motion/1000", "motion_frac", 100, 0.001),
        ("min_object_px", "min_object_px", 8000, 1.0),
        ("ref_delta", "ref_delta", 120, 1.0),
        ("morph_px", "morph_px", 25, 1.0)]


def readBars(cfg: Config) -> Config:
    for label, field, _max, scale in BARS:
        v = cv2.getTrackbarPos(label, WINDOW) * scale
        setattr(cfg, field, v if scale != 1.0 else int(v))
    cfg.morph_px = max(1, cfg.morph_px | 1) #Odd, or the structuring element has no centre
    return cfg


def overlay(frame, mask, regions, boxes) -> np.ndarray:
    view = frame.copy()
    view[mask.astype(bool)] = (0.45 * view[mask.astype(bool)] +
                               0.55 * np.array([60, 220, 60])).astype(np.uint8)
    for r in regions:
        cv2.rectangle(view, (r[0], r[1]), (r[2], r[3]), (140, 140, 140), 1)
    for b in boxes:
        cv2.rectangle(view, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 2)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="tune against a recording instead of the camera")
    ap.add_argument("--calib", default=CALIB_PATH)
    ap.add_argument("--chroma", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    cap = openVideo(args.video) if args.video else openCamera(args.camera)
    first = cap.read()[1]
    if args.video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    H = solveHomography(first)
    reference = first
    if H is None:
        if not haveCalib(args.calib):
            raise SystemExit(f"no markers in view and {args.calib} is missing")
        H, reference = loadCalib(args.calib)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for label, field, top, scale in BARS:
        cv2.createTrackbar(label, WINDOW, int(getattr(cfg, field) / scale), top, lambda _v: None)
    print(HINT)

    prev = first
    while True:
        ok, frame = cap.read()
        if not ok:
            if args.video: #Loop the file, so a short clip is still usable to tune against
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        cfg = readBars(cfg)
        det = (ChromaDetector(cfg, H, frame.shape) if args.chroma
               else RefDiffDetector(reference, cfg))
        regions = changedRegions(prev, frame, cfg)
        try:
            mask, boxes = det.mask(frame), det.detect(frame, regions)
        except DetectorError as e:
            mask, boxes = np.zeros(frame.shape[:2], np.uint8), []
            print(e)

        moving = float((cv2.absdiff(gray(frame), gray(prev)) > cfg.pixel_delta).mean())
        view = overlay(frame, mask, regions, boxes)
        cv2.putText(view, f"motion {moving:.4f} vs {cfg.motion_frac:.3f}   "
                          f"{len(boxes)} object(s), {len(components(mask, 1))} blob(s)",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 255) if moving > cfg.motion_frac else (120, 255, 120), 2)
        cv2.putText(view, HINT, (12, view.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        cv2.imshow(WINDOW, view)

        k = cv2.waitKey(1) & 0xFF
        ch = chr(k).lower() if 32 <= k < 127 else ""
        if k == 27:
            break
        elif ch == "s":
            cfg.save()
            print(f"saved {cfg}")
        elif ch == "r":
            reference = frame.copy()
            print("reference re-taken from the current frame")
        prev = frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
