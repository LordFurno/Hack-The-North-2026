import cv2
import os

#Hand-labelled crops for the exemplar banks, captured before perception exists. Nothing
#here is part of the pipeline: hold an object in the box, press its letter, press space.

NAMES = {"a": "phone", "b": "headphones", "p": "pen", "k": "keys"}

CROP_DIR = "crops"
CAM = 0
SIDE_FRAC = 0.35 #Of the shorter side. Square, because the embedder pads square anyway
WINDOW = "tempCapture"


def lockCamera(cap: cv2.VideoCapture):
    #Auto-exposure shifts the instant a hand enters the frame, which shifts every
    #embedding. Crops taken under auto settings will not match the live feed. Silently
    #ignored by cameras that do not expose these, which is the best we can do here.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) #0.25 = manual on most backends
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)


def rectFor(frame) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    side = int(SIDE_FRAC * min(h, w))
    x, y = (w - side) // 2, (h - side) // 2
    return (x, y, x + side, y + side)


def nextPath(name: str) -> str:
    #Auto-increment past what is already there, so a second session keeps counting.
    n = 0
    while True:
        path = os.path.join(CROP_DIR, f"{name}_{n}.png")
        if not os.path.exists(path):
            return path
        n += 1


def main():
    os.makedirs(CROP_DIR, exist_ok=True)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) #DSHOW opens in milliseconds on Windows
    if not cap.isOpened():
        raise SystemExit(f"no camera at index {CAM}")
    lockCamera(cap)

    name, saved = "", 0
    print(" ".join(f"{k}={v}" for k, v in NAMES.items()) + " | space saves | esc quits")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("dropped frame")
            continue

        x0, y0, x1, y1 = rectFor(frame)
        view = frame.copy() #Draw on a copy, so the saved crop carries no overlay
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(view, name or "pick an object", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(WINDOW, view)

        k = cv2.waitKey(1) & 0xFF
        ch = chr(k).lower() if 32 <= k < 127 else ""
        if k == 27:
            break
        if ch in NAMES:
            name = NAMES[ch]
            print(f"object is now {name}")
        elif ch == " ":
            if not name:
                print("pick an object first")
                continue
            path = nextPath(name)
            cv2.imwrite(path, frame[y0:y1, x0:x1])
            saved += 1
            print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"{saved} crop(s) this session, all of them in {CROP_DIR}/")


if __name__ == "__main__":
    main()
