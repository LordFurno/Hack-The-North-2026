import numpy as np
import cv2, os, sys

from identity import Embedder

#Does the embedder actually tell these objects apart? Point it at a directory of
#hand-captured crops and read the gap. Nothing here is part of the pipeline.

CROP_DIR = "crops"


def loadCrops(dirpath: str) -> list[tuple[str, int, np.ndarray]]: #(object, n, image)
    out = []
    for fn in sorted(os.listdir(dirpath)):
        if not fn.lower().endswith(".png"):
            continue
        obj, _, tail = fn[:-4].rpartition("_")
        if not obj or not tail.isdigit():
            print(f"skipping {fn}, not <object>_<n>.png")
            continue
        img = cv2.imread(os.path.join(dirpath, fn))
        if img is None:
            print(f"skipping {fn}, unreadable")
            continue
        out.append((obj, int(tail), img))
    out.sort(key=lambda t: (t[0], t[1])) #Grouped by object, so the matrix reads in blocks
    return out


def printMatrix(labels: list[str], objs: list[str], sims: np.ndarray):
    w = max(max(len(s) for s in labels), 6) + 2
    print(" " * w + "".join(f"{s:>{w}}" for s in labels))
    for i, row in enumerate(sims):
        if i and objs[i] != objs[i - 1]:
            print() #Blank line between objects: within-object blocks should stand out
        print(f"{labels[i]:>{w}}" + "".join(f"{v:>{w}.3f}" for v in row))


def meanOr(xs: list[float]) -> str:
    return f"{sum(xs) / len(xs):.3f}" if xs else "n/a"


def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else CROP_DIR
    crops = loadCrops(dirpath)
    if len(crops) < 2:
        raise SystemExit(f"{dirpath}/ holds {len(crops)} usable crop(s), need at least 2")

    emb = Embedder()
    print(f"{emb.name} on {emb.device}, {len(crops)} crops, "
          f"{len(set(o for o, _, _ in crops))} objects")

    vecs = [emb(img, (0, 0, img.shape[1], img.shape[0])) for _, _, img in crops]
    objs = [o for o, _, _ in crops]
    labels = [f"{o}_{n}" for o, n, _ in crops]
    sims = np.array(vecs) @ np.array(vecs).T #Vectors are L2-normalised, so this is cosine

    print()
    printMatrix(labels, objs, sims)

    within, between = [], []
    for i in range(len(crops)):
        for j in range(i + 1, len(crops)):
            (within if objs[i] == objs[j] else between).append(float(sims[i, j]))

    gap = (sum(within) / len(within) - sum(between) / len(between)) if within and between else None
    print()
    print(f"within-object   {meanOr(within):>6}  ({len(within)} pairs)")
    print(f"between-object  {meanOr(between):>6}  ({len(between)} pairs)")
    print(f"gap             {f'{gap:.3f}' if gap is not None else 'n/a':>6}")


if __name__ == "__main__":
    main()
