from core import Event, Observation, Status
from calib import CALIB_PATH, haveCalib, loadCalib, solveHomography
from world import World
from resolve import settle
from identity import match
from agent import AgentTracker
from perceive import (ChromaDetector, Config, RefDiffDetector, openVideo, run, videoClock)
import numpy as np
import cv2, argparse, json, os, time

#The same settle loop over a recorded file, with a picture of what it thought per settle.
#Record the demo once and every later iteration is this script instead of performing it
#again, which is the difference between tuning ten times and tuning twice.

OUT_DIR = "replay"

COLOURS = {Status.VISIBLE: (120, 240, 140), Status.HIDDEN: (250, 190, 130),
           Status.UNRESOLVED: (90, 200, 250), Status.OFF_DESK: (160, 160, 160)}


def mmToPxVia(Hinv: np.ndarray, pt: tuple[float, float]) -> tuple[int, int]:
    q = Hinv @ np.array([pt[0], pt[1], 1.0])
    return (int(q[0] / q[2]), int(q[1] / q[2]))


def drawRect(view, Hinv, rect, colour, thickness=2):
    x0, y0, x1, y1 = rect
    quad = [mmToPxVia(Hinv, p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    cv2.polylines(view, [np.int32(quad)], True, colour, thickness)
    return quad


def annotate(frame, world: World, obs: Observation, events: list[Event],
             H: np.ndarray, scores: dict[int, str]) -> np.ndarray:
    #Everything the settle saw and everything it concluded, on one image.
    view = frame.copy()
    Hinv = np.linalg.inv(H)

    for r in obs.changed: #What was examined at all
        drawRect(view, Hinv, r, (90, 90, 90), 1)
    for r in obs.agent_swept:
        drawRect(view, Hinv, r, (60, 140, 220), 1)

    for i, d in enumerate(obs.detections):
        w, h = d.size
        rect = (d.centroid[0] - w / 2, d.centroid[1] - h / 2,
                d.centroid[0] + w / 2, d.centroid[1] + h / 2)
        quad = drawRect(view, Hinv, rect, (255, 255, 255), 2)
        cv2.putText(view, scores.get(i, "?"), (quad[0][0], quad[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    for e in world.entities.values(): #And what it now believes, which is the point
        if e.isAgent or e.footprint == (0.0, 0.0):
            continue
        colour = COLOURS.get(e.status, (200, 200, 200))
        quad = drawRect(view, Hinv, world.footprintRect(e), colour, 2)
        cv2.putText(view, f"{e.label}:{e.id[:4]} {e.status.value[:3]}",
                    (quad[3][0], quad[3][1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    y = 30
    cv2.putText(view, f"{obs.frame_ref}  t={obs.ts:.2f}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for ev in events:
        y += 26
        bad = ev.kind.value == "BELIEF_FALSIFIED"
        cv2.putText(view, f"{ev.kind.value} {ev.note} ({ev.confidence:.2f})", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (80, 80, 255) if bad else (120, 240, 140), 2)
    return view


def matchReport(world: World, obs: Observation) -> dict[int, str]:
    #Scored BEFORE settle() runs, because settle is what changes the answer.
    out = {}
    for i, d in enumerate(obs.detections):
        m = match(world, d)
        if m.entity is None:
            out[i] = f"new {m.score:.2f}"
        else:
            out[i] = f"{m.entity.label}:{m.entity.id[:4]} {m.score:.2f}" + \
                     ("?" if m.ambiguous else "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="synth.mp4")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--calib", default=CALIB_PATH)
    ap.add_argument("--chroma", action="store_true", help="hue detector, for a coloured mat")
    ap.add_argument("--no-agent", action="store_true")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    os.makedirs(args.out, exist_ok=True)
    cap = openVideo(args.video)

    #A file with markers in it needs no calibration at all: solve H off its first frame.
    first = cap.read()[1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    H = solveHomography(first)
    reference = first
    if H is None:
        if not haveCalib(args.calib):
            raise SystemExit(f"{args.video} shows no markers and {args.calib} is missing")
        H, reference = loadCalib(args.calib)
        print(f"no markers in {args.video}, using {args.calib}")
    else:
        print(f"homography from the markers in {args.video}")

    det = (ChromaDetector(cfg, H, reference.shape) if args.chroma
           else RefDiffDetector(reference, cfg))
    agent = None if args.no_agent else AgentTracker(det.mask, det.bounds)

    embed = None
    if not args.no_embed:
        from identity import Embedder
        embed = Embedder()
        print(f"{embed.name} on {embed.device}")

    world = World()
    logPath = os.path.join(args.out, "events.jsonl")
    log = open(logPath, "w")
    n = [0]

    def sink(obs: Observation):
        scores = matchReport(world, obs)
        events = settle(world, obs, cfg.confirm_settles)
        #The keyframe analyse() just wrote is by definition the frame this was believed
        #from, so annotate that rather than whatever the capture has moved on to.
        path = os.path.join(args.out, f"{n[0]:03d}.jpg")
        cv2.imwrite(path, annotate(cv2.imread(obs.frame_ref), world, obs, events, H, scores))
        for ev in events:
            log.write(json.dumps({"settle": n[0], "ts": ev.ts, "entity": ev.entity,
                                  "label": world.entities[ev.entity].label,
                                  "kind": ev.kind.value, "cause": ev.cause,
                                  "confidence": ev.confidence, "note": ev.note,
                                  "frame_ref": ev.frame_ref}) + "\n")
        print(f"{path}  {len(obs.detections)} det  " +
              (", ".join(f"{ev.kind.value} {world.entities[ev.entity].label}"
                         for ev in events) or "nothing"))
        n[0] += 1

    run(cap, det, sink, cfg, H, agent=agent, embed=embed,
        clock=videoClock(cap, time.time()))
    log.close()
    cap.release()
    print(f"{n[0]} settle(s), {args.out}/*.jpg and {logPath}")


if __name__ == "__main__":
    main()
