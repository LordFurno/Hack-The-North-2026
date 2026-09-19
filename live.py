from core import Observation
from calib import (CALIB_PATH, CAM_WIDTH, CAM_HEIGHT, EXPOSURE, haveCalib, loadCalib,
                   loadCamera, openCamera, solveHomography)
from world import World
from resolve import settle
from agent import AgentTracker
from perceive import (Config, RefDiffDetector, ChromaDetector, SNAP_DIR,
                      observationJson, openVideo, run, videoClock)
import omni
import cv2, argparse, json, threading, time, urllib.error, urllib.request

#The whole pipeline, pointed at a camera. Perception runs here and the world model may be
#in another process entirely -- Observation is the only thing crossing the wire, which is
#what lets the detector change without the reasoning code noticing.

SERVICE_URL = "http://127.0.0.1:8000"


def postObservation(url: str, obs: Observation) -> list[dict]:
    req = urllib.request.Request(f"{url.rstrip('/')}/observation",
                                 data=json.dumps(observationJson(obs)).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10.0) as r:
        return json.loads(r.read()).get("events", [])


def eventLine(ev: dict) -> str:
    return (f"{ev['kind']:<16} {ev.get('label') or ev['entity']:<12} "
            f"{ev['confidence']:.2f}  {ev['note']}  [{ev['cause']}]")


def httpSink(url: str):
    def sink(obs: Observation):
        try:
            events = postObservation(url, obs)
        except (urllib.error.URLError, OSError) as err:
            print(f"service unreachable ({err}), observation dropped")
            return
        report(obs, [eventLine(ev) for ev in events])
    return sink


class PreviewPoster:
    #The camera feed, for a human watching the dashboard. Newest frame wins and the rest
    #are dropped: the settle loop hands a frame over and returns immediately, because a
    #socket write in that loop is dropped frames and a settle that never fires.
    def __init__(self, url: str, fps: float = 8.0, width: int = 640, quality: int = 60):
        self.url = f"{url.rstrip('/')}/preview"
        self.interval = 1.0 / fps
        self.width = width
        self.quality = quality
        self.latest = None
        self.ready = threading.Event()
        self.sent = self.failed = 0
        self.stop = False
        threading.Thread(target=self.pump, daemon=True).start()

    def __call__(self, frame): #What perceive.run calls, once per frame
        self.latest = frame #One assignment, so the pump either gets this one or the next
        self.ready.set()

    def encode(self, frame) -> bytes:
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else b""

    def pump(self):
        while not self.stop:
            self.ready.wait()
            self.ready.clear()
            frame = self.latest
            if frame is None:
                continue
            try:
                jpeg = self.encode(frame)
                if jpeg:
                    req = urllib.request.Request(
                        self.url, data=jpeg, headers={"Content-Type": "image/jpeg"})
                    urllib.request.urlopen(req, timeout=5.0).close()
                    self.sent += 1
            except (urllib.error.URLError, OSError):
                self.failed += 1 #The dashboard is optional; perception carries on regardless
            time.sleep(self.interval) #Rate limit here, not in the capture loop


def localSink(world: World, cfg: Config, lock: threading.Lock):
    #Same code path, no HTTP. The world model lives in this process instead.
    def sink(obs: Observation):
        with lock:
            events = settle(world, obs, cfg.confirm_settles)
        report(obs, [f"{ev.kind.value:<16} {world.entities[ev.entity].label:<12} "
                     f"{ev.confidence:.2f}  {ev.note}  [{ev.cause}]" for ev in events])
    return sink


def report(obs: Observation, lines: list[str]):
    print(f"-- settle {obs.frame_ref}  {len(obs.detections)} detection(s), "
          f"{len(obs.changed)} changed region(s)"
          + (", hand in frame" if obs.agent_present else ""))
    for line in lines or ["(nothing to explain)"]:
        print("   " + line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None,
                    help="camera index; default is whatever calibrate.py used")
    ap.add_argument("--video", help="run on a file instead of a camera, same code path")
    ap.add_argument("--calib", default=CALIB_PATH)
    ap.add_argument("--snaps", default=SNAP_DIR, help="where keyframes and crops land")
    ap.add_argument("--url", default=SERVICE_URL)
    ap.add_argument("--local", action="store_true", help="skip HTTP, settle in-process")
    ap.add_argument("--chroma", action="store_true", help="hue detector, for a coloured mat")
    ap.add_argument("--no-agent", action="store_true", help="stub the agent, H3 goes quiet")
    ap.add_argument("--no-embed", action="store_true",
                    help="colour histograms instead of DINO, for a machine with no torch")
    ap.add_argument("--no-preview", action="store_true",
                    help="stop pushing the camera feed to the dashboard")
    ap.add_argument("--preview-fps", type=float, default=8.0)
    ap.add_argument("--preview-width", type=int, default=640)
    args = ap.parse_args()

    cfg = Config.load()
    #Reopen the camera exactly as calibrate.py left it. A different exposure against the
    #saved reference frame makes every pixel on the desk read as changed.
    cam = loadCamera(args.calib)
    if args.video:
        cap = openVideo(args.video)
    else:
        cap = openCamera(args.camera if args.camera is not None else cam.get("index", 0),
                         width=cam.get("width", CAM_WIDTH),
                         height=cam.get("height", CAM_HEIGHT),
                         exposure=cam.get("exposure", EXPOSURE))
        print(f"camera {cam.get('index', args.camera)} at "
              f"{cam.get('width', CAM_WIDTH)}x{cam.get('height', CAM_HEIGHT)}, "
              f"exposure {cam.get('exposure', EXPOSURE)}")

    #Markers first: a recording that shows four of them needs no calibration at all, and
    #on a live camera they make a bumped rig free. Clicked corners are the fallback.
    first = cap.read()[1]
    if args.video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) #Rewind past the frame we just peeked at
    H, reference = solveHomography(first), first

    if H is None or not args.video:
        if not haveCalib(args.calib):
            raise SystemExit(f"no markers in view and {args.calib} is missing. "
                             f"Run calibrate.py")
        saved, reference = loadCalib(args.calib)
        H = saved if H is None else H #Markers beat a saved H when both are available
    clock = videoClock(cap, time.time()) if args.video else time.time

    det = (ChromaDetector(cfg, H, reference.shape) if args.chroma
           else RefDiffDetector(reference, cfg))
    agent = None if args.no_agent else AgentTracker(det.mask, det.bounds)

    embed = None
    if not args.no_embed:
        from identity import Embedder #Costs a torch import, so only when it is wanted
        embed = Embedder()
        print(f"{embed.name} on {embed.device}")

    if args.local:
        world, lock = World(), threading.Lock() #One lock: the labeller writes what settle reads
        omni.attach(world, lock)
        sink = localSink(world, cfg, lock)
    else:
        sink = httpSink(args.url)
        print(f"posting observations to {args.url}")

    #The feed goes to the service whether or not the world model lives there, so --local
    #still fills the pane. It is a picture for a person, not an input to anything.
    preview = None
    if not args.no_preview:
        preview = PreviewPoster(args.url, args.preview_fps, args.preview_width)
        print(f"camera feed to {args.url}/preview at {args.preview_fps:g} fps")

    try:
        run(cap, det, sink, cfg, H, agent=agent, embed=embed, clock=clock,
            root=args.snaps, preview=preview)
    except KeyboardInterrupt:
        pass
    finally:
        if preview is not None:
            preview.stop = True
            print(f"camera feed: {preview.sent} frame(s) posted, {preview.failed} failed")
        cap.release()


if __name__ == "__main__":
    main()
