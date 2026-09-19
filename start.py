from calib import CALIB_PATH, REFERENCE_PATH
from dataclasses import dataclass
import omni
import voice
import argparse, json, os, signal, socket, subprocess, sys, threading, time
import urllib.error, urllib.request, webbrowser

#One command for the whole desk. Calibration first -- the reference frame IS the detector,
#so nothing downstream means anything until it exists -- then the world model, then the
#camera, then the voice, each started only once the thing it talks to is answering.
#
#The parts stay separate processes, which is not ceremony: Observation crossing a socket is
#what lets perception die and come back without the world model losing a belief. All this
#file does is start them in order, tag their output so one terminal stays readable, and
#shut them down the other way round.

HERE = os.path.dirname(os.path.abspath(__file__))
HEALTH_TIMEOUT_S = 40.0 #Generous: the first import of torch on a cold machine is not quick
STOP_GRACE_S = 6.0
POLL_S = 0.5

#A child in its own process group is one the console's Ctrl-C cannot reach, which is the
#point: shutdown happens here, in order. CTRL_BREAK is then the only signal Windows will
#deliver to it, and Python turns it into the KeyboardInterrupt each child already handles.
BREAK = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT)
NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


@dataclass
class Child:
    name: str
    proc: subprocess.Popen


def tag(name: str) -> str:
    return f"[{name:<7}] "


# ---- what each part is told ---------------------------------------------

def clientHost(host: str) -> str:
    #What a child should dial. 0.0.0.0 is an address to listen on, not one to connect to.
    return "127.0.0.1" if host in ("0.0.0.0", "::", "") else host


def serviceUrl(args) -> str:
    return f"http://{clientHost(args.host)}:{args.port}"


#-u on every child because a pipe is not a terminal: without it their prints sit in a
#buffer for minutes and the one thing this file exists for stops working.

def serviceArgv(args) -> list[str]:
    argv = [sys.executable, "-u", "service.py", "--host", args.host, "--port", str(args.port)]
    return argv + ["--fake", "--speed", str(args.speed)] if args.fake else argv


def cameraArgv(args) -> list[str]:
    argv = [sys.executable, "-u", "live.py", "--url", serviceUrl(args)]
    if args.video:
        argv += ["--video", args.video]
    elif args.camera is not None:
        argv += ["--camera", str(args.camera)]
    if args.chroma:
        argv.append("--chroma")
    if args.no_embed:
        argv.append("--no-embed")
    return argv


def voiceArgv(args) -> list[str]:
    argv = [sys.executable, "-u", "voice.py", "--url", serviceUrl(args), "--wake", args.wake]
    return argv + ["--no-image"] if args.no_image else argv


def extras(args, hasKey: bool) -> list[tuple[str, list[str]]]:
    #What starts once the world model answers. --fake replaces the camera with a script,
    #and a missing API key replaces the voice with a printed line rather than a process
    #that exits on its first breath.
    out = []
    if not (args.fake or args.no_camera):
        out.append(("camera", cameraArgv(args)))
    if not args.no_voice and hasKey:
        out.append(("voice", voiceArgv(args)))
    return out


# ---- calibration --------------------------------------------------------

def calibrated(path: str = CALIB_PATH) -> bool:
    #Both halves or neither: calib.json names the reference frame, and a homography with
    #no empty desk to subtract is not a calibration, it is a file.
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            ref = json.load(f).get("reference", REFERENCE_PATH)
    except (OSError, ValueError):
        return False
    return os.path.exists(os.path.join(os.path.dirname(os.path.abspath(path)), ref))


def needsCalibration(args) -> bool:
    #Only a run that will really open a camera needs an empty desk to subtract: --fake has
    #no camera, --no-camera has no camera, and a recording often carries its own markers.
    #--calibrate asks for it outright and gets it either way.
    return bool(args.calibrate) or not (args.fake or args.no_camera or args.video
                                        or calibrated())


def calibrate(args) -> bool:
    #Interactive: a window, a live preview and keys. It gets the terminal to itself and
    #everything else waits, because everything else is meaningless without what it writes.
    argv = [sys.executable, "calibrate.py"]
    if args.camera is not None:
        argv += ["--camera", str(args.camera)]

    print(f"\n{tag('start')}calibration. Clear the desk, then:")
    print(f"{tag('')}  c  click the four desk corners, clockwise from top left")
    print(f"{tag('')}  r  capture the empty desk -- this frame IS the detector")
    print(f"{tag('')}  s  save, then esc to carry on\n")
    subprocess.call(argv, cwd=HERE)
    return calibrated()


# ---- children -----------------------------------------------------------

def pump(name: str, stream, out=sys.stdout):
    #Three processes into one terminal, each line saying who said it.
    for line in stream:
        out.write(tag(name) + line)
        out.flush()


def spawn(name: str, argv: list[str]) -> Child:
    proc = subprocess.Popen(argv, cwd=HERE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            errors="replace", creationflags=NEW_GROUP,
                            start_new_session=(os.name != "nt"))
    threading.Thread(target=pump, args=(name, proc.stdout), daemon=True).start()
    print(tag("start") + f"{name}: {' '.join(argv[2:])}")
    return Child(name, proc)


def portFree(host: str, port: int) -> bool:
    #Checked before anything starts, because a world model left over from the last run is
    #worse than none at all: the new one fails to bind, the health check passes against the
    #old one, and the camera spends the demo posting into a desk nobody is looking at.
    #
    #Asked by trying to bind, which is the question the service is about to ask, and on the
    #same address. Connecting answers a different one -- a listener whose backlog is full
    #neither accepts nor refuses, and would read as free.
    with socket.socket() as s:
        try:
            s.bind((host or "0.0.0.0", port))
        except OSError:
            return False
    return True


def waitForService(url: str, child: Child, timeout: float = HEALTH_TIMEOUT_S) -> bool:
    #Poll the endpoint rather than sleeping a guessed number of seconds: both of the other
    #two refuse to start against a world model that is not answering yet.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if child.proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{url}/state", timeout=1.0) as r:
                json.loads(r.read())
            return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(POLL_S)
    return False


def stop(child: Child, grace: float = STOP_GRACE_S):
    #On POSIX this is SIGINT and the child runs the KeyboardInterrupt handler it already
    #has. On Windows CTRL_BREAK is the only signal that reaches a child in its own group,
    #and Python does not turn that one into a KeyboardInterrupt -- the child is terminated
    #without its finally blocks. Nothing is lost that matters: the camera and microphone
    #are handles the OS reclaims, and every belief is in the other process anyway.
    if child.proc.poll() is not None:
        return
    try:
        child.proc.send_signal(BREAK)
    except (OSError, ValueError):
        pass
    try:
        child.proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        child.proc.kill()


def supervise(children: list[Child], essential: str = "service"):
    #The world model going down takes the rest with it, because nothing else has anywhere
    #to write. Anything else dying costs its own job and nothing else -- which is the whole
    #point of the split: a camera process that crashed has not cost a single belief, and
    #the dashboard still answers with everything it knew a second ago.
    try:
        while children:
            time.sleep(POLL_S)
            for child in list(children):
                code = child.proc.poll()
                if code is None:
                    continue
                children.remove(child)
                print(tag("start") + f"{child.name} exited ({code})")
                if child.name == essential:
                    return
    except KeyboardInterrupt:
        print("\n" + tag("start") + "shutting down")


# ---- the access point ---------------------------------------------------

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="calibrate, then run the whole desk: world model, camera and voice")
    ap.add_argument("--camera", type=int, help="camera index, for calibration and perception")
    ap.add_argument("--video", help="run perception over a recording instead of the camera")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fake", action="store_true",
                    help="scripted observations instead of a camera; needs no calibration")
    ap.add_argument("--speed", type=float, default=1.0, help="--fake script speed")
    ap.add_argument("--calibrate", action="store_true",
                    help="recalibrate even if calib.json is already there")
    ap.add_argument("--wake", default=voice.WAKE_PHRASE, help="say this, then the question")
    ap.add_argument("--chroma", action="store_true", help="chroma detector instead of refdiff")
    ap.add_argument("--no-embed", action="store_true", help="histograms instead of DINO")
    ap.add_argument("--no-image", action="store_true",
                    help="do not send the camera frame with a spoken question")
    ap.add_argument("--no-camera", action="store_true", help="world model and voice only")
    ap.add_argument("--no-voice", action="store_true", help="no microphone, no Omni voice calls")
    ap.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    return ap


def main():
    args = parser().parse_args()
    url = serviceUrl(args)

    if not portFree(args.host, args.port):
        raise SystemExit(f"something is already listening on {args.port} -- an earlier "
                         f"start.py? Stop it, or run with --port")

    if needsCalibration(args) and not calibrate(args):
        raise SystemExit("nothing was saved, so there is no empty desk to detect against. "
                         "Run calibrate.py and press c, then r, then s")

    children = [spawn("service", serviceArgv(args))]
    try: #Anything at all from here on still takes the children down with it. A Ctrl-C
        #during the health check would otherwise leave a world model holding the port.
        if not waitForService(url, children[0]):
            raise SystemExit(f"the world model never answered at {url}")
        print(tag("start") + f"dashboard at {url}")
        if not args.no_browser:
            webbrowser.open(url)

        hasKey = omni.apiKey() is not None
        if not args.no_voice and not hasKey:
            print(tag("start") + f"no {omni.API_KEY_ENV}: no voice, and every new entity "
                                 f"will stay 'unknown'. Put the key in .env")
        elif not args.no_voice:
            print(tag("start") + f'say "{args.wake}", then your question')

        for name, argv in extras(args, hasKey):
            children.append(spawn(name, argv))
        supervise(children)
    finally:
        for child in reversed(children):
            stop(child)
        print(tag("start") + "stopped")


if __name__ == "__main__":
    main()
