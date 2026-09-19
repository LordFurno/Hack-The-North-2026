from io import StringIO
import json
import start
import pytest


def argsFor(*argv) -> object:
    return start.parser().parse_args(list(argv))


def test_the_default_run_is_camera_and_voice_against_one_service():
    args = argsFor()
    names = [name for name, _argv in start.extras(args, hasKey=True)]

    assert names == ["camera", "voice"]
    assert start.serviceUrl(args) == "http://127.0.0.1:8000"
    assert "--url" in start.cameraArgv(args) and "--url" in start.voiceArgv(args)


def test_every_child_is_told_the_same_port():
    args = argsFor("--port", "8123")
    service, camera, voice = (start.serviceArgv(args), start.cameraArgv(args),
                              start.voiceArgv(args))

    assert service[service.index("--port") + 1] == "8123"
    assert camera[camera.index("--url") + 1] == "http://127.0.0.1:8123"
    assert voice[voice.index("--url") + 1] == "http://127.0.0.1:8123"


def test_a_service_listening_on_everything_is_still_dialled_on_loopback():
    #0.0.0.0 is an address to listen on. A child that tried to connect to it would fail.
    args = argsFor("--host", "0.0.0.0")
    assert start.serviceArgv(args)[start.serviceArgv(args).index("--host") + 1] == "0.0.0.0"
    assert start.serviceUrl(args) == "http://127.0.0.1:8000"


def test_every_child_runs_unbuffered():
    #Their stdout is a pipe, not a terminal. Without -u the tags arrive minutes late.
    args = argsFor()
    for argv in (start.serviceArgv(args), start.cameraArgv(args), start.voiceArgv(args)):
        assert argv[1] == "-u"


def test_fake_replaces_the_camera_with_the_script():
    args = argsFor("--fake", "--speed", "4")
    service = start.serviceArgv(args)

    assert "--fake" in service and service[service.index("--speed") + 1] == "4.0"
    assert [name for name, _ in start.extras(args, hasKey=True)] == ["voice"]


def test_without_a_key_the_voice_is_not_started():
    #voice.py exits on its first line without one, and a process that dies immediately is
    #worse than a printed sentence saying why there is no voice.
    args = argsFor()
    assert [name for name, _ in start.extras(args, hasKey=False)] == ["camera"]


def test_the_switches_that_turn_parts_off():
    assert [n for n, _ in start.extras(argsFor("--no-camera"), True)] == ["voice"]
    assert [n for n, _ in start.extras(argsFor("--no-voice"), True)] == ["camera"]
    assert start.extras(argsFor("--no-camera", "--no-voice"), True) == []


def test_a_recording_goes_to_perception_instead_of_a_camera_index():
    camera = start.cameraArgv(argsFor("--video", "synth.mp4", "--camera", "2"))
    assert camera[camera.index("--video") + 1] == "synth.mp4"
    assert "--camera" not in camera #One source, and the file is the one that was asked for


def test_detector_and_embedder_switches_reach_perception():
    camera = start.cameraArgv(argsFor("--chroma", "--no-embed", "--camera", "1"))
    assert camera[camera.index("--camera") + 1] == "1"
    assert "--chroma" in camera and "--no-embed" in camera


def test_the_wake_phrase_reaches_the_voice():
    assert start.voiceArgv(argsFor())[-1] == start.voice.WAKE_PHRASE #One default, in voice.py
    voiceArgv = start.voiceArgv(argsFor("--wake", "hey desk", "--no-image"))
    assert voiceArgv[voiceArgv.index("--wake") + 1] == "hey desk"
    assert "--no-image" in voiceArgv


# ---- calibration --------------------------------------------------------

def test_a_calibration_is_both_halves_or_neither(tmp_path):
    #calib.json names the reference frame. A homography with no empty desk to subtract is
    #not a calibration, and starting the camera against one is a desk that reads as changed.
    path = tmp_path / "calib.json"
    assert start.calibrated(str(path)) is False #Nothing at all

    path.write_text(json.dumps({"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                "reference": "reference.png"}))
    assert start.calibrated(str(path)) is False #Names a frame that is not there

    (tmp_path / "reference.png").write_bytes(b"\x89PNG")
    assert start.calibrated(str(path)) is True


def test_only_a_run_that_opens_a_camera_has_to_calibrate(monkeypatch):
    monkeypatch.setattr(start, "calibrated", lambda *a: False)

    assert start.needsCalibration(argsFor()) is True #A camera, and no empty desk saved
    assert start.needsCalibration(argsFor("--fake")) is False
    assert start.needsCalibration(argsFor("--no-camera")) is False
    assert start.needsCalibration(argsFor("--video", "synth.mp4")) is False
    assert start.needsCalibration(argsFor("--calibrate", "--fake")) is True #Asked outright

    monkeypatch.setattr(start, "calibrated", lambda *a: True)
    assert start.needsCalibration(argsFor()) is False #Already done, and it is kept
    assert start.needsCalibration(argsFor("--calibrate")) is True #Done, and redone anyway


def test_a_corrupt_calibration_reads_as_none(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text("{not json")
    assert start.calibrated(str(path)) is False


def test_the_reference_is_found_next_to_the_calibration_not_the_shell(tmp_path):
    #Run from anywhere: the children are given the repo as their working directory, so the
    #check has to resolve the frame the same way rather than against wherever this was run.
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"reference": "empty.png"}))
    (tmp_path / "empty.png").write_bytes(b"\x89PNG")
    assert start.calibrated(str(path)) is True


def test_a_port_already_in_use_is_noticed_before_anything_starts(tmp_path):
    #The failure this prevents: the new world model cannot bind, the health check passes
    #against the old one, and the camera posts into a desk nobody is watching.
    import socket
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1) #A backlog of one, never accepted from: a connect test reads this
        port = taken.getsockname()[1] #as free, which is the reason this asks by binding
        assert start.portFree("127.0.0.1", port) is False
        assert start.portFree("127.0.0.1", port) is False #Twice: no state left behind
    assert start.portFree("127.0.0.1", port) is True #Released again


# ---- the terminal -------------------------------------------------------

def test_output_says_which_process_said_it():
    out = StringIO()
    start.pump("camera", ["MOVED  mug 0.95\n", "COVERED  mug 0.90\n"], out)

    assert out.getvalue().splitlines() == ["[camera ] MOVED  mug 0.95",
                                           "[camera ] COVERED  mug 0.90"]
    assert len(start.tag("service")) == len(start.tag("voice")) #Columns line up


class Proc: #poll() answers from a script, and None means still running
    def __init__(self, *codes):
        self.codes = list(codes)

    def poll(self):
        return self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]


def test_perception_crashing_costs_the_camera_and_nothing_else(monkeypatch, capsys):
    #The world model still holds every belief and the dashboard still answers. This is
    #what the process split buys, so the supervisor must not undo it by shutting down.
    monkeypatch.setattr(start, "POLL_S", 0.0)
    children = [start.Child("service", Proc(None, None, 0)),
                start.Child("camera", Proc(1))]

    start.supervise(children, essential="service")
    out = capsys.readouterr().out

    assert out.index("camera exited (1)") < out.index("service exited (0)")
    assert children == [] #Both gone, in that order, and only then did it return


def test_the_world_model_is_the_one_death_that_ends_the_run(monkeypatch):
    monkeypatch.setattr(start, "POLL_S", 0.0)
    children = [start.Child("service", Proc(1)), start.Child("voice", Proc(None))]

    start.supervise(children, essential="service")
    assert [c.name for c in children] == ["voice"] #Returned at once, leaving it to be stopped


def test_ctrl_c_returns_with_the_children_still_listed(monkeypatch, capsys):
    #How a run normally ends. The children are in process groups of their own precisely so
    #the console's Ctrl-C does NOT reach them: it arrives here, and they get stopped in
    #order. Coming back with an empty list would leave every one of them running.
    class Interrupted:
        def poll(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(start, "POLL_S", 0.0)
    children = [start.Child("service", Interrupted()), start.Child("camera", Interrupted())]

    start.supervise(children)
    assert [c.name for c in children] == ["service", "camera"]
    assert "shutting down" in capsys.readouterr().out
