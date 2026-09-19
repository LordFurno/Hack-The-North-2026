from fastapi.testclient import TestClient
import numpy as np
import cv2
import service
import pytest


@pytest.fixture(autouse=True)
def blankPreview():
    service.PREVIEW.update(jpeg=None, ts=0.0, seq=0)
    yield
    service.PREVIEW.update(jpeg=None, ts=0.0, seq=0)


def jpeg(value: int = 120) -> bytes:
    return cv2.imencode(".jpg", np.full((36, 64, 3), value, np.uint8))[1].tobytes()


def test_no_camera_is_said_plainly_rather_than_shown_as_a_blank(  ):
    client = TestClient(service.app)
    assert client.get("/preview.jpg").status_code == 404

    status = client.get("/preview/status").json()
    assert status == {"live": False, "seq": 0, "seconds_since_frame": None}


def test_a_posted_frame_comes_back_out():
    client = TestClient(service.app)
    sent = jpeg()

    posted = client.post("/preview", content=sent,
                         headers={"Content-Type": "image/jpeg"}).json()
    assert posted == {"accepted": True, "seq": 1, "bytes": len(sent)}

    got = client.get("/preview.jpg")
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/jpeg"
    assert got.content == sent
    assert got.headers["cache-control"] == "no-store"


def test_the_newest_frame_wins():
    client = TestClient(service.app)
    client.post("/preview", content=jpeg(10), headers={"Content-Type": "image/jpeg"})
    client.post("/preview", content=jpeg(240), headers={"Content-Type": "image/jpeg"})

    assert client.get("/preview.jpg").content == jpeg(240)
    assert client.get("/preview/status").json()["seq"] == 2


def test_an_empty_body_is_refused():
    #Better a 400 than a zero-byte frame the page would render as a broken image.
    client = TestClient(service.app)
    assert client.post("/preview", content=b"",
                       headers={"Content-Type": "image/jpeg"}).status_code == 400


def test_a_feed_that_stopped_reports_stale_rather_than_live():
    #The page drops the stream when this goes false. A frozen last frame presented as a
    #live desk is the page telling a lie, which is the one thing it must not do.
    client = TestClient(service.app)
    client.post("/preview", content=jpeg(), headers={"Content-Type": "image/jpeg"})
    assert client.get("/preview/status").json()["live"] is True

    service.PREVIEW["ts"] -= service.PREVIEW_STALE_S + 1.0
    status = client.get("/preview/status").json()
    assert status["live"] is False
    assert status["seq"] == 1 #Still says a feed existed, so the page can say "stalled"


def test_a_stream_frame_is_framed_the_way_a_browser_expects():
    #Tested on the framing alone. Opening the endpoint here would hang: the generator is
    #meant to run until the client disconnects, and the test client never does.
    body = service.mjpegFrame(jpeg())

    assert body.startswith(b"--frame\r\n")
    assert b"Content-Type: image/jpeg\r\n" in body
    assert f"Content-Length: {len(jpeg())}".encode() in body
    assert body.endswith(jpeg() + b"\r\n")
    assert "boundary=frame" in service.MJPEG_TYPE #Must match the --frame separator above


def test_the_preview_never_reaches_the_world():
    #Perception writes to the world ONLY via Observation. A picture for a person to look
    #at is not an observation, and must leave no trace in the model.
    client = TestClient(service.app)
    before = (len(service.WORLD.entities), len(service.WORLD.events),
              len(service.WORLD.snapshots))

    client.post("/preview", content=jpeg(), headers={"Content-Type": "image/jpeg"})

    assert (len(service.WORLD.entities), len(service.WORLD.events),
            len(service.WORLD.snapshots)) == before
