from core import (AGENT_ID, DESK, Detection, Entity, Event, EventKind,
                  Observation, Relation, Status)
from calib import MAT_BOUNDS, quadrant
from world import World
from resolve import settle
from perceive import Config, SNAP_DIR
import fake
import omni

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
import numpy as np
import argparse, asyncio, os, re, threading, time
import uvicorn

#Perception is the only writer, everything else reads. Phrasing is the model's job,
#grounding is this file's job, so every tool returns compact JSON and never prose.

HISTORY_LIMIT = 20
STREAM_HZ = 5.0 #State is small enough to re-send whole, so decay is watchable
UI_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
SNAP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), SNAP_DIR)

PREVIEW_HZ = 10.0 #Ceiling on what the mjpeg stream pushes, whatever arrives
PREVIEW_STALE_S = 3.0 #After this with nothing posted, say so rather than showing a lie

PREP = {Relation.IN: "in", Relation.UNDER: "under",
        Relation.ON: "on", Relation.HELD: "held by"}

#COVERED counts as movement: the thing itself sat still, but something was put on it,
#and its cause names the occluder, which is the answer the user wants.
MOVEMENT_KINDS = (EventKind.MOVED, EventKind.PICKED_UP, EventKind.PUT_DOWN,
                  EventKind.LEFT_DESK, EventKind.COVERED)

STOP_WORDS = {"the", "a", "an", "my", "our", "your", "this", "that", "those",
              "these", "where", "what", "who", "is", "in", "on", "it"}

WORLD = World()
LOCK = threading.Lock() #The fake feed and out-of-process perception write from other threads
CONFIG = Config.load() #Only confirm_settles matters on this side of the wire

#The camera feed, for a person to look at. Deliberately NOT part of the world: nothing
#reads it, nothing reasons about it, and losing it costs a pane on a page and nothing
#else. Observation remains the only thing perception writes into the model.
PREVIEW = {"jpeg": None, "ts": 0.0, "seq": 0}

app = FastAPI(title="spatial memory")
omni.attach(WORLD, LOCK) #No API key, no labelling, no difference to anything else


# ---- phrasing -----------------------------------------------------------

def phraseLocation(world: World, e: Entity) -> str:
    #Walk the parent chain, deepest first. Nobody wants to hear millimetres.
    if e.relation is Relation.HELD and e.parent == AGENT_ID:
        return "in your hand"

    parts, node = [], e
    while node.parent and node.parent != DESK:
        par = world.entities[node.parent]
        parts.append(f"{PREP[node.relation]} the {par.label}")
        node = par

    tail = "on the desk" if node.parent == DESK else "off the desk"
    if not parts:
        return f"on the desk, {quadrant(world.absolute(e))}" if node.parent == DESK else tail
    return ", ".join(parts) + ", " + tail


def resolveReferent(world: World, query: str) -> Entity | None:
    #Substring match on the label, most recently confirmed first. The hard referents
    #("the one I just had") are Omni's job: it sees the scene and asks by id or label.
    q = query.strip().lower()
    if not q:
        return None
    if q in world.entities:
        return world.entities[q] #Ids come straight back from a previous tool call

    words = [w for w in re.split(r"[^a-z0-9]+", q) if w and w not in STOP_WORDS]
    cands = []
    for e in world.entities.values():
        if e.isAgent:
            continue
        lab = e.label.lower()
        if lab == q or lab in words:
            rank = 0
        elif any(w in lab or lab in w for w in words):
            rank = 1
        elif lab in q:
            rank = 2
        else:
            continue
        cands.append((rank, -e.last_confirmed, e))

    if not cands:
        return None
    cands.sort(key=lambda t: t[:2]) #Never compare the entities themselves
    return cands[0][2]


# ---- json ---------------------------------------------------------------

def entityJson(world: World, e: Entity, now: float) -> dict:
    x, y = world.absolute(e)
    return {"id": e.id, "label": e.label, "status": e.status.value,
            "parent": e.parent, "relation": e.relation.value,
            "pos": [round(x, 1), round(y, 1)], "footprint": list(e.footprint),
            "is_container": e.isContainer, "is_agent": e.isAgent,
            "location": phraseLocation(world, e),
            "confidence": round(world.decayed(e, now), 2),
            "last_seen": e.last_seen, "last_confirmed": e.last_confirmed,
            "seconds_since_confirmed": round(now - e.last_confirmed)}


def eventJson(world: World, ev: Event) -> dict:
    #The full record, for the timeline. from_state/to_state are the receipt.
    e = world.entities.get(ev.entity)
    return {"ts": ev.ts, "entity": ev.entity, "label": e.label if e else "",
            "kind": ev.kind.value, "cause": ev.cause,
            "confidence": round(ev.confidence, 2),
            "from_state": ev.from_state, "to_state": ev.to_state,
            "alternatives": ev.alternatives, "frame_ref": ev.frame_ref, "note": ev.note}


def eventBrief(ev: Event, now: float) -> dict:
    #What a voice answer can be built from, without the state dicts.
    return {"kind": ev.kind.value, "cause": ev.cause, "ts": ev.ts,
            "seconds_ago": round(now - ev.ts), "confidence": round(ev.confidence, 2),
            "frame_ref": ev.frame_ref, "note": ev.note}


def lastEventFor(world: World, eid: str, kinds: tuple[EventKind, ...] = ()) -> Event | None:
    for ev in reversed(world.events):
        if ev.entity == eid and (not kinds or ev.kind in kinds):
            return ev
    return None


def observationFrom(payload: dict) -> Observation:
    #Perception may be out of process, so embeddings arrive as plain lists. core.py
    #stays the only contract: this is a decode, not a second schema.
    dets = [Detection(centroid=tuple(d["centroid"]), size=tuple(d["size"]),
                      embedding=np.asarray(d["embedding"], dtype=float),
                      crop_path=d.get("crop_path", ""))
            for d in payload.get("detections", [])]
    return Observation(ts=float(payload.get("ts") or time.time()),
                       frame_ref=payload.get("frame_ref", ""),
                       detections=dets,
                       changed=[tuple(r) for r in payload.get("changed", [])],
                       agent_swept=[tuple(r) for r in payload.get("agent_swept", [])],
                       agent_present=bool(payload.get("agent_present", False)))


def stateJson(world: World, now: float) -> dict:
    return {"ts": now, "mat": list(MAT_BOUNDS),
            "entities": [entityJson(world, e, now) for e in world.entities.values()]}


def recordSnapshot(world: World, obs: Observation):
    #Rewind indexes this list directly. Inverting the event log to reconstruct a past
    #state is tempting and it is a trap: positions are resolved here, at the moment they
    #were true, because a parent that moves later would otherwise rewrite history.
    world.snapshots.append(dict(stateJson(world, obs.ts),
                                frame_ref=obs.frame_ref, seq=len(world.events)))


def applyObservation(world: World, obs: Observation) -> list[Event]:
    #The one write path, shared by the HTTP endpoint and the fake feed.
    events = settle(world, obs, CONFIG.confirm_settles)
    recordSnapshot(world, obs)
    return events


def streamJson(world: World, now: float, since: int) -> dict:
    #Whole state plus the events the client has not had yet. Confidence decays
    #between settles, so a client that only listened for events would watch a
    #frozen scene. `seq` is the event's index in the log: it survives a reconnect,
    #which resends the tail, and lets the page drop what it has already drawn.
    payload = stateJson(world, now)
    payload["events"] = [dict(eventJson(world, ev), seq=i)
                         for i, ev in enumerate(world.events[since:], since)]
    payload["seq"] = len(world.events)
    return payload


# ---- tools --------------------------------------------------------------

def whereIs(world: World, query: str) -> dict:
    e = resolveReferent(world, query)
    if e is None:
        return {"found": False, "query": query}
    now = time.time()
    ev = lastEventFor(world, e.id)
    return {"found": True, "entity": e.label, "id": e.id,
            "location": phraseLocation(world, e), "status": e.status.value,
            "confidence": round(world.decayed(e, now), 2),
            "last_confirmed": e.last_confirmed,
            "seconds_since_confirmed": round(now - e.last_confirmed),
            "supporting_event": eventBrief(ev, now) if ev else None}


def whatsIn(world: World, container: str) -> dict:
    now = time.time()
    e = resolveReferent(world, container)
    if e is None:
        if "desk" in container.strip().lower(): #The desk is a surface, not an entity
            kids = [c for c in world.childrenOf(DESK) if not c.isAgent]
            return {"found": True, "container": "desk", "id": DESK,
                    "count": len(kids), "contents": [contentJson(world, c, now) for c in kids]}
        return {"found": False, "query": container}

    inside = world.descendants(e.id)
    return {"found": True, "container": e.label, "id": e.id,
            "status": e.status.value, "count": len(inside),
            "contents": [contentJson(world, c, now) for c in inside]}


def contentJson(world: World, e: Entity, now: float) -> dict:
    return {"entity": e.label, "id": e.id, "relation": e.relation.value,
            "location": phraseLocation(world, e), "status": e.status.value,
            "confidence": round(world.decayed(e, now), 2),
            "seconds_since_confirmed": round(now - e.last_confirmed)}


def whoMoved(world: World, query: str) -> dict:
    e = resolveReferent(world, query)
    if e is None:
        return {"found": False, "query": query}
    now = time.time()
    ev = lastEventFor(world, e.id, MOVEMENT_KINDS)
    if ev is None:
        return {"found": True, "entity": e.label, "id": e.id, "moved": False}
    return {"found": True, "entity": e.label, "id": e.id, "moved": True,
            **eventBrief(ev, now)}


def history(world: World, query: str, limit: int = HISTORY_LIMIT) -> dict:
    e = resolveReferent(world, query)
    if e is None:
        return {"found": False, "query": query}
    now = time.time()
    evs = [ev for ev in world.events if ev.entity == e.id][-max(1, limit):]
    return {"found": True, "entity": e.label, "id": e.id, "count": len(evs),
            "events": [eventBrief(ev, now) for ev in evs]} #Chronological, oldest first


# ---- write --------------------------------------------------------------

@app.post("/observation")
def postObservation(payload: dict):
    obs = observationFrom(payload)
    with LOCK:
        events = applyObservation(WORLD, obs)
        return {"accepted": True, "frame_ref": obs.frame_ref,
                "events": [eventJson(WORLD, ev) for ev in events]}


# ---- read ---------------------------------------------------------------

@app.get("/")
def getPage():
    #One static page, no framework. It reads the same stream any other client would.
    return FileResponse(UI_PAGE, media_type="text/html")


@app.get("/state")
def getState():
    with LOCK:
        return stateJson(WORLD, time.time())


@app.get("/events")
def getEvents(since: float = 0.0):
    with LOCK:
        return {"since": since, "ts": time.time(),
                "events": [eventJson(WORLD, ev) for ev in WORLD.events if ev.ts > since]}


@app.get("/snapshots")
def getSnapshots(since: int = 0):
    with LOCK:
        return {"count": len(WORLD.snapshots), "since": since,
                "snapshots": WORLD.snapshots[since:]}


@app.get("/snaps/{name}")
def getSnap(name: str):
    #The keyframe an Event was believed from. Basename only: the path comes off the wire.
    path = os.path.join(SNAP_ROOT, os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="no such keyframe")
    return FileResponse(path, media_type="image/jpeg")


# ---- the camera feed ----------------------------------------------------

@app.post("/preview")
async def postPreview(request: Request):
    #Perception pushes frames here for the dashboard to show. A single assignment, so no
    #lock: a reader either gets the previous frame or the next one, and both are fine.
    jpeg = await request.body()
    if not jpeg:
        raise HTTPException(status_code=400, detail="empty frame")
    PREVIEW.update(jpeg=jpeg, ts=time.time(), seq=PREVIEW["seq"] + 1)
    return {"accepted": True, "seq": PREVIEW["seq"], "bytes": len(jpeg)}


@app.get("/preview.jpg")
def getPreviewFrame():
    if PREVIEW["jpeg"] is None:
        raise HTTPException(status_code=404, detail="no camera feed; is live.py running?")
    return Response(PREVIEW["jpeg"], media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/preview/status")
def getPreviewStatus():
    age = time.time() - PREVIEW["ts"] if PREVIEW["jpeg"] is not None else None
    return {"live": age is not None and age < PREVIEW_STALE_S,
            "seq": PREVIEW["seq"],
            "seconds_since_frame": round(age, 2) if age is not None else None}


MJPEG_TYPE = "multipart/x-mixed-replace; boundary=frame"


def mjpegFrame(jpeg: bytes) -> bytes:
    return (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")


@app.get("/preview.mjpg")
async def getPreviewStream(request: Request):
    #One connection, frames pushed as they arrive. Only send what is new, so a stalled
    #camera costs an idle socket rather than the same frame ten times a second.
    async def frames():
        sent = -1
        while not await request.is_disconnected(): #A tab that closed must end the loop,
            jpeg, seq = PREVIEW["jpeg"], PREVIEW["seq"] #not leave it running forever
            if jpeg is not None and seq != sent:
                sent = seq
                yield mjpegFrame(jpeg)
            await asyncio.sleep(1.0 / PREVIEW_HZ)

    return StreamingResponse(frames(), media_type=MJPEG_TYPE,
                             headers={"Cache-Control": "no-store"})


@app.websocket("/stream")
async def stream(ws: WebSocket):
    #Push, on a tick. A client that connects late gets the whole event log first.
    await ws.accept()
    sent = 0
    try:
        while True:
            with LOCK:
                payload = streamJson(WORLD, time.time(), sent)
                sent = payload["seq"]
            await ws.send_json(payload)
            await asyncio.sleep(1.0 / STREAM_HZ)
    except (WebSocketDisconnect, RuntimeError):
        return #The page reconnects on its own, and a closed socket is not an error


@app.get("/tools/where_is")
def getWhereIs(query: str):
    with LOCK:
        return whereIs(WORLD, query)


@app.get("/tools/whats_in")
def getWhatsIn(container: str):
    with LOCK:
        return whatsIn(WORLD, container)


@app.get("/tools/who_moved")
def getWhoMoved(query: str):
    with LOCK:
        return whoMoved(WORLD, query)


@app.get("/tools/history")
def getHistory(query: str, limit: int = HISTORY_LIMIT):
    with LOCK:
        return history(WORLD, query, limit)


# ---- fake feed ----------------------------------------------------------

def feedFake(speed: float = 1.0):
    #One pass of the script into the live world. Decay keeps the state moving after it
    #ends, which is the point: the endpoints stay interesting with no camera attached.
    for obs in fake.stream(speed):
        with LOCK:
            events = applyObservation(WORLD, obs)
        print(f"[fake] {obs.frame_ref} {', '.join(ev.kind.value for ev in events) or 'nothing'}")
    print("[fake] script exhausted, world is live and decaying")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="feed the scripted observations in on a timer")
    ap.add_argument("--speed", type=float, default=1.0, help="script speed multiplier")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.fake:
        threading.Thread(target=feedFake, args=(args.speed,), daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
