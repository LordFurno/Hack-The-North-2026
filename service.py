from core import (AGENT_ID, DESK, Detection, Entity, Event, EventKind,
                  Observation, Relation, Status)
from calib import MAT_BOUNDS, quadrant
from world import World
from resolve import settle
import fake

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import argparse, asyncio, re, threading, time
import uvicorn

#Perception is the only writer, everything else reads. Phrasing is the model's job,
#grounding is this file's job, so every tool returns compact JSON and never prose.

HISTORY_LIMIT = 20
STREAM_HZ = 5.0 #State is small enough to re-send whole, so decay is watchable

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

app = FastAPI(title="spatial memory")


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
        events = settle(WORLD, obs)
        return {"accepted": True, "frame_ref": obs.frame_ref,
                "events": [eventJson(WORLD, ev) for ev in events]}


# ---- read ---------------------------------------------------------------

@app.get("/state")
def getState():
    with LOCK:
        return stateJson(WORLD, time.time())


@app.get("/events")
def getEvents(since: float = 0.0):
    with LOCK:
        return {"since": since, "ts": time.time(),
                "events": [eventJson(WORLD, ev) for ev in WORLD.events if ev.ts > since]}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    #Whole state plus the new events, on a tick. Confidence decays between settles,
    #so a client that only listened for events would show a frozen scene.
    await ws.accept()
    sent = 0
    try:
        while True:
            with LOCK:
                payload = stateJson(WORLD, time.time())
                payload["events"] = [eventJson(WORLD, ev) for ev in WORLD.events[sent:]]
                sent = len(WORLD.events)
            await ws.send_json(payload)
            await asyncio.sleep(1.0 / STREAM_HZ)
    except WebSocketDisconnect:
        return


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
            events = settle(WORLD, obs)
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
