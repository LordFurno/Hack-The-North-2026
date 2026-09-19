from dataclasses import dataclass
from core import (AGENT_ID, Detection, Entity, Event, EventKind,
                  Observation, Relation, Status)
from calib import dist, overlapFraction, quadrant
from world import (World, COVER_MIN, missingEntities, placeOn,
                   snapshot, surfaceUnder)
from identity import match
from verify import revealed, verifyChildren

SWEPT_MIN = 0.3 #Footprint overlap with the agent's path that counts as handled
MOVE_MIN_MM = 15.0 #Below this a detection is the same thing sitting still, not a move


@dataclass
class Hypothesis:
    kind: EventKind
    confidence: float
    parent: str | None
    relation: Relation
    status: Status
    cause: str
    note: str


def h1Moved(world: World, e: Entity, matches: dict[str, Detection]) -> Hypothesis | None:
    #Not really a disappearance. Checked first because it is the common case and it
    #is free, the match already happened.
    hit = matches.get(e.id)
    if hit is None:
        return None
    return Hypothesis(EventKind.MOVED, 0.95,
                      parent=surfaceUnder(world, hit.centroid, ignore=e.id),
                      relation=Relation.ON, status=Status.VISIBLE,
                      cause="agent", note=f"moved {quadrant(hit.centroid)}")


def h2Covered(world: World, e: Entity, seen: list[Entity]) -> Hypothesis | None:
    #Something that is there now overlaps where it used to be.
    fp = world.footprintRect(e)
    best, bestOv = None, 0.0
    for c in seen:
        if c.id == e.id or c.isAgent:
            continue
        ov = overlapFraction(fp, world.footprintRect(c))
        if ov > bestOv:
            best, bestOv = c, ov
    if best is None or bestOv < COVER_MIN:
        return None

    rel = Relation.IN if best.isContainer else Relation.UNDER
    return Hypothesis(EventKind.COVERED, min(0.98, 0.6 + 0.4 * bestOv),
                      parent=best.id, relation=rel, status=Status.HIDDEN,
                      cause=f"covered_by:{best.id}",
                      note=f"{rel.value.lower()} the {best.label}")


def h3Carried(world: World, e: Entity, obs: Observation) -> Hypothesis | None:
    #The agent swept where it was and it matched nothing anywhere.
    fp = world.footprintRect(e)
    if not any(overlapFraction(fp, s) > SWEPT_MIN for s in obs.agent_swept):
        return None
    if obs.agent_present:
        return Hypothesis(EventKind.PICKED_UP, 0.80,
                          parent=AGENT_ID, relation=Relation.HELD,
                          status=Status.HIDDEN, cause="agent", note="in your hand")
    return Hypothesis(EventKind.LEFT_DESK, 0.85,
                      parent=None, relation=Relation.ON, status=Status.OFF_DESK,
                      cause="agent", note="taken off the desk")


def h4Lost(world: World, e: Entity, obs: Observation) -> Hypothesis:
    #Nothing fits. Not an embarrassing state, the honest one.
    return Hypothesis(EventKind.LOST, 0.40,
                      parent=e.parent, relation=e.relation,
                      status=Status.UNRESOLVED, cause="unexplained",
                      note="lost track of it")


def resolve(world: World, e: Entity, obs: Observation,
            matches: dict[str, Detection], seen: list[Entity]) -> Event:
    cands = [h for h in (h1Moved(world, e, matches),
                         h2Covered(world, e, seen),
                         h3Carried(world, e, obs),
                         h4Lost(world, e, obs)) if h is not None]
    winner, rest = cands[0], cands[1:]
    before = snapshot(world, e)

    absPos = matches[e.id].centroid if winner.status == Status.VISIBLE else world.absolute(e)
    world.reparent(e, winner.parent, winner.relation, absPos)
    e.status, e.confidence = winner.status, winner.confidence
    e.last_seen = obs.ts
    if winner.status == Status.VISIBLE:
        e.last_confirmed = obs.ts

    return Event(ts=obs.ts, entity=e.id, kind=winner.kind, cause=winner.cause,
                 confidence=winner.confidence, from_state=before,
                 to_state=snapshot(world, e), frame_ref=obs.frame_ref,
                 note=winner.note,
                 alternatives=[{"kind": h.kind.value, "confidence": h.confidence,
                                "note": h.note} for h in rest])


def settle(world: World, obs: Observation) -> list[Event]:
    #One settle, start to finish: match what is there, resolve what is not, then
    #test the beliefs that anything moving has put in reach.
    events: list[Event] = []
    matches: dict[str, Detection] = {}
    confirmed: set[str] = set() #Seen where we already believed it was, nothing to explain
    seen: list[Entity] = [] #Entities that produced a detection at this settle
    toVerify: list[Entity] = []

    world.entities[AGENT_ID].status = Status.VISIBLE if obs.agent_present else Status.HIDDEN

    for det in obs.detections:
        m = match(world, det)
        if m.entity is None:
            e = world.mint(det, obs.ts)
            seen.append(e)
            confirmed.add(e.id)
            events.append(Event(ts=obs.ts, entity=e.id, kind=EventKind.APPEARED,
                                cause="agent", confidence=1.0,
                                to_state=snapshot(world, e), frame_ref=obs.frame_ref,
                                note=f"appeared {quadrant(det.centroid)}"))
            continue
        if m.ambiguous:
            continue #Two candidates and no way to tell them apart: let the resolver own it

        e = m.entity
        matches[e.id] = det
        seen.append(e)
        e.bank.offer(det.embedding, m.score)
        e.footprint = det.size

        if e.status != Status.VISIBLE:
            events.append(revealed(world, e, det, obs))
            confirmed.add(e.id)
        elif dist(world.absolute(e), det.centroid) <= MOVE_MIN_MM:
            e.confidence = 1.0 #Free re-confirmation, the match already happened
            e.last_seen = e.last_confirmed = obs.ts
            confirmed.add(e.id)

    missing = missingEntities(world, obs, confirmed)
    missing.sort(key=lambda e: e.id not in matches) #Things we can place go first, they are evidence for the rest

    for e in missing:
        prior = world.absolute(e)
        events.append(resolve(world, e, obs, matches, seen))
        if e.status != Status.VISIBLE or dist(prior, world.absolute(e)) > MOVE_MIN_MM:
            toVerify.append(e) #It moved, or it stopped holding anything down

    for occluder in toVerify:
        events.extend(verifyChildren(world, occluder, obs))

    world.events.extend(events)
    return events
