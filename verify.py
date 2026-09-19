from core import Detection, Entity, Event, EventKind, Observation, Status, Relation
from calib import dist
from world import World, covers, placeOn, snapshot

REVEAL_RADIUS_MM = 60.0


def revealed(world: World, e: Entity, det: Detection, obs: Observation) -> Event:
    #A belief that was never tested is an assertion. Something we thought was hidden
    #has turned up: say whether it turned up where we claimed it would.
    before = snapshot(world, e)
    expected = world.absolute(e)
    cause = f"revealed_by:{e.parent}" if e.parent in world.entities else "seen_again"
    near = dist(det.centroid, expected) < REVEAL_RADIUS_MM

    placeOn(world, e, det.centroid)
    e.status, e.confidence = Status.VISIBLE, 1.0
    e.last_seen = e.last_confirmed = obs.ts

    return Event(ts=obs.ts, entity=e.id, kind=EventKind.REVEALED, cause=cause,
                 confidence=1.0, from_state=before, to_state=snapshot(world, e),
                 frame_ref=obs.frame_ref,
                 note="confirmed where I thought it was" if near
                      else "found it, though not where I expected")



def verifyChildren(world: World, occluder: Entity, obs: Observation) -> list[Event]:
    #Called when `occluder` moves, shrinks or leaves the desk. Anything of its that
    #turned up has already been reparented out by revealed(), so what is still
    #parented here is what did not turn up.
    events = []
    for child in world.childrenOf(occluder.id):
        if child.status != Status.HIDDEN:
            continue
        if occluder.status == Status.VISIBLE and covers(world, occluder, child):
            continue #Still covered, so the belief is untested rather than wrong
        if occluder.status in (Status.OFF_DESK, Status.HIDDEN) and child.relation == Relation.IN:
            child.status = occluder.status      # it's in there, wherever there is now
            continue
        before = snapshot(world, child)
        child.status, child.confidence = Status.UNRESOLVED, 0.2
        events.append(Event(ts=obs.ts, entity=child.id, kind=EventKind.BELIEF_FALSIFIED,
                            cause=f"expected under {occluder.label}, absent on reveal",
                            confidence=0.2, from_state=before,
                            to_state=snapshot(world, child), frame_ref=obs.frame_ref,
                            note=f"I was wrong, it is not under the {occluder.label}"))
    return events
