from core import (DESK, AGENT_ID, Rect, Relation, Status,
                  Detection, Entity, Event, Observation, absolutePosition)
from calib import overlapFraction

COVER_MIN = 0.5 #Fraction of the hidden thing's footprint that counts as covered
CHANGED_MIN = 0.2 #Footprint overlap with a changed region that makes an entity worth resolving

HALF_LIFE = { #seconds
    Status.HIDDEN: 3600.0, #Well-founded: the occluder is right there
    Status.OFF_DESK: 1800.0,
    Status.UNRESOLVED: 300.0, #Nothing is keeping this belief true
}


class World:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.events: list[Event] = []
        self._provisional: dict[int, tuple[Detection, int]] = {} #Flicker guard, filled by perception
        self.entities[AGENT_ID] = Entity(id=AGENT_ID, label="hand", isAgent=True)

    # ---- geometry -------------------------------------------------------

    def get(self, e: Entity | str) -> Entity:
        return self.entities[e] if isinstance(e, str) else e

    def absolute(self, e: Entity | str) -> tuple[float, float]:
        return absolutePosition(self.get(e), self.entities) #reparent() keeps the chain acyclic

    def footprintRect(self, e: Entity | str) -> Rect:
        e = self.get(e)
        cx, cy = self.absolute(e)
        w, h = e.footprint
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    # ---- structure ------------------------------------------------------

    def childrenOf(self, pid: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.parent == pid]

    def descendants(self, pid: str) -> list[Entity]:
        out, stack = [], [pid]
        while stack:
            for c in self.childrenOf(stack.pop()):
                out.append(c)
                stack.append(c.id)
        return out

    def reparent(self, e: Entity, parent: str | None,
                 relation: Relation, absPos: tuple[float, float]):
        #Set parent and store pose relative to it. Rejects cycles.
        if parent and parent != DESK:
            if e.id == parent or any(d.id == parent for d in self.descendants(e.id)):
                raise ValueError(f"reparent {e.id} under {parent} would cycle")
            px, py = self.absolute(parent)
        else:
            px, py = (0.0, 0.0)
        e.parent = parent
        e.relation = relation
        e.pose = (absPos[0] - px, absPos[1] - py)

    def mint(self, det: Detection, now: float, label: str = "unknown") -> Entity:
        e = Entity.new(label=label, footprint=det.size)
        e.last_seen = e.last_confirmed = now
        e.bank.vectors.append(det.embedding) #Birth view, no match score to weigh it against
        self.entities[e.id] = e
        placeOn(self, e, det.centroid)
        return e

    # ---- decay ----------------------------------------------------------

    def decayed(self, e: Entity, now: float) -> float:
        if e.status == Status.VISIBLE:
            return 1.0
        dt = now - e.last_confirmed
        c = e.confidence * 0.5 ** (dt / HALF_LIFE[e.status])

        #If the occluder is still visible and hasn't moved the evidence genuinely
        #hasn't degraded, so don't pretend it has.
        if e.status == Status.HIDDEN and e.parent in self.entities:
            occ = self.entities[e.parent]
            if occ.status == Status.VISIBLE and occ.last_confirmed >= e.last_confirmed:
                c = max(c, 0.8)
        return c


def surfaceUnder(world: World, pt: tuple[float, float], ignore: str = "") -> str:
    #The smallest container whose footprint swallows this point, else the desk itself.
    banned = {d.id for d in world.descendants(ignore)} if ignore else set()
    best, bestArea = DESK, float("inf")
    for e in world.entities.values():
        if e.id == ignore or e.id in banned or e.isAgent or not e.isContainer:
            continue
        if e.status != Status.VISIBLE:
            continue
        x0, y0, x1, y1 = world.footprintRect(e)
        if not (x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1):
            continue
        area = (x1 - x0) * (y1 - y0)
        if area < bestArea:
            best, bestArea = e.id, area
    return best


def placeOn(world: World, e: Entity, pt: tuple[float, float]):
    #Put an entity where it was just seen, on whatever surface is under that point.
    pid = surfaceUnder(world, pt, ignore=e.id)
    rel = Relation.IN if pid != DESK and world.entities[pid].isContainer else Relation.ON
    world.reparent(e, pid, rel, pt)


def covers(world: World, occluder: Entity, e: Entity) -> bool:
    return overlapFraction(world.footprintRect(e), world.footprintRect(occluder)) >= COVER_MIN


def snapshot(world: World, e: Entity) -> dict:
    return {"parent": e.parent, "relation": e.relation.value,
            "pose": e.pose, "status": e.status.value}


def missingEntities(world: World, obs: Observation, matched: set[str]) -> list[Entity]:
    #Most entities produce no detection on any given settle, because only changed
    #regions are segmented. An entity is missing only if it sat where something changed.
    out = []
    for e in world.entities.values():
        if e.id in matched or e.isAgent or e.status != Status.VISIBLE:
            continue
        fp = world.footprintRect(e)
        if any(overlapFraction(fp, region) > CHANGED_MIN for region in obs.changed):
            out.append(e)
    return out
