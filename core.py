from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time, uuid

DESK = "DESK"

class Relation(str, Enum):
    ON = "ON"
    IN = "IN"
    UNDER = "UNDER"
    HELD = "HELD"

class Status(str, Enum):
    VISIBLE = "VISIBLE"
    HIDDEN = "HIDDEN"
    OFF_DESK = "OFF_DESK"
    UNRESOLVED = "UNRESOLVED"


class EventKind(str, Enum):
    APPEARED = "APPEARED"
    MOVED = "MOVED"
    COVERED = "COVERED"
    REVEALED = "REVEALED"
    PICKED_UP = "PICKED_UP"
    PUT_DOWN = "PUT_DOWN"
    LEFT_DESK = "LEFT_DESK"
    LOST = "LOST"
    CONFIRMED = "CONFIRMED"
    BELIEF_FALSIFIED = "BELIF_FALSIFIED"



@dataclass
class ExampleBank: #Holds Dino vectors for an object to compare
    vectors: list[np.ndarray] = field(default_factory=list)
    cap: int = 8

    ADD_MIN_MATCH = 0.85 #Only learn from confident matches
    ADD_MAX_SIM = 0.95 #Only learn if it adds info

    def best(self, q: np.ndarray):
        return max((float(q @ v) for v in self.vectors), default=0.0)

    def offer(self, v: np.ndarray, matchScore: float):
        if matchScore < self.ADD_MIN_MATCH:
            return
        if self.best(v) > self.ADD_MAX_SIM:
            return

        self.vectors.append(v)
        if len(self.vectors) > self.cap:
            self.killWorst()

    def killWorst(self):
        sims = [sum(float(a @ b) for b in self.vectors if b is not a) for a in self.vectors]
        self.vectors.pop(int(np.argmax(sims)))

@dataclass
class Entity:
    id: str
    label: str = "unknown"
    isContainer: bool = False #Can it hold things (important bc it could just be under it)
    isAgent: bool = False #Hand or smth

    bank: ExampleBank = field(default_factory=ExampleBank)  

    parent: str | None = DESK #None is just means its on the desk
    relation: Relation = Relation.ON
    pose: tuple[float, float] = (0.0, 0.0) #(x,y) mm relative to parent
    footprint: tuple[float, float] = (0.0, 0.0)#(w,h) mm axis-aligned

    status: Status = Status.VISIBLE #Visible, hidden, off_deks, unresolved
    confidence: float = 1.0
    last_seen: float = 0.0
    last_confirmed: float = 0.0

    @staticmethod
    def new(**kw) -> "Entity":
        now = time.time()
        return Entity(id=uuid.uuid4().hex[:8], last_seen=now, last_confirmed=now, **kw)


@dataclass
class Event:
    ts: float
    entity: str
    kind: EventKind
    cause: str #"agent" | "covered_by:<id>" | "unexplained"
    confidence: float
    from_state: dict | None = None #{parent, relation, pose, status}
    to_state:   dict | None = None
    alternatives: list[dict] = field(default_factory=list)
    frame_ref: str = "" #keyframe that produced this
    note: str = "" #human-readable for the timeline


def absolutePosition(e: Entity, world: dict[str, Entity]):
    x,y = e.pose
    p = e.parent
    while p and p!= "DESK":
        par = world[p]
        x += par.pose[0]
        y += par.pose[1]
        p = par.parent
    return (x,y)

