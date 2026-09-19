from dataclasses import dataclass
from core import Detection, Entity, Status
from calib import dist
from world import World
import math

#The embedder (DINOv3, torch) belongs to perception. Matching is pure arithmetic
#over the exemplar banks, so it lives here and runs anywhere.

MATCH_LO = 0.55 #Below this it is a new entity, not a returning one
MARGIN_MIN = 0.05 #Winner must beat the runner-up by this
PRIOR_WEIGHT = 0.10
PRIOR_SIGMA = 150.0 #mm


@dataclass
class MatchResult:
    entity: Entity | None
    score: float
    ambiguous: bool = False
    runner_up: Entity | None = None


def match(world: World, det: Detection) -> MatchResult:
    scored = []
    for e in world.entities.values():
        if e.status == Status.OFF_DESK or e.isAgent:
            continue
        s = e.bank.best(det.embedding)
        d = dist(det.centroid, world.absolute(e))
        s += PRIOR_WEIGHT * math.exp(-d / PRIOR_SIGMA) #The thing that was here 200ms ago is the thing here now
        scored.append((s, e))

    if not scored:
        return MatchResult(None, 0.0)
    scored.sort(key=lambda t: -t[0])
    s1, e1 = scored[0]
    s2, e2 = scored[1] if len(scored) > 1 else (0.0, None)

    if s1 < MATCH_LO:
        return MatchResult(None, s1)
    if s1 - s2 < MARGIN_MIN:
        return MatchResult(e1, s1, ambiguous=True, runner_up=e2) #Do not guess silently
    return MatchResult(e1, s1)
