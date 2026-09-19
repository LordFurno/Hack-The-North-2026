from core import Detection, EventKind, Observation
from world import World, bucketOf, neighbours
from resolve import settle
import numpy as np
import pytest

#The guard is off by default: two-settle confirmation costs a settle of lag before a
#placed object reaches the map, and fake.py's one-action-per-settle script never
#re-examines a region, so nothing there would ever mint. Turn it on once phantoms
#actually appear on a real desk -- config.json, confirm_settles.

DIM = 8


def vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


def obsWith(ts: float, *dets: Detection) -> Observation:
    #Everything the detections sit on counts as changed, which is what makes them
    #resolvable at all.
    changed = [(d.centroid[0] - 60, d.centroid[1] - 60,
                d.centroid[0] + 60, d.centroid[1] + 60) for d in dets]
    return Observation(ts=ts, frame_ref=f"flick-{ts}", detections=list(dets), changed=changed)


def det(pos, seed: int = 1) -> Detection:
    return Detection(centroid=pos, size=(60.0, 60.0), embedding=vec(seed))


def realEntities(world: World) -> list:
    return [e for e in world.entities.values() if not e.isAgent]


def test_buckets_are_coarse_and_look_at_their_neighbours():
    assert bucketOf((10.0, 10.0)) == bucketOf((39.0, 39.0)) #Same 40 mm cell
    assert bucketOf((10.0, 10.0)) != bucketOf((41.0, 10.0))
    assert bucketOf((41.0, 10.0)) in neighbours(bucketOf((39.0, 10.0)))
    assert len(neighbours((0, 0))) == 9


def test_off_by_default_a_single_sighting_mints_immediately():
    world = World()
    settle(world, obsWith(1.0, det((150.0, 200.0))))
    assert len(realEntities(world)) == 1


def test_a_detection_seen_once_and_gone_mints_nothing():
    world = World()
    settle(world, obsWith(1.0, det((150.0, 200.0))), confirm=2)
    assert realEntities(world) == []
    assert world.events == [] #No entity means no APPEARED either

    settle(world, obsWith(2.0, det((420.0, 300.0), seed=2))) #Somewhere else entirely
    assert not any(e for e in realEntities(world)
                   if world.absolute(e)[0] == pytest.approx(150.0))


def test_a_detection_seen_twice_in_the_same_place_mints_once():
    world = World()
    settle(world, obsWith(1.0, det((150.0, 200.0))), confirm=2)
    assert realEntities(world) == []

    settle(world, obsWith(2.0, det((155.0, 204.0))), confirm=2) #Roughly the same place
    assert len(realEntities(world)) == 1
    assert [ev.kind for ev in world.events] == [EventKind.APPEARED]

    settle(world, obsWith(3.0, det((158.0, 206.0))), confirm=2) #And again, still one
    assert len(realEntities(world)) == 1


def test_a_detection_on_a_bucket_boundary_still_mints():
    #Keying on a bucket alone lets an object sitting on a cell edge jitter between two
    #buckets forever and never be believed.
    world = World()
    edge = 40.0 * 4 #Exactly a bucket boundary
    settle(world, obsWith(1.0, det((edge - 1.0, 200.0))), confirm=2)
    settle(world, obsWith(2.0, det((edge + 1.0, 200.0))), confirm=2)
    assert len(realEntities(world)) == 1


def test_flicker_in_two_different_places_mints_neither():
    world = World()
    settle(world, obsWith(1.0, det((100.0, 100.0))), confirm=2)
    settle(world, obsWith(2.0, det((500.0, 400.0), seed=2)), confirm=2)
    settle(world, obsWith(3.0, det((300.0, 120.0), seed=3)), confirm=2)
    assert realEntities(world) == []
