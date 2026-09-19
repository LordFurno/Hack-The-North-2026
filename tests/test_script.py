from core import DESK, Entity, EventKind, Relation, Status
from world import World
from resolve import settle
from fake import SCRIPT, FakeDesk
import pytest


def runScript(upto: int) -> tuple[World, FakeDesk]:
    world, desk = World(), FakeDesk()
    for row in SCRIPT[:upto]:
        settle(world, desk.step(*row))
    return world, desk


def entityFor(world: World, desk: FakeDesk, name: str) -> Entity:
    #The fake stream carries no labels, so find an entity the way the system does.
    v = desk.embedding(name)
    return max(world.entities.values(), key=lambda e: e.bank.best(v))


def kinds(world: World, e: Entity) -> list[EventKind]:
    return [ev.kind for ev in world.events if ev.entity == e.id]


def test_cover_puts_the_mug_under_the_box():
    world, desk = runScript(3)
    mug, box = entityFor(world, desk, "mug"), entityFor(world, desk, "box")

    assert mug.status is Status.HIDDEN
    assert mug.parent == box.id
    assert mug.relation is Relation.UNDER
    assert EventKind.COVERED in kinds(world, mug)


def test_sliding_the_box_carries_the_hidden_mug():
    world, desk = runScript(3)
    mug, box = entityFor(world, desk, "mug"), entityFor(world, desk, "box")
    assert world.absolute(mug) == pytest.approx((150.0, 200.0))

    settle(world, desk.step(7.5, "move", "box", (170, 215))) #Still covering it

    assert world.absolute(box) == pytest.approx((170.0, 215.0))
    assert world.absolute(mug) == pytest.approx((170.0, 215.0))
    assert mug.status is Status.HIDDEN #Nothing was exposed, so nothing was tested
    assert mug.parent == box.id


def test_lifting_the_box_reveals_the_mug():
    world, desk = runScript(4)
    mug = entityFor(world, desk, "mug")

    assert mug.status is Status.VISIBLE
    assert mug.confidence == 1.0
    assert mug.parent == DESK
    assert world.absolute(mug) == pytest.approx((150.0, 200.0))
    assert EventKind.REVEALED in kinds(world, mug)


def test_removing_the_mug_takes_it_off_the_desk():
    world, desk = runScript(5)
    mug = entityFor(world, desk, "mug")

    assert mug.status is Status.OFF_DESK
    assert mug.parent is None
    assert EventKind.LEFT_DESK in kinds(world, mug)


def test_absolute_position_follows_the_parent():
    #Moving a parent is a single write, with no propagation pass to get wrong.
    world = World()
    box, mug = Entity.new(label="box", footprint=(160, 140)), Entity.new(label="mug", footprint=(80, 80))
    world.entities[box.id], world.entities[mug.id] = box, mug
    world.reparent(box, DESK, Relation.ON, (150, 200))
    world.reparent(mug, box.id, Relation.UNDER, (160, 210))

    box.pose = (400.0, 300.0)

    assert world.absolute(mug) == pytest.approx((410.0, 310.0))


def test_belief_is_falsified_when_the_box_comes_up_empty():
    world, desk = runScript(3)
    mug = entityFor(world, desk, "mug")
    desk.palm("mug") #Taken out while it was covered, nothing to see

    settle(world, desk.step(15.0, "remove", "box"))

    assert mug.status is Status.UNRESOLVED
    assert mug.confidence == pytest.approx(0.2)
    assert EventKind.BELIEF_FALSIFIED in kinds(world, mug)
