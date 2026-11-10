"""Tests for running one world under many seeds."""

from pathlib import Path

from demuvo.run.seeds import Row, run_seeds, table
from demuvo.world import load_world

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"


def test_each_seed_gets_a_row_in_order():
    """Three seeds from the world's own give three rows, seeds counting up."""
    world = load_world(DEMO)
    rows = run_seeds(world, 3, limit=2.0)
    assert [r.seed for r in rows] == [world.seed + i for i in range(3)]
    assert all(r.time >= 2.0 and not r.done for r in rows)


def test_processes_give_the_same_rows_as_one_after_another():
    """Running two at once changes nothing in the results."""
    world = load_world(DEMO)
    assert run_seeds(world, 2, 2.0, jobs=2) == run_seeds(world, 2, 2.0)


def test_a_run_that_finishes_says_so():
    """With time enough the demo robot reaches its goal."""
    (row,) = run_seeds(load_world(DEMO), 1, limit=120.0)
    assert row.done
    assert row.time < 120.0


def test_the_table_lists_rows_and_counts_those_done():
    """One line a row, then how many runs finished."""
    text = table([Row(1, True, 12.5, 0), Row(2, False, 30.0, 3)])
    assert text.splitlines() == [
        "seed  done  time (s)  collisions",
        "   1   yes     12.50           0",
        "   2    no     30.00           3",
        "1 of 2 runs done",
    ]
