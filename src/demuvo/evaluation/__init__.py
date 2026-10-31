"""Judging a run: which task a robot is on, when it is done, and the result."""

from .judge import STOPPED, Judge, goal_error, reached
from .result import result_lines

__all__ = ["STOPPED", "Judge", "goal_error", "reached", "result_lines"]
