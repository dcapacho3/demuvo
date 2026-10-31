"""Pytest setup: the view tests run only if the view's packages are installed."""

from importlib.util import find_spec

collect_ignore = [] if find_spec("pygame") and find_spec("PIL") else ["test_view.py"]
