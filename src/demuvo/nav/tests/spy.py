"""A controller that records what it is given, as module text a test can import."""

MODULE = """
from demuvo import Twist
seen = []
class Spy:
    def __init__(self, speed=0.5, lookahead=0.6):
        pass
    def command(self, obs):
        seen.append(obs)
        return Twist()
"""
