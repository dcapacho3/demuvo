"""A controller you can read in one go: turn to face the path, then drive.

Each time it runs it looks at a point of the path a little way ahead. If the robot is
not facing it, the robot turns on the spot; if it is, the robot drives and steers a
little. At the goal it stops. It is simpler than `PurePursuit` and wastes more time
turning, which is the point: change the numbers and watch what happens.

Use it in a world file:

    controller = "turn_then_drive:TurnThenDrive"
    controller_params = { speed = 0.5 }
"""

import math

from robomath import wrap_angle

from demuvo import Twist


class TurnThenDrive:
    """Turn on the spot until the point ahead is in front, then drive to it.

    Args:
        speed: forward speed in m/s.
        ahead: distance in metres to the path point it faces.
        turn_rate: how fast it turns on the spot, in rad/s.
        face_within: error in radians below which it drives instead of turning.
    """

    def __init__(self, speed=0.5, ahead=0.4, turn_rate=1.0, face_within=0.2):
        """Keep the four numbers."""
        self.speed, self.ahead = speed, ahead
        self.turn_rate, self.face_within = turn_rate, face_within

    def command(self, obs):
        """The command for this step."""
        if len(obs.path) == 0:
            return Twist()
        x, y, yaw = obs.pose.xyt
        gx, gy = obs.goal.translation
        if math.hypot(gx - x, gy - y) <= obs.goal_tolerance[0]:
            return Twist()
        gaps = [math.hypot(px - x, py - y) for px, py, _ in obs.path]
        nearest = gaps.index(min(gaps))
        target = obs.path[nearest]
        for point, gap in zip(obs.path[nearest:], gaps[nearest:]):
            target = point
            if gap >= self.ahead:
                break
        error = wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw)
        if abs(error) > self.face_within:
            return Twist(wz=math.copysign(self.turn_rate, error))
        return Twist(vx=self.speed, wz=2.0 * error)
