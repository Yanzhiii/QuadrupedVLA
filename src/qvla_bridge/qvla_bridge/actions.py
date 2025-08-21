#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Literal, Optional
import math

from geometry_msgs.msg import Twist


ActionKind = Literal['forward', 'backward', 'turn_left', 'turn_right', 'stop']


@dataclass
class DiscreteAction:
    kind: ActionKind
    magnitude: float  # meters for forward/backward; degrees for turns; ignored for stop


class ActionCatalog:
    allowed_forward_meters: List[float] = [0.25, 0.50, 0.75]
    allowed_turn_degrees: List[float] = [15.0, 30.0, 45.0]

    @staticmethod
    def round_distance_meters(distance_m: float) -> float:
        values = ActionCatalog.allowed_forward_meters
        return min(values, key=lambda v: abs(v - abs(distance_m)))

    @staticmethod
    def round_turn_degrees(degrees_val: float) -> float:
        values = ActionCatalog.allowed_turn_degrees
        return min(values, key=lambda v: abs(v - abs(degrees_val)))

    @staticmethod
    def create_twist_for_action(action: DiscreteAction, base_linear_speed: float, base_angular_speed: float) -> Twist:
        t = Twist()
        if action.kind == 'forward':
            t.linear.x = abs(base_linear_speed)
            t.angular.z = 0.0
        elif action.kind == 'backward':
            t.linear.x = -abs(base_linear_speed)
            t.angular.z = 0.0
        elif action.kind == 'turn_left':
            t.linear.x = 0.0
            t.angular.z = abs(base_angular_speed)
        elif action.kind == 'turn_right':
            t.linear.x = 0.0
            t.angular.z = -abs(base_angular_speed)
        else:  # stop
            t.linear.x = 0.0
            t.angular.z = 0.0
        t.linear.y = 0.0
        return t

    @staticmethod
    def compute_duration_seconds(action: DiscreteAction, base_linear_speed: float, base_angular_speed: float,
                                 forward_comp_m: float, turn_comp_deg: float) -> float:
        if action.kind in ('forward', 'backward'):
            commanded_m = max(0.0, float(action.magnitude) - max(0.0, forward_comp_m))
            speed = max(1e-6, abs(base_linear_speed))
            return commanded_m / speed
        if action.kind in ('turn_left', 'turn_right'):
            commanded_deg = float(action.magnitude) + max(0.0, turn_comp_deg)
            speed = max(1e-6, abs(base_angular_speed))
            return math.radians(commanded_deg) / speed
        return 0.0


