#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


PROJECT_TMP_DIR = "/home/yz/Projects/qvla_ws/src/qvla_bridge/tmp"
FORWARD_COMP_M = 0.2   # default overshoot compensation for forward motions (meters)
TURN_COMP_DEG = 10.0   # default undershoot compensation for turns (degrees)


@dataclass
class AtomicActionSpec:
    name: str
    kind: str  # "forward" | "turn_left" | "turn_right" | "stop"
    magnitude: float  # meters for forward, degrees for turn; unused for stop


def build_action_spec(action_name: str) -> AtomicActionSpec:
    mapping = {
        "forward_25cm": ("forward", 0.25),
        "forward_50cm": ("forward", 0.50),
        "forward_75cm": ("forward", 0.75),
        "turn_left_15deg": ("turn_left", 15.0),
        "turn_left_30deg": ("turn_left", 30.0),
        "turn_left_45deg": ("turn_left", 45.0),
        "turn_right_15deg": ("turn_right", 15.0),
        "turn_right_30deg": ("turn_right", 30.0),
        "turn_right_45deg": ("turn_right", 45.0),
        "stop": ("stop", 0.0),
    }
    if action_name not in mapping:
        raise ValueError(f"Unsupported action: {action_name}")
    kind, mag = mapping[action_name]
    return AtomicActionSpec(name=action_name, kind=kind, magnitude=mag)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # yaw (Z) from quaternion
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_normalize(angle: float) -> float:
    # Normalize to [-pi, pi]
    a = (angle + math.pi) % (2.0 * math.pi)
    if a < 0:
        a += 2.0 * math.pi
    return a - math.pi


def yaw_diff(a: float, b: float) -> float:
    # shortest signed difference b - a in [-pi, pi]
    return angle_normalize(b - a)


class OdomOnce(Node):
    def __init__(self, odom_topic: str):
        super().__init__('measure_action_odom_once')
        self._odom_msg: Optional[Odometry] = None
        self._sub = self.create_subscription(Odometry, odom_topic, self._on_odom, 10)

    def _on_odom(self, msg: Odometry) -> None:
        if self._odom_msg is None:
            self._odom_msg = msg

    def wait_for_one(self, timeout_s: float) -> Optional[Odometry]:
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._odom_msg is not None:
                return self._odom_msg
        return None


def odom_to_pose2d(msg: Odometry) -> Tuple[float, float, float]:
    px = msg.pose.pose.position.x
    py = msg.pose.pose.position.y
    ox = msg.pose.pose.orientation.x
    oy = msg.pose.pose.orientation.y
    oz = msg.pose.pose.orientation.z
    ow = msg.pose.pose.orientation.w
    yaw = quaternion_to_yaw(ox, oy, oz, ow)
    return px, py, yaw


def ensure_tmp_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_odom_yaml(path: str, msg: Odometry) -> None:
    # Minimal YAML writer without external deps
    px = msg.pose.pose.position.x
    py = msg.pose.pose.position.y
    pz = msg.pose.pose.position.z
    ox = msg.pose.pose.orientation.x
    oy = msg.pose.pose.orientation.y
    oz = msg.pose.pose.orientation.z
    ow = msg.pose.pose.orientation.w
    sec = msg.header.stamp.sec
    nsec = msg.header.stamp.nanosec
    frame_id = msg.header.frame_id
    child_frame_id = msg.child_frame_id

    lines = []
    lines.append("header:")
    lines.append("  stamp:")
    lines.append(f"    sec: {sec}")
    lines.append(f"    nanosec: {nsec}")
    lines.append(f"  frame_id: {frame_id}")
    lines.append(f"child_frame_id: {child_frame_id}")
    lines.append("pose:")
    lines.append("  pose:")
    lines.append("    position:")
    lines.append(f"      x: {px}")
    lines.append(f"      y: {py}")
    lines.append(f"      z: {pz}")
    lines.append("    orientation:")
    lines.append(f"      x: {ox}")
    lines.append(f"      y: {oy}")
    lines.append(f"      z: {oz}")
    lines.append(f"      w: {ow}")
    content = "\n".join(lines) + "\n---\n"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def stream_cmd_vel(node: Node, pub, twist: Twist, duration_s: float, rate_hz: float) -> None:
    period = 1.0 / max(1e-6, rate_hz)
    end_time = time.time() + max(0.0, duration_s)
    while time.time() < end_time and rclpy.ok():
        pub.publish(twist)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(period)


def build_twist_for_action(spec: AtomicActionSpec, v_lin: float, v_ang: float) -> Twist:
    t = Twist()
    if spec.kind == "forward":
        t.linear.x = float(v_lin)
        t.angular.z = 0.0
    elif spec.kind == "turn_left":
        t.linear.x = 0.0
        t.angular.z = float(abs(v_ang))
    elif spec.kind == "turn_right":
        t.linear.x = 0.0
        t.angular.z = -float(abs(v_ang))
    elif spec.kind == "stop":
        t.linear.x = 0.0
        t.angular.z = 0.0
    else:
        raise ValueError(f"Unknown action kind: {spec.kind}")
    t.linear.y = 0.0
    return t


def compute_duration_s(spec: AtomicActionSpec, v_lin: float, v_ang: float, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    if spec.kind == "forward":
        # apply simple default overshoot compensation
        commanded_dist = max(0.0, float(spec.magnitude) - max(0.0, FORWARD_COMP_M))
        return float(commanded_dist / max(1e-6, abs(v_lin)))
    if spec.kind in ("turn_left", "turn_right"):
        # apply simple default undershoot compensation (turns tend to under-rotate)
        commanded_deg = float(spec.magnitude) + max(0.0, TURN_COMP_DEG)
        radians = math.radians(commanded_deg)
        return float(radians / max(1e-6, abs(v_ang)))
    return 0.2  # stop: short no-op duration


def main():
    parser = argparse.ArgumentParser(description="Measure atomic action using odom before/after snapshots.")
    parser.add_argument("--action", required=True,
                        choices=[
                            "forward_25cm", "forward_50cm", "forward_75cm",
                            "turn_left_15deg", "turn_left_30deg", "turn_left_45deg",
                            "turn_right_15deg", "turn_right_30deg", "turn_right_45deg",
                            "stop"
                        ], help="Action to execute")
    parser.add_argument("--odom-topic", default="/odom", help="Odom topic (default: /odom)")
    parser.add_argument("--cmd-topic", default="/cmd_vel", help="CmdVel topic (default: /cmd_vel)")
    parser.add_argument("--v-lin", type=float, default=0.3, help="Linear speed m/s for forward actions")
    parser.add_argument("--v-ang", type=float, default=0.5, help="Angular speed rad/s for turn actions")
    parser.add_argument("--rate-hz", type=float, default=20.0, help="Publish rate for cmd_vel")
    parser.add_argument("--duration-s", type=float, default=None, help="Override duration seconds")
    parser.add_argument("--timeout-s", type=float, default=3.0, help="Timeout to wait a single odom message")
    args = parser.parse_args()

    ensure_tmp_dir(PROJECT_TMP_DIR)

    rclpy.init()

    # 1) Take before snapshot
    odom_once_before = OdomOnce(args.odom_topic)
    before_msg = odom_once_before.wait_for_one(args.timeout_s)
    if before_msg is None:
        print(f"Failed to receive odom from {args.odom_topic} within {args.timeout_s}s", file=sys.stderr)
        odom_once_before.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    px0, py0, yaw0 = odom_to_pose2d(before_msg)

    # 2) Execute action
    spec = build_action_spec(args.action)

    # separate node for publishing to avoid callback interference
    pub_node = Node('measure_action_publisher')
    pub = pub_node.create_publisher(Twist, args.cmd_topic, 10)

    twist = build_twist_for_action(spec, args.v_lin, args.v_ang)
    duration_s = compute_duration_s(spec, args.v_lin, args.v_ang, args.duration_s)

    # 2.1) Stream command for duration, then brake to zero
    stream_cmd_vel(pub_node, pub, twist, duration_s, args.rate_hz)
    time.sleep(0.02)
    zero = Twist()
    pub.publish(zero)
    time.sleep(0.05)
    pub.publish(zero)

    # 3) Take after snapshot (new node to ensure fresh subscription)
    odom_once_after = OdomOnce(args.odom_topic)
    after_msg = odom_once_after.wait_for_one(args.timeout_s)
    if after_msg is None:
        # still try to save before snapshot for debugging
        ts = int(time.time())
        before_path = os.path.join(PROJECT_TMP_DIR, f"odom_before_{spec.name}_{ts}.yaml")
        save_odom_yaml(before_path, before_msg)
        odom_once_before.destroy_node()
        odom_once_after.destroy_node()
        pub_node.destroy_node()
        rclpy.shutdown()
        sys.exit(2)

    px1, py1, yaw1 = odom_to_pose2d(after_msg)

    # 4) Save snapshots with action + timestamp in filename
    ts = int(time.time())
    before_path = os.path.join(PROJECT_TMP_DIR, f"odom_before_{spec.name}_{ts}.yaml")
    after_path = os.path.join(PROJECT_TMP_DIR, f"odom_after_{spec.name}_{ts}.yaml")
    save_odom_yaml(before_path, before_msg)
    save_odom_yaml(after_path, after_msg)

    # 5) Compute deltas
    dx = px1 - px0
    dy = py1 - py0
    dist = math.hypot(dx, dy)
    dyaw = yaw_diff(yaw0, yaw1)

    # expected values by spec
    if spec.kind == "forward":
        expected_dist = spec.magnitude
        expected_yaw_deg = 0.0
    elif spec.kind in ("turn_left", "turn_right"):
        expected_dist = 0.0
        expected_yaw_deg = spec.magnitude if spec.kind == "turn_left" else -spec.magnitude
    else:
        expected_dist = 0.0
        expected_yaw_deg = 0.0

    print("=== Measure Result ===")
    print(f"action:   {spec.name}")
    print(f"topics:   cmd={args.cmd_topic}, odom={args.odom_topic}")
    print(f"publish:  rate={args.rate_hz:.1f}Hz, duration={duration_s:.3f}s, v_lin={args.v_lin:.3f} m/s, v_ang={args.v_ang:.3f} rad/s")
    print(f"saved:    before={before_path}")
    print(f"          after = {after_path}")
    print(f"measured: dist={dist:.3f} m, yaw={math.degrees(dyaw):.2f} deg")
    if spec.kind == "forward":
        commanded_dist = max(0.0, expected_dist - max(0.0, FORWARD_COMP_M))
        err_m = dist - expected_dist
        print(f"desired:  dist={expected_dist:.3f} m  |  comp={FORWARD_COMP_M:.3f} m  |  commanded={commanded_dist:.3f} m")
        print(f"error:    dist={err_m:+.3f} m   (measured - desired)")
    elif spec.kind in ("turn_left", "turn_right"):
        yaw_deg = math.degrees(dyaw)
        commanded_yaw = expected_yaw_deg + (TURN_COMP_DEG if spec.kind == "turn_left" else -TURN_COMP_DEG)
        err_deg = yaw_deg - expected_yaw_deg
        print(f"desired:  yaw={expected_yaw_deg:.2f} deg  |  comp={TURN_COMP_DEG:.2f} deg  |  commanded={commanded_yaw:.2f} deg")
        print(f"error:    yaw={err_deg:+.2f} deg   (measured - desired)")

    # clean up
    odom_once_before.destroy_node()
    odom_once_after.destroy_node()
    pub_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


