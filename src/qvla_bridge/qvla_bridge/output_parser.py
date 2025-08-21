#!/usr/bin/env python3
"""
Model output parsing utilities 
"""
import numpy as np
from geometry_msgs.msg import Twist
from .model_adapters import ModelType
from .actions import ActionCatalog

def parse_model_output(model_type, model_output):
    """
    Unified parse entry -> (twist, action)
    """
    if model_type == ModelType.NAVILA:
        # NaVILA text output: keywords only
        return parse_text_action_keywords(model_output)
    elif model_type == ModelType.OPENVLA:
        # OpenVLA action vector -> Twist
        return action_vector_to_twist(model_output)
    else:
        # Unknown model type
        print(f"Warning: Unknown model type {model_type}")
        return action_vector_to_twist(np.zeros(7))

def parse_text_action_keywords(text_output):
    """
    Based on NaVILA text output, parse:
    - Action type: forward/backward/turn left/turn right/stop
    - Distance (m) or angle (deg): simple number extraction, rounded to action library supported levels
    Returns (twist, action_array or dict), where dict carries parsed DiscreteAction
    """
    action_vec = np.zeros(7)
    text = (text_output or "").lower()

    kind = None
    # Action type matching
    if any(p in text for p in ["turn left", "rotate left", "left turn"]):
        kind = 'turn_left'
        action_vec[5] = 0.5
    elif any(p in text for p in ["turn right", "rotate right", "right turn"]):
        kind = 'turn_right'
        action_vec[5] = -0.5
    elif any(p in text for p in ["move forward", "go forward", "go ahead", "forward", "straight", "ahead"]):
        kind = 'forward'
        action_vec[0] = 0.3
    elif any(p in text for p in ["move backward", "backward", "reverse", "back"]):
        kind = 'backward'
        action_vec[0] = -0.3
    elif any(p in text for p in ["stop", "halt"]):
        kind = 'stop'
        action_vec[:] = 0.0

    # Simple number extraction (integer or decimal), no unit parsing
    import re
    numbers = [float(m.group()) for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", text)]

    discrete = None
    if kind in ('forward', 'backward'):
        # Distance: if number exists use nearest level; otherwise default 0.25m
        dist = numbers[0] if numbers else 0.25
        dist = ActionCatalog.round_distance_meters(dist)
        discrete = {"kind": kind, "magnitude": dist}
    elif kind in ('turn_left', 'turn_right'):
        # Angle: if number exists use nearest level; otherwise default 15deg
        deg = numbers[0] if numbers else 15.0
        deg = ActionCatalog.round_turn_degrees(deg)
        discrete = {"kind": kind, "magnitude": deg}
    elif kind == 'stop':
        discrete = {"kind": 'stop', "magnitude": 0.0}

    # Return two types of information: Twist symbol vector + original discrete action object
    twist, _ = action_vector_to_twist(action_vec)
    return twist, {"action": action_vec.tolist(), "discrete": discrete}

def action_vector_to_twist(action_data):
    """
    Convert model output to Twist message
    
    Args:
        action_data: model output data, could be 7D vector (OpenVLA) or parsed action
    
    Returns:
        tuple: (twist_msg, raw_action_array)
    """
    twist = Twist()
    
    # Extract action array
    if isinstance(action_data, dict) and 'action' in action_data:
        action = np.array(action_data['action'])
    else:
        action = np.array(action_data)
    
    # Ensure array length is sufficient
    if len(action) < 6:
        print(f"Warning: Expected at least 6D action vector, got {len(action)}D")
        return twist, np.zeros(7)
    
    # Generate placeholder Twist (symbolic values with magnitude 1), convenient for upper layer to override with fixed speed
    twist.linear.x = float(action[0])
    twist.linear.y = float(action[1])
    twist.angular.z = float(action[5])
    
    # Safety clamp -> [-1, 1]
    max_val = 1.0
    twist.linear.x = np.clip(twist.linear.x, -max_val, max_val)
    twist.linear.y = np.clip(twist.linear.y, -max_val, max_val)
    twist.angular.z = np.clip(twist.angular.z, -max_val, max_val)
    
    return twist, action 