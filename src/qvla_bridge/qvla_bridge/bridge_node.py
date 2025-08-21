#!/usr/bin/env python3
import rclpy, json, cv2, numpy as np, time
import json_numpy
from collections import deque
from enum import Enum
from PIL import Image

json_numpy.patch()  # Enable numpy array JSON serialization

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# Import our utility modules
from .model_adapters import ModelType, call_model
from .sampling_utils import adaptive_frame_sampling
from .output_parser import parse_model_output
from .actions import ActionCatalog, DiscreteAction, DiscreteAction as _DA
from .debug_utils import SimpleDebugger, LogManager

class RobotState(Enum):
    # Reserved placeholder (for future use), currently no longer depends on this state machine
    IDLE = "idle"

class Bridge(Node):
    def __init__(self):
        super().__init__("qvla_bridge")
        self.br = CvBridge()
                # System state
        self.current_instruction = None
        self.latest_image = None
        self.control_active = False
        self.robot_state = RobotState.IDLE
        self.session_start_time = time.time()
        
        # Parameter configuration
        self.declare_parameter('model_type', 'navila')  # 'openvla' | 'navila'
        self.declare_parameter('debug_enabled', True)
        self.declare_parameter('image_buffer_size', 1024)
        self.declare_parameter('target_frames', 8)
        # No longer depends on velocity thresholds/state machine
        # Parameterize topics to avoid dependency on SDK fixed topic names
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('state_topic', '/go2_states')
        # Fixed action parameters (passed by startup script)
        self.declare_parameter('base_linear_speed', 0.3)
        self.declare_parameter('base_angular_speed', 0.5)
        # Discrete action execution parameters (based on calibrated duration and compensation)
        self.declare_parameter('action_mode_enabled', True)
        self.declare_parameter('action_publish_rate_hz', 20.0)
        self.declare_parameter('forward_comp_m', 0.2)
        self.declare_parameter('turn_comp_deg', 10.0)
        # Pause duration after action completion (seconds)
        self.declare_parameter('post_action_pause_s', 0.0)
        # Debug: whether to create new log session directory for each navigation task
        self.declare_parameter('debug_new_session_per_task', True)
        
        # Get configuration
        model_type_str = self.get_parameter('model_type').value
        self.model_type = ModelType.NAVILA if model_type_str == 'navila' else ModelType.OPENVLA
        self.debug_enabled = self.get_parameter('debug_enabled').value
        buffer_size = self.get_parameter('image_buffer_size').value
        self.buffer_maxlen = buffer_size
        self.target_frames = self.get_parameter('target_frames').value
        # No velocity thresholds
        self.base_linear_speed = float(self.get_parameter('base_linear_speed').value)
        self.base_angular_speed = float(self.get_parameter('base_angular_speed').value)
        self.action_mode_enabled = bool(self.get_parameter('action_mode_enabled').value)
        self.action_publish_rate_hz = float(self.get_parameter('action_publish_rate_hz').value)
        self.forward_comp_m = float(self.get_parameter('forward_comp_m').value)
        self.turn_comp_deg = float(self.get_parameter('turn_comp_deg').value)
        self.post_action_pause_s = float(self.get_parameter('post_action_pause_s').value)
        self.debug_new_session_per_task = bool(self.get_parameter('debug_new_session_per_task').value)
        
        # Image buffer (only stores images during motion)
        self.image_buffer = deque(maxlen=buffer_size)
        
        # Debug and logging utilities
        self.debugger = SimpleDebugger(enabled=self.debug_enabled)
        self.log_manager = LogManager(self.get_logger())
        
        # ROS2 subscribers and publishers
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        # User instructions
        self.create_subscription(String, "/qvla/user_cmd", self.update_instruction, 10)
        # Navigation task cache (one task from start to stop of each natural language instruction)
        self.nav_task_images = []  # [{'ts': float, 'img': PIL.Image}]
        self.nav_loop_records = []  # [{'ts': float, 'instruction': str, 'selected_indices': List[int]}]
        
        # Image subscription (configurable)
        camera_topic = self.get_parameter('camera_topic').value
        # Compatible with different publisher QoS (RELIABLE and BEST_EFFORT)
        self.create_subscription(ROSImage, camera_topic, self.process_image_continuous, qos_profile)
        self.create_subscription(ROSImage, camera_topic, self.process_image_continuous, qos_profile_sensor_data)

        # No longer subscribe to robot state (can be enabled as needed)
        
        # Control publishing
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # Action execution state (simplified state machine: action start -> continuous publishing -> end)
        self.current_twist = Twist()
        self.active_action = None  # type: DiscreteAction | None
        self.action_end_time = 0.0
        # Pause control after action completion (maintain zero velocity and pause inference during pause period)
        self.pause_until_time = 0.0
        # No longer based on count/frequency
        
        # Control timer (high frequency for action publishing; using action frequency here)
        timer_period = 1.0 / max(1e-6, self.action_publish_rate_hz)
        self.control_timer = self.create_timer(timer_period, self.continuous_control_loop)
        
        # Start logging
        debug_path = self.debugger.get_session_path() if self.debug_enabled else None
        self.log_manager.log_node_startup(self.model_type.value, self.debug_enabled, debug_path)
    
    def update_instruction(self, msg):
        """Update current instruction"""
        new_instruction = msg.data.strip().lower()
        
        if new_instruction == "stop":
            self.current_instruction = None
            self.control_active = False
            stop_twist = Twist()
            self.pub.publish(stop_twist)
            # Immediately clear any pause timer
            self.pause_until_time = 0.0
            self.log_manager.log_instruction_stopped()
            # Navigation task ended: persist task logs
            if self.debug_enabled and len(self.nav_task_images) > 0:
                try:
                    self.debugger.save_nav_task(self.current_instruction or "", self.nav_task_images, self.nav_loop_records)
                except Exception:
                    pass
            self.nav_task_images = []
            self.nav_loop_records = []
            # Clear sampling buffer to prevent old frames from mixing into next task
            self.image_buffer = deque(maxlen=self.buffer_maxlen)
        else:
            self.current_instruction = new_instruction
            self.control_active = True
            self.log_manager.log_instruction_received(self.current_instruction, self.model_type.value)
            # New task started: clear task cache
            self.nav_task_images = []
            self.nav_loop_records = []
            # Synchronously clear buffer used for model sampling to ensure first frame is task start point
            self.image_buffer = deque(maxlen=self.buffer_maxlen)
            # Optional: create new log session directory for each task
            if self.debug_enabled and self.debug_new_session_per_task:
                try:
                    self.debugger = SimpleDebugger(enabled=True)
                    self.get_logger().info(f"Debug logs: {self.debugger.get_session_path()}")
                except Exception:
                    pass
    
    def process_image_continuous(self, msg):
        """Process continuous image stream"""
        # Always update latest image (for OpenVLA single frame input)
        self.latest_image = msg
        
        # Continuously accumulate images during navigation tasks; also sample during action execution or movement
        now = time.time()
        in_action = (self.active_action is not None) and (now < self.action_end_time)
        if self.control_active:
            img = self.br.imgmsg_to_cv2(msg, "bgr8")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            self.image_buffer.append(pil_img)
            # Record to current navigation task complete image sequence
            if self.debug_enabled:
                self.nav_task_images.append({'ts': now, 'img': pil_img})
    
    # Removed robot state callback
    
    def continuous_control_loop(self):
        """Simplified control loop:
        - If there's a discrete action being executed, publish current_twist at set frequency until completion
        - Otherwise perform one inference, if discrete action is parsed, calculate duration based on calibration parameters and execute
        """
        if not self.control_active or not self.current_instruction or not self.latest_image:
            return
        
        now = time.time()
        # Action execution in progress: continuous output, stop when time is up
        if self.active_action is not None:
            if now < self.action_end_time:
                # Publish at control timer frequency
                self.pub.publish(self.current_twist)
                return
            else:
                # Action ended, brake
                self.pub.publish(Twist())
                self.active_action = None
                self.robot_state = RobotState.IDLE
                # Enter pause period: maintain zero velocity and pause inference during this period
                self.pause_until_time = now + max(0.0, float(self.post_action_pause_s))
                return
        
        # If in pause period, continuously publish zero velocity and skip inference
        if now < self.pause_until_time:
            self.pub.publish(Twist())
            return

        # Start inference (currently not using fine-grained state machine, only for log placeholder)
        self.robot_state = RobotState.IDLE
        
        try:
            # Prepare input based on model type (sampling rule: must include task first frame and latest frame)
            if self.model_type == ModelType.OPENVLA:
                # OpenVLA only needs current frame
                img = self.br.imgmsg_to_cv2(self.latest_image, "bgr8")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                model_input = pil_img
                
            elif self.model_type == ModelType.NAVILA:
                # Use existing sampling strategy (don't change sampling implementation): preserve current frame + historical interpolation sampling
                frames = list(self.image_buffer)
                sampled = adaptive_frame_sampling(frames, target_frames=int(self.target_frames), preserve_current=True)
                model_input = sampled
                # Only for log index restoration: map through object identity to avoid rewriting sampling logic
                id_to_idx = {id(img): i for i, img in enumerate(frames)}
                selected_idx = [id_to_idx.get(id(img), -1) for img in sampled]
                
                # If not enough historical images, fill with current image
                if not model_input or len(model_input) == 0:
                    img = self.br.imgmsg_to_cv2(self.latest_image, "bgr8")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img)
                    model_input = [pil_img] * self.target_frames
                    self.get_logger().warn("No motion history, using current frame for sequence")
            
            # Call model inference (triggered immediately after last action ends)
            t0 = time.time()
            model_output = call_model(self.model_type, model_input, self.current_instruction)
            infer_ms = (time.time() - t0) * 1000.0
            
            if model_output is not None:
                # Use unified interface to parse model output
                navila_text = model_output["text"] if isinstance(model_output, dict) and "text" in model_output else model_output
                navila_prompt = model_output.get("prompt") if isinstance(model_output, dict) else None
                twist, raw_action = parse_model_output(self.model_type, navila_text)
                
                # NaVILA parsing may contain discrete action objects
                discrete = None
                if isinstance(raw_action, dict) and 'discrete' in raw_action:
                    discrete = raw_action['discrete']

                # If model outputs stop: immediately end current navigation task
                if self.action_mode_enabled and isinstance(discrete, dict) and discrete.get('kind') == 'stop':
                    zero_twist = Twist()
                    self.pub.publish(zero_twist)

                    # Log: model requested stop, ending task
                    try:
                        self.get_logger().info("Model requested stop -> ending navigation task")
                    except Exception:
                        pass

                    # Record this round (including time cost/prompt/parsing)
                    try:
                        if self.debug_enabled and self.model_type == ModelType.NAVILA:
                            self.nav_loop_records.append({
                                'ts': now,
                                'instruction': self.current_instruction,
                                'prompt': navila_prompt,
                                'selected_indices': selected_idx if 'selected_idx' in locals() else [],
                                'model_text': navila_text,
                                'parsed': raw_action,
                                'twist': {
                                    'linear_x': 0.0,
                                    'angular_z': 0.0,
                                },
                                'inference_ms': infer_ms,
                            })
                    except Exception:
                        pass

                    # Save task logs and cleanup
                    if self.debug_enabled and self.nav_task_images:
                        try:
                            self.debugger.save_nav_task(self.current_instruction or "", self.nav_task_images, self.nav_loop_records)
                        except Exception:
                            pass
                    self.nav_task_images = []
                    self.nav_loop_records = []
                    self.control_active = False
                    self.active_action = None
                    self.action_end_time = 0.0
                    self.pause_until_time = 0.0
                    # Clear sampling buffer to prevent old frames from mixing into next task
                    self.image_buffer = deque(maxlen=self.buffer_maxlen)
                    return

                # If action mode is enabled and discrete action is parsed, execute by action; otherwise fall back to fixed velocity publishing once for symbol vector (compatibility)
                if self.action_mode_enabled and isinstance(discrete, dict) and 'kind' in discrete and 'magnitude' in discrete:
                    # Calculate duration based on calibration parameters
                    # Build typed DiscreteAction for executor
                    da = _DA(kind=discrete['kind'], magnitude=float(discrete['magnitude']))
                    act_twist = ActionCatalog.create_twist_for_action(da, self.base_linear_speed, self.base_angular_speed)
                    duration = ActionCatalog.compute_duration_seconds(da, self.base_linear_speed, self.base_angular_speed, self.forward_comp_m, self.turn_comp_deg)

                    # Enter action execution
                    self.current_twist = act_twist
                    self.active_action = da
                    self.action_end_time = time.time() + max(0.0, duration)
                    # Send immediately once
                    self.pub.publish(self.current_twist)
                else:
                    # Compatibility path: map symbol vector to fixed velocity, send once
                    twist.linear.x = float(np.sign(twist.linear.x)) * (self.base_linear_speed if twist.linear.x != 0.0 else 0.0)
                    twist.linear.y = 0.0
                    twist.angular.z = float(np.sign(twist.angular.z)) * (self.base_angular_speed if twist.angular.z != 0.0 else 0.0)
                    self.pub.publish(twist)
                
                # Record this round (including time cost/prompt/parsing)
                if self.debug_enabled and self.model_type == ModelType.NAVILA:
                    out_twist = self.current_twist if self.active_action is not None else twist
                    self.nav_loop_records.append({
                        'ts': now,
                        'instruction': self.current_instruction,
                        'prompt': navila_prompt,
                        'selected_indices': selected_idx if 'selected_idx' in locals() else [],
                        'model_text': navila_text,
                        'parsed': raw_action,
                        'twist': {
                            'linear_x': float(out_twist.linear.x),
                            'angular_z': float(out_twist.angular.z),
                        },
                        'inference_ms': infer_ms,
                    })

                # Record loop selected frame indices (NaVILA only)
                if self.debug_enabled and self.model_type == ModelType.NAVILA:
                    self.nav_loop_records.append({
                        'ts': now,
                        'instruction': self.current_instruction,
                        'selected_indices': selected_idx if 'selected_idx' in locals() else []
                    })
                
                # Unified inference logging (ensure passing numerical vector not dictionary)
                action_for_log = raw_action['action'] if isinstance(raw_action, dict) and 'action' in raw_action else raw_action
                self.log_manager.log_model_inference(
                    model_type=self.model_type.value.lower(),
                    instruction=self.current_instruction,
                    model_output=navila_text,
                    action_vector=action_for_log,
                    twist=twist
                    )
            else:
                self.log_manager.log_inference_failed()
                
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
        finally:
            # If not executing action, set to IDLE
            if self.active_action is None:
                self.robot_state = RobotState.IDLE
    
    def destroy_node(self):
        """Cleanup work when node is destroyed"""
        if self.debug_enabled and hasattr(self, 'debugger'):
            end_time = time.time()
            self.debugger.save_session_summary(
                total_steps=self.debugger.counter,
                start_time=self.session_start_time,
                end_time=end_time
            )
            self.log_manager.log_session_ended(self.debugger.get_session_path())
        
        super().destroy_node()

def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.log_manager.log_node_stopped()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
