#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
from pathlib import Path

class LogManager:
    """Unified log manager"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def log_node_startup(self, model_type, debug_enabled, debug_path=None):
        """Log node startup information"""
        self.logger.info(f"QVLA Bridge Node started - Model: {model_type}, Debug: {debug_enabled}")
        if debug_enabled and debug_path:
            self.logger.info(f"Debug logs: {debug_path}")
    
    def log_instruction_received(self, instruction, model_type):
        """Log received new instruction"""
        self.logger.info(f"New instruction: '{instruction}' (Model: {model_type})")
    
    def log_instruction_stopped(self):
        """Log stop instruction"""
        self.logger.info("🛑 Stopped")
    
    def log_model_inference(self, model_type, instruction, model_output, action_vector, twist):
        """Unified log for model inference results"""
        if model_type == "navila":
            # NaVILA specific log format
            self.logger.info(f"NaVILA raw output: '{model_output}'")
            self.logger.info(f"NaVILA parsed action: {action_vector}")
            self.logger.info(
                f"🤖 [{instruction}] "
                f"NaVILA: '{model_output}' → "
                f"Action: [{action_vector[0]:.3f}, {action_vector[1]:.3f}, {action_vector[5]:.3f}] → "
                f"Twist: x={twist.linear.x:.3f}, y={twist.linear.y:.3f}, z={twist.angular.z:.3f}"
            )
        else:
            # OpenVLA or other model log format
            self.logger.info(
                f"🤖 [{instruction}] "
                f"Raw: [{action_vector[0]:.3f}, {action_vector[1]:.3f}, {action_vector[5]:.3f}] → "
                f"Twist: x={twist.linear.x:.3f}, y={twist.linear.y:.3f}, z={twist.angular.z:.3f}"
            )
    
    def log_inference_failed(self):
        """Log inference failure"""
        self.logger.error("Model inference failed")
    
    def log_session_ended(self, debug_path):
        """Log session end"""
        self.logger.info(f"Session ended. Debug logs saved to: {debug_path}")
    
    def log_node_stopped(self):
        """Log node stop"""
        self.logger.info("Bridge node stopped")

class SimpleDebugger:
    """Simple debug information recorder"""
    
    def __init__(self, enabled=True, save_dir="debug_logs"):
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.save_dir = Path(save_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.save_dir / f"session_{self.session_id}"
        self.counter = 0
        
        # Create session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
    def save_inference_step(self, instruction, model_input, model_output, twist_output, model_type="unknown", robot_state="unknown", save_images: bool = False):
        """Maintain backward compatibility but no longer write step_* directories."""
        if not self.enabled:
            return
        # Compatibility function: no longer save step-level information
        return
    
    def save_session_summary(self, total_steps, start_time, end_time):
        """Save session summary information"""
        if not self.enabled:
            return
            
        summary = {
            "session_id": self.session_id,
            "total_steps": total_steps,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": end_time - start_time if end_time and start_time else 0
        }
        
        try:
            with open(self.session_dir / "session_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Session summary save failed: {e}")
    
    def get_session_path(self):
        """Get current session directory path"""
        return str(self.session_dir) if self.enabled else None 

    def save_nav_task(self, instruction, task_images, loop_records):
        """Save complete information for one navigation task:
        - Only save complete task images (images/)
        - Save single JSON file (task_log.json): each round record contains user instruction, complete prompt, used image indices, model output text, parsed control information
        """
        if not self.enabled:
            return
        from datetime import datetime
        try:
            task_id = datetime.now().strftime('%H%M%S_%f')[:-3]
            task_dir = self.session_dir / f"task_{task_id}"
            images_dir = task_dir / "images"
            task_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)

            frames_meta = []
            # Save all images (with index and timestamp)
            for idx, item in enumerate(task_images):
                ts = float(item.get('ts', 0.0))
                img = item.get('img', None)
                fname = f"frame_{idx:04d}_{int(ts)}.jpg"
                if hasattr(img, 'save'):
                    img.save(images_dir / fname)
                frames_meta.append({
                    "index": idx,
                    "timestamp": ts,
                    "file": f"images/{fname}"
                })

            # Assemble single task_log.json
            # Expand each round record in loop_records to required fields (instruction, complete prompt, used image indices, model output text, parsed control)
            # Each item in loop_records should contain:
            # { 'ts': float, 'instruction': str, 'selected_indices': [...], 'prompt': str, 'model_text': str, 'parsed': any, 'twist': {...} }
            task_log = {
                "instruction": instruction,
                "created_at": datetime.now().isoformat(),
                "total_images": len(task_images),
                "frames": frames_meta,
                "loops": loop_records,
            }
            with open(task_dir / "task_log.json", "w") as f:
                json.dump(task_log, f, indent=2)

        except Exception as e:
            print(f"Save nav task failed: {e}")