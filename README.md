English | [中文](README_zh.md)

## QuadrupedVLA / qvla_bridge

`qvla_bridge` is a ROS 2 bridge node that connects vision-language navigation models (NaVILA multi-frame style / OpenVLA single-frame style) to the Unitree Go2 robot control stack. It subscribes to camera images and natural language instructions, calls the backend model, and publishes `/cmd_vel` to actuate the robot.

### Demo
![Demo](assets/demo.gif)

Instruction executed by `a8cheng/navila-llama3-8b-8f`:

"walk straight along the corridor, turn left slowly at the green exit sign, continue walking along the corridor, making sure to go straight along the corridor, and stop 2 meter away from the shoes."

### Features
- Subscribes: `/camera/image_raw` (sensor images), `/qvla/user_cmd` (natural language)
- Publishes: `/cmd_vel` (velocity control)
- Modes: `navila` (sequence inference) or `openvla` (single-frame)
- Optional debug logging and task data under `debug_logs/`

### Prerequisites
- Ubuntu 22.04 + ROS 2 Humble
- Python 3.10+
- Go2 driver (`go2_ros2_sdk`) that provides `/camera/image_raw` and consumes `/cmd_vel`
- One backend model (NaVILA recommended):
  - NaVILA: `navila_repo_main`
  - OpenVLA: `openvla` (requires its API server)

This repository does not include those third-party projects. Please clone and set them up separately following their documentation.

### Install and Build (clone all repos first, then build once)
1) Prepare workspace and clone all repos:
```bash
mkdir -p ~/ws_qvla/src && cd ~/ws_qvla/src

# Clone this repo (skip if already present)
git clone https://github.com/Yanzhiii/QuadrupedVLA.git qvla_ws

# Clone NaVILA (recommended) and Go2 driver (adjust to your upstreams)
git clone https://github.com/AnjieCheng/NaVILA.git navila_repo_main
git clone https://github.com/abizovnuralem/go2_ros2_sdk.git go2_ros2_sdk
```

2) Prepare NaVILA environment (name must match start script):
```bash
conda create -n navila python=3.10 -y
conda activate navila
pip install -r ~/ws_qvla/src/navila_repo_main/requirements.txt
# Optional: pip install -e ~/ws_qvla/src/navila_repo_main
```

3) Install ROS dependencies and build all at once:
```bash
source /opt/ros/humble/setup.bash
cd ~/ws_qvla
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

### Quick Start (workspace-level start script)
`start_qvla_system.sh` will:
- Launch the Go2 driver (`go2_ros2_sdk`)
- Launch the bridge in NaVILA or OpenVLA mode
- Open a terminal with example `/qvla/user_cmd` commands

Usage (default `navila`):
```bash
cd ~/ws_qvla
# If the script is not at repo root, copy from the package backup:
cp src/qvla_bridge/scripts/start_qvla_system.sh ./start_qvla_system.sh
chmod +x ./start_qvla_system.sh

# Optional: robot IP and connection type (webrtc / wifi, etc.)
export ROBOT_IP=192.168.xxx.xxx
export CONN_TYPE=webrtc

./start_qvla_system.sh navila
# Or: ./start_qvla_system.sh openvla  (requires OpenVLA API running)
```
Note: In NaVILA mode the script runs `conda activate navila` and sets `PYTHONPATH` to `navila_repo_main`. Ensure your environment name and path match the setup above.

### Example command
```bash
ros2 topic pub /qvla/user_cmd std_msgs/String "data: 'move forward, stop in front of the wall'" --once
```

### Debug and logs
- When `debug_enabled` is true, the node creates a session directory under `debug_logs/`, saving task image sequences and per-iteration inference records.

### License
MIT


