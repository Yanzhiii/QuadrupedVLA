#!/bin/bash
set -euo pipefail

# QVLA System Launcher - Portable (Backup Copy)
# Usage: ./start_qvla_system.sh [openvla|navila]

# Always operate relative to workspace root (two levels up from scripts dir)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$WS_DIR"

MODEL_TYPE=${1:-"navila"}
ROBOT_IP=${ROBOT_IP:-"192.168.0.100"}
CONN_TYPE=${CONN_TYPE:-"webrtc"}

echo "🤖 Starting QVLA System (Model: $MODEL_TYPE)"
echo "============================================="

# Resolve conda activation script if available
CONDA_SH=""
for cdir in "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [ -f "$cdir/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$cdir/etc/profile.d/conda.sh"
        break
    fi
done

launch_term() {
    local title="$1"; shift
    local cmd="$*"
    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="$title" -- bash -c "$cmd; echo; echo '[done]'; exec bash" &
    else
        setsid bash -c "$cmd" >/dev/null 2>&1 &
    fi
    sleep 1
}

warn_if_empty_ip() {
    if [ -z "${ROBOT_IP:-}" ]; then
        echo "[WARN] ROBOT_IP is empty. The driver may fail to connect. Export ROBOT_IP before running."
    fi
}

# 1. Robot Driver (always needed)
echo "🚁 Starting Robot Driver..."
warn_if_empty_ip
launch_term "Go2-Driver" "source /opt/ros/humble/setup.bash && source '$WS_DIR'/install/setup.bash && export ROBOT_IP='$ROBOT_IP' && export CONN_TYPE='$CONN_TYPE' && ros2 launch go2_robot_sdk robot.launch.py"

# 2. Model-specific components
if [ "$MODEL_TYPE" = "openvla" ]; then
    echo "📡 Starting OpenVLA API Server..."
    launch_term "OpenVLA-API" "cd '$WS_DIR'/src/openvla && { [ -n '$CONDA_SH' ] && source '$CONDA_SH' && conda activate openvla || true; } && python vla-scripts/deploy.py --host 0.0.0.0 --port 8000"
    
    echo "🌉 Starting QVLA Bridge (OpenVLA mode)..."
    launch_term "QVLA-Bridge" "source /opt/ros/humble/setup.bash && source '$WS_DIR'/install/setup.bash && cd '$WS_DIR' && python3 -m qvla_bridge.bridge_node --ros-args -p model_type:=openvla"
    
elif [ "$MODEL_TYPE" = "navila" ]; then
    echo "Starting QVLA Bridge (NaVILA mode)..."
    launch_term "QVLA-Bridge" "{ [ -n '$CONDA_SH' ] && source '$CONDA_SH' && conda activate navila || true; } && export PYTHONPATH='$WS_DIR'/src/navila_repo_main:\$PYTHONPATH && source /opt/ros/humble/setup.bash && source '$WS_DIR'/install/setup.bash && cd '$WS_DIR' && ros2 launch qvla_bridge bridge_launch.py model_type:=navila base_linear_speed:=0.24 base_angular_speed:=0.75"
else
    echo "❌ Invalid model type: $MODEL_TYPE"
    echo "Usage: $0 [openvla|navila]"
    exit 1
fi

# 3. Command Interface
echo "💬 Starting Command Interface..."
launch_term "Commands" "source /opt/ros/humble/setup.bash && source '$WS_DIR'/install/setup.bash && echo '🎮 QVLA Commands ('$MODEL_TYPE' mode):' && echo '=================================' && echo 'ros2 topic pub /qvla/user_cmd std_msgs/String \"data: \\\"move forward\\\"\" --once' && echo 'ros2 topic pub /qvla/user_cmd std_msgs/String \"data: \\\"turn left\\\"\" --once' && echo 'ros2 topic pub /qvla/user_cmd std_msgs/String \"data: \\\"stop\\\"\" --once' && echo ''"

echo ""
echo "✅ System started! Use 'Commands' terminal to control the robot."
echo "🔄 To change models: $0 [openvla|navila]"


