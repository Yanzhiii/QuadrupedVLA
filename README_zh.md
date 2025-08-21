[English](README.md) | 中文

## QuadrupedVLA / qvla_bridge

`qvla_bridge` 是一个将视觉-语言导航模型（NaVILA 多帧风格 / OpenVLA 单帧风格）与 Unitree Go2 机器人 ROS 2 控制栈对接的桥接节点。该节点订阅相机图像与自然语言指令，调用后端模型返回动作向量，并发布 `/cmd_vel` 控制机器人。

### 演示
![演示](assets/demo.gif)

使用 `a8cheng/navila-llama3-8b-8f` 执行的指令：

“walk straight along the corridor, turn left slowly at the green exit sign, continue walking along the corridor, making sure to go straight along the corridor, and stop 2 meter away from the shoes.”

### 功能概览
- 订阅：`/camera/image_raw`（传感器图像）、`/qvla/user_cmd`（自然语言指令）
- 发布：`/cmd_vel`（速度控制）
- 模式：`navila`（序列推理）或 `openvla`（单帧推理）
- 可选调试日志与导航任务数据落盘（见 `debug_logs/`）

### 依赖与前置条件
- Ubuntu 22.04 + ROS 2 Humble
- Python 3.10+
- Go2 驱动（`go2_ros2_sdk`，发布 `/camera/image_raw` 并订阅 `/cmd_vel`）
- 后端模型（二选一，推荐 NaVILA）：
  - NaVILA：`navila_repo_main`
  - OpenVLA：`openvla`（需其 API 服务）

本仓库不包含上述第三方项目，请按照各自文档独立部署。

### 安装与构建（先拉齐依赖，再一次性构建）
1) 准备工作空间并拉取所有代码：
```bash
mkdir -p ~/ws_qvla/src && cd ~/ws_qvla/src

# 克隆本仓库（如已存在可跳过）
git clone https://github.com/Yanzhiii/QuadrupedVLA.git qvla_ws

# 拉取 NaVILA 与 Go2 驱动（按你的上游来源）
git clone https://github.com/AnjieCheng/NaVILA.git navila_repo_main
git clone https://github.com/abizovnuralem/go2_ros2_sdk.git go2_ros2_sdk
```

2) 准备 NaVILA 环境（名称需与启动脚本一致）：
```bash
conda create -n navila python=3.10 -y
conda activate navila
pip install -r ~/ws_qvla/src/navila_repo_main/requirements.txt
# 可选：pip install -e ~/ws_qvla/src/navila_repo_main
```

3) 一次性安装 ROS 依赖并构建：
```bash
source /opt/ros/humble/setup.bash
cd ~/ws_qvla
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

### 快速启动（工作空间根目录的 start 脚本）
`start_qvla_system.sh` 将：
- 启动 Go2 驱动（`go2_ros2_sdk`）
- 按模式启动桥接（NaVILA 或 OpenVLA）
- 打开一个终端，展示 `/qvla/user_cmd` 指令示例

用法（默认 `navila`）：
```bash
cd ~/ws_qvla
# 如根目录不存在脚本，可从包内拷贝：
cp src/qvla_bridge/scripts/start_qvla_system.sh ./start_qvla_system.sh
chmod +x ./start_qvla_system.sh

# 可选：设置机器人 IP 与连接方式
export ROBOT_IP=192.168.xxx.xxx
export CONN_TYPE=webrtc

./start_qvla_system.sh navila
# 或：./start_qvla_system.sh openvla  (需先启动 OpenVLA API)
```
说明：在 NaVILA 模式下脚本会执行 `conda activate navila` 并设置 `PYTHONPATH` 指向 `navila_repo_main`，请确保名称与路径一致。

### 指令示例
```bash
ros2 topic pub /qvla/user_cmd std_msgs/String "data: 'move forward, stop in front of the wall'" --once
```

### 调试与日志
- 开启 `debug_enabled` 时，节点会在 `debug_logs/` 下创建会话目录，保存任务图像序列与每轮推理记录。

### 许可
MIT


