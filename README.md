# DRL Robot Navigation — Explorer Node

Autonomous robot exploration using frontier-based navigation and Vision-Language Model (VLM) object detection in Gazebo simulation.

---

## Prerequisites

- ROS (sourced workspace)
- Gazebo with `multi_robot_scenario` package built
- Python 3.10+
- pip dependencies (see [Installation](#installation))

---

## Installation

### 1. Install Moondream

Moondream is **not** included in `requirements.txt`. Follow the official setup guide for your preferred backend:

> **https://docs.moondream.ai/transformers/**

### 2. Install remaining dependencies

```bash
cd TD3/

# Create a virtual environment
python3 -m venv vlmenv

# Activate it
source vlmenv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

> **Note — PyTorch + CUDA:** For GPU acceleration (strongly recommended for Moondream), install the CUDA build before running the command above:
>
> ```bash
> pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu124
> ```
>
> Then re-run `pip install -r requirements.txt` (torch is already present, the rest will install).

### 3. Build the ROS workspace

The `build/`, `devel/`, `build_isolated/`, and `devel_isolated/` directories are **not included** in the repository and must be generated locally.

> **Important:** Deactivate any Python virtual environment before building, so CMake uses the system Python required by ROS.

```bash
# Deactivate venv if active
deactivate

cd catkin_ws/

# Standard build (generates build/ and devel/)
catkin_make

# --- OR isolated build (generates build_isolated/ and devel_isolated/) ---
catkin_make_isolated
```

After building, source the workspace in every terminal that uses ROS:

```bash
source catkin_ws/devel/setup.bash
# or:
source catkin_ws/devel_isolated/setup.bash
```

---

## Quick Start

### Terminal 1 — Start the VLM Server (Moondream)

```bash
cd TD3/
source vlmenv/bin/activate
uvicorn vlm_server:app --host 127.0.0.1 --port 8000 --workers 1
```

API docs available at: http://127.0.0.1:8000/docs

---

### Terminal 2 — Run the Explorer Robot

Set up the ROS environment (replace `<your_path>` with the absolute path where you cloned this repo):

```bash
export PROJECT_ROOT=<your_path>/VLM_DRL-based-autonomous-robot-ObjectNav

export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_PORT_SIM=11311
export GAZEBO_RESOURCE_PATH=$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch
source ~/.bashrc
source $PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash
export GAZEBO_RESOURCE_PATH=$PROJECT_ROOT/TD3:$GAZEBO_RESOURCE_PATH
```

> **Example:** if you cloned into `/home/john/projects`, then set:
> `export PROJECT_ROOT=/home/john/projects/DRL-robot-navigation`

Then activate the environment and launch the explorer:

```bash
cd TD3/
source vlmenv/bin/activate
python3 explorer_node.py
```

#### Explorer Node Options

| Argument | Values | Description |
|---|---|---|
| `--vlm` / `-a` | `moondream` (default), `gpt4o` | VLM backend for object detection |
| `--world` / `-w` | `TD3.world` (default), `turtlebot3_house.world` | Gazebo simulation world |

**Examples:**

```bash
# Default (Moondream VLM, one-room world)
python3 explorer_node.py

# Use GPT-4o mini VLM
python3 explorer_node.py --vlm gpt4o

# Six-room house world
python3 explorer_node.py --world turtlebot3_house.world

# Combine options
python3 explorer_node.py --vlm moondream --world turtlebot3_house.world
```

**Available worlds:**

| World file | Description |
|---|---|
| `TD3.world` | Single room |
| `turtlebot3_house.world` | Six-room house |

---

### Terminal 3 — Post-Exploration Pipeline

Run **after** exploration is complete:

```bash
source vlmenv/bin/activate
python3 coordinate_retriever.py
```

---

## Project Structure

```
DRL-robot-navigation/
├── TD3/
│   ├── requirements.txt          # All pip dependencies (install once)
│   ├── explorer_node.py          # Main exploration node
│   ├── vlm_server.py             # VLM inference server (FastAPI + Moondream)
│   ├── pixel_to_cords.py         # VLM response processor
│   ├── coordinate_retriever.py   # Post-exploration pipeline
│   ├── td3_agent.py              # TD3 RL agent
│   ├── models.py                 # Neural network models
│   ├── realsense_env.py          # RealSense camera environment
│   ├── velodyne_env.py           # Velodyne LiDAR environment
│   └── assets/
│       └── multi_robot_scenario.launch
└── catkin_ws/
    └── src/
        └── multi_robot_scenario/ # ROS package (robot URDF, launch, config)
```
