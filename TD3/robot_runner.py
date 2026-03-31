#!/usr/bin/env python3
import time
import gym
import numpy as np
import torch
from gym import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import pandas as pd
import os
import threading
import rospy
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel, DeleteModel
# Import the Gazebo environment
from real_env_dd import GazeboEnv

############################################
# Environment Wrapper for Stable Baselines3
############################################
class GazeboGymWrapper(gym.Env):
    """
    Wraps the GazeboEnv to conform to Gym's API.
    Converts the tuple (image, scalars) into a dict observation.
    """
    def __init__(self):
        super(GazeboGymWrapper, self).__init__()
        self.env = GazeboEnv()
        # Observation: image is (1, 64, 64), scalars is a 7D vector.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(1, 64, 64), dtype=np.float32),
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })
        # Define the action space: e.g. linear [0,1] and angular [-1,1]
        self.action_space = spaces.Box(low=np.array([0, -1.0]),
                                       high=np.array([1.0,  1.0]),
                                       dtype=np.float32)

    def reset(self):
        obs = self.env.reset()  # returns (image, scalars)
        return {"image": obs[0], "scalars": obs[1]}

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return {"image": next_obs[0], "scalars": next_obs[1]}, reward, done, info

############################################
# Custom Feature Extractor for Combined Inputs
############################################
class CombinedExtractor(BaseFeaturesExtractor):
    """
    Fuses CNN features from the image input and MLP features from the scalar input.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super(CombinedExtractor, self).__init__(observation_space, features_dim)
        # CNN for image input
        n_input_channels = observation_space.spaces["image"].shape[0]
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        cnn_output_dim = 64

        # MLP for scalar input (7D)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(observation_space.spaces["scalars"].shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU()
        )
        combined_dim = cnn_output_dim + 64
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, features_dim),
            torch.nn.ReLU()
        )
        self._features_dim = features_dim

    def forward(self, observations):
        img = observations["image"]
        scalars = observations["scalars"]
        cnn_out = self.cnn(img)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        mlp_out = self.mlp(scalars)
        combined = torch.cat([cnn_out, mlp_out], dim=1)
        return self.fc(combined)


class RobotRunner:
    def __init__(self):
        # Instantiate the environment
        self.env = GazeboGymWrapper()
        time.sleep(4)  # Allow Gazebo to stabilize
        model_path = "td3_gazebo_custom_policy.zip"
        self.model = TD3.load(model_path, env=self.env)
        print("Loaded trained model from", model_path)

        # Track navigation metrics
        self.start_pos = (0.0, 0.0)
        self.current_pos = (0.0, 0.0)
        self.path_length = 0.0

        # ── Gazebo path trail ─────────────────────────────────────────────
        self._marker_count       = 0
        self._trial_marker_ids   = []
        self._last_marker_pos    = None
        self._marker_spacing     = 0.25
        try:
            rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5)
            rospy.wait_for_service('/gazebo/delete_model', timeout=5)
            self._spawn_sdf    = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            self._delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            self._trail_enabled = True
            print("✅ Gazebo path-trail ready")
        except Exception as e:
            self._trail_enabled = False
            print(f"⚠️  Path trail disabled: {e}")

    # ── Path trail helpers ────────────────────────────────────────────────

    def _spawn_marker(self, x, y, idx):
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='path_marker_{idx}'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry><sphere><radius>0.06</radius></sphere></geometry>
        <material>
          <ambient>0 0.9 0 1</ambient>
          <diffuse>0 1 0 1</diffuse>
          <emissive>0 0.4 0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.05
        pose.orientation.w = 1.0
        try:
            self._spawn_sdf(f'path_marker_{idx}', sdf, '', pose, 'world')
        except Exception:
            pass

    def _maybe_drop_marker(self, x, y):
        if not self._trail_enabled:
            return
        if (self._last_marker_pos is None or
                np.sqrt((x - self._last_marker_pos[0])**2 +
                        (y - self._last_marker_pos[1])**2) >= self._marker_spacing):
            idx = self._marker_count
            self._marker_count += 1
            self._trial_marker_ids.append(idx)
            self._last_marker_pos = (x, y)
            threading.Thread(target=self._spawn_marker, args=(x, y, idx),
                             daemon=True).start()

    def _clear_trail(self):
        if not self._trail_enabled:
            return
        for idx in self._trial_marker_ids:
            try:
                self._delete_model(f'path_marker_{idx}')
            except Exception:
                pass
        self._trial_marker_ids = []
        self._last_marker_pos  = None

    def _x_marker_sdf(self, name, r, g, b):
        return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='{name}'>
    <static>true</static>
    <link name='link'>
      <visual name='bar1'>
        <pose>0 0 0.03 0 0 0.7854</pose>
        <geometry><box><size>0.7 0.1 0.05</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <emissive>{r*0.6} {g*0.6} {b*0.6} 1</emissive>
        </material>
      </visual>
      <visual name='bar2'>
        <pose>0 0 0.03 0 0 -0.7854</pose>
        <geometry><box><size>0.7 0.1 0.05</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <emissive>{r*0.6} {g*0.6} {b*0.6} 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def _spawn_endpoint_markers(self, start_x, start_y, goal_x, goal_y):
        if not self._trail_enabled:
            return
        for name, x, y, r, g, b in [
            ('start_marker', start_x, start_y, 0, 0, 1),
            ('goal_marker',  goal_x,  goal_y,  1, 0, 0),
        ]:
            try:
                self._delete_model(name)
            except Exception:
                pass
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            try:
                self._spawn_sdf(name, self._x_marker_sdf(name, r, g, b), '', pose, 'world')
            except Exception as e:
                print(f"⚠️  Could not spawn {name}: {e}")

    def get_robot_position(self):
        """Get current robot position from environment"""
        # Assuming the environment has robot position in its state
        # You may need to adjust this based on your actual implementation
        try:
            return (self.env.env.odom_x, self.env.env.odom_y)
        except AttributeError:
            # Fallback if position not directly available
            return self.current_pos

    def sendGoal(self, goal_x, goal_y, true_x, true_y, object_name):
        """
        Navigate to goal position and track metrics

        Returns:
            dict: Contains success status, path_length, start/end positions, and reward
        """
        self.env.env.clear_all_trials()

        self.env.env.goal_x = goal_x
        self.env.env.goal_y = goal_y

        results = []

        for i in range(10):

            self._clear_trail()
            obs = self.env.reset()

            # Initialize tracking
            self.start_pos = self.get_robot_position()
            last_pos = self.start_pos
            self.path_length = 0.0

            self._spawn_endpoint_markers(*self.start_pos, goal_x, goal_y)
            self._maybe_drop_marker(*self.start_pos)

            done = False
            total_reward = 0.0
            step_count = 0

            print(f"🚀 Starting navigation to ({goal_x:.2f}, {goal_y:.2f})")

            while not done:
                # Use deterministic policy (no exploration noise)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward

                # Update path length
                current_pos = self.get_robot_position()
                # print(current_pos)
                # print(last_pos)
                dx = current_pos[0] - last_pos[0]
                dy = current_pos[1] - last_pos[1]
                self.path_length += np.sqrt(dx**2 + dy**2)
                last_pos = current_pos
                self.current_pos = current_pos

                self._maybe_drop_marker(*current_pos)

                step_count += 1
                time.sleep(0.1)

            final_pos = self.get_robot_position()

            final_distance = float(np.sqrt((final_pos[0] - true_x)**2 + (final_pos[1] - true_y)**2))

            success = False

            distance_threshold = 1.01 # meters
            if final_distance <= distance_threshold:
                success = True
            # else:
            #     # fallback to info / reward if distance criterion not met
            #     if isinstance(info, dict):
            #         success = info.get('success', False)
            #     elif isinstance(info, bool):
            #         success = info
            #     else:
            #         # last resort: check reward
            #         success = total_reward > 0

            print(f"{'✅ Success' if success else '❌ Failed'} | Reward: {total_reward:.2f} | "
                f"Path: {self.path_length:.2f}m | Steps: {step_count} | "
                f"Final distance to goal: {final_distance:.2f}m")

            results.append({
            'goal_x': goal_x,
            'goal_y': goal_y,
            'success': success,
            'reward': total_reward,
            'path_length': self.path_length,
            'steps': step_count,
            'final_distance': final_distance,
            'start_x': self.start_pos[0],
            'start_y': self.start_pos[1],
            'final_x': final_pos[0],
            'final_y': final_pos[1],
            'object_name': object_name,
            'true_x': true_x,
            'true_y': true_y,
            'taken_time': self.env.env.total_time
            })
            i+=1
        df_new = pd.DataFrame(results)
        filename = "navigation_trials_signs.xlsx"
        if os.path.exists(filename):
            df_existing = pd.read_excel(filename)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new

        df.to_excel(filename, index=False)
        print(f"\n📁 Excel updated → {filename} ({len(df_new)} new rows added)\n")
        return results


