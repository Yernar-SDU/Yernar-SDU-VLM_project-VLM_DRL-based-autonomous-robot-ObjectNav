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

            obs = self.env.reset()
            
            # Initialize tracking
            self.start_pos = self.get_robot_position()
            last_pos = self.start_pos
            self.path_length = 0.0
            
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
                
                step_count += 1
                time.sleep(0.1)      
            
            final_pos = self.get_robot_position()
            
            # Calculate final distance to goal
            final_distance = float(np.sqrt((final_pos[0] - true_x)**2 + (final_pos[1] - true_y)**2))

            success = False 
            
            # Determine success - info is a boolean in your environment
            distance_threshold = 1.00 # meters
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
        filename = "navigation_trials_mnd.xlsx"
        if os.path.exists(filename):
            df_existing = pd.read_excel(filename)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new

        df.to_excel(filename, index=False)
        print(f"\n📁 Excel updated → {filename} ({len(df_new)} new rows added)\n")
        return results

        