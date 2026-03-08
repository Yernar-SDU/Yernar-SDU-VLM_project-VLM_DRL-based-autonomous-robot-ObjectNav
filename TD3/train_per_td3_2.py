#!/usr/bin/env python3
import time
import gym
import numpy as np
import torch
import torch.nn as nn
# import gymnasium as gym
from gym import spaces
from geometry_msgs.msg import Twist
from stable_baselines3 import TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise  # Import the noise class
from stable_baselines3.common.callbacks import BaseCallback
import rospy
# Import the improved environment:
from realsense_env import GazeboEnv

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
            # "image": spaces.Box(low=0, high=1, shape=(1, 64, 64), dtype=np.float32),
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        })
        # Define the action space: e.g. linear [0,1] and angular [-1,1]
        self.action_space = spaces.Box(low=np.array([0, -1.0]),
                                       high=np.array([1.0,  1.0]),
                                       dtype=np.float32)

    def reset(self):
        obs = self.env.reset()  # returns (image, scalars)
        return {"image": obs[0], "scalars": obs[1]}

    def step(self, action):
        # action[0] = 0
        # action[1] = 0
        print('action', action[0], action[1])
        next_obs, reward, done, target = self.env.step(action)
        # Wrap the "target" flag into info
        info = {"target": target}
        return {"image": next_obs[0], "scalars": next_obs[1]}, reward, done, info

############################################
# Custom Feature Extractor for Combined Inputs
############################################
class CombinedExtractor(BaseFeaturesExtractor):
    """
    Fuses CNN features from the image input and
    MLP features from the scalar input.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super(CombinedExtractor, self).__init__(observation_space, features_dim)
        # CNN for image input
        n_input_channels = observation_space.spaces["image"].shape[0]  # e.g. 1 channel
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # outputs shape: (batch, 64, 1, 1)
        )
        cnn_output_dim = 64  # from the last conv

        # MLP for scalar input (7D)
        self.mlp = nn.Sequential(
            nn.Linear(observation_space.spaces["scalars"].shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Combine CNN and MLP
        combined_dim = cnn_output_dim + 64  # e.g., 64 + 64 = 128
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
        self._features_dim = features_dim

    def forward(self, observations):
        img = observations["image"]         # shape (B, 1, 64, 64)
        scalars = observations["scalars"]     # shape (B, 7)
        cnn_out = self.cnn(img)               # shape (B, 64, 1, 1)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # shape (B, 64)
        mlp_out = self.mlp(scalars)           # shape (B, 64)
        combined = torch.cat([cnn_out, mlp_out], dim=1)  # shape (B, 128)
        return self.fc(combined)              # shape (B, features_dim)
    
class SaveEveryNStepsCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = f"{self.save_path}/model_{self.num_timesteps}_steps"
            self.model.save(path)
            if self.verbose > 0:
                print(f"✔️ Model saved at {path}")
        return True

if __name__ == "__main__":
    # Instantiate the wrapped environment
    env = GazeboGymWrapper()
    time.sleep(4)  # let Gazebo stabilize

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # rospy.init_node("test_mover")
    pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(10)

    # twist = Twist()
    # twist.linear.x = 0.2   # move forward slowly
    # twist.angular.z = 0.2  # turn slightly

    # while not rospy.is_shutdown():
    #     pub.publish(twist)
    #     rate.sleep()

    # Create an action noise object for exploration.
    n_actions = env.action_space.shape[-1]
    # # Increase sigma to increase exploration noise.
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.8 * np.ones(n_actions))
    
    # # Example of updated hyperparameters for TD3
    model = TD3(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,           # smaller LR can help stabilize
        batch_size=256,               # bigger batch for image-based input
        tau=0.005,                    # polyak update
        policy_delay=2,               # update policy every 2 critic steps
        target_policy_noise=0.2,      # smoothing noise for target policy
        target_noise_clip=0.5,
        buffer_size=1_000_000,          # large replay buffer
        action_noise=action_noise,    # add exploration noise here
        verbose=1,
        tensorboard_log="./runs" 
    )
    # # Create the callback
    save_callback = SaveEveryNStepsCallback(
        save_freq=50000,               # every 1000 steps
        save_path="./checkpoints",    # folder to save models
        verbose=1,
    )

    # # Train with callback
    model.learn(total_timesteps=int(1e6), callback=save_callback, tb_log_name="turtle_obs_2")


    # # Save the trained model
    model.save("td3_gazebo_custom_policy_multiroom")
    print("Training complete and model saved!")


