#!/usr/bin/env python3
import time
import gym
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from gym import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
            torch.nn.Linear(observation_space.spaces["scalars"].shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
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


def save_results_to_excel(results, filename=None):
    """Save evaluation results to Excel file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.xlsx"
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = {
        'Metric': [
            'Total Episodes',
            'Success Rate (%)',
            'Average Reward',
            'Std Reward',
            'Max Reward',
            'Min Reward',
            'Average Steps',
            'Average Path Length',
            'Average Time (s)',
            'Total Successes',
            'Total Collisions',
            'Total Timeouts',
            'Total Stuck',
        ],
        'Value': [
            len(results),
            (sum(1 for r in results if r['success']) / len(results)) * 100,
            np.mean([r['total_reward'] for r in results]),
            np.std([r['total_reward'] for r in results]),
            np.max([r['total_reward'] for r in results]),
            np.min([r['total_reward'] for r in results]),
            np.mean([r['steps'] for r in results]),
            np.mean([r['path_length'] for r in results]),
            np.mean([r['episode_time'] for r in results]),
            sum(1 for r in results if r['success']),
            sum(1 for r in results if r['result'] == 'collision'),
            sum(1 for r in results if r['result'] == 'timeout'),
            sum(1 for r in results if r['result'] == 'stuck'),
        ]
    }
    df_summary = pd.DataFrame(summary)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Episode Details', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Results saved to: {filename}")
    return filename


def main():
    # Instantiate the environment
    env = GazeboGymWrapper()
    time.sleep(4)  # Allow Gazebo to stabilize

    # Load the trained model
    model_path = "td3_gazebo_custom_policy.zip"
    model = TD3.load(model_path, env=env)
    print("Loaded trained model from", model_path)

    # ============================================
    # EVALUATION WITH DATA COLLECTION
    # ============================================
    num_episodes = 100
    results = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        episode_start_time = time.time()
        
        # Track start position
        start_x = env.env.odom_x
        start_y = env.env.odom_y
        goal_x = env.env.goal_x
        goal_y = env.env.goal_y
        
        # Track path length
        path_length = 0.0
        last_x, last_y = start_x, start_y
        
        while not done:
            # Use deterministic policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Calculate path length
            current_x = env.env.odom_x
            current_y = env.env.odom_y
            path_length += np.sqrt((current_x - last_x)**2 + (current_y - last_y)**2)
            last_x, last_y = current_x, current_y
            
            time.sleep(0.1)
        
        episode_time = time.time() - episode_start_time
        
        # Determine result type
        final_dist = np.sqrt((env.env.odom_x - goal_x)**2 + (env.env.odom_y - goal_y)**2)
        success = final_dist < 1.0  # GOAL_REACHED_DIST
        
        # Determine result type based on reward/conditions
        if success:
            result_type = 'goal'
        elif total_reward <= -100:
            if env.env.stuck_counter >= 60:
                result_type = 'stuck'
            else:
                result_type = 'collision'
        else:
            result_type = 'timeout'
        
        # Calculate optimal path (straight line)
        optimal_path = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        spl = (optimal_path / max(path_length, optimal_path)) if success else 0.0
        
        # Store episode data
        episode_data = {
            'episode': ep + 1,
            'total_reward': round(total_reward, 2),
            'steps': steps,
            'episode_time': round(episode_time, 2),
            'success': success,
            'result': result_type,
            'start_x': round(start_x, 2),
            'start_y': round(start_y, 2),
            'goal_x': round(goal_x, 2),
            'goal_y': round(goal_y, 2),
            'final_x': round(env.env.odom_x, 2),
            'final_y': round(env.env.odom_y, 2),
            'final_distance': round(final_dist, 2),
            'path_length': round(path_length, 2),
            'optimal_path': round(optimal_path, 2),
            'spl': round(spl, 3),
        }
        results.append(episode_data)
        
        # Print progress
        print(f"Episode {ep+1}/{num_episodes} | Reward: {total_reward:.2f} | "
              f"Steps: {steps} | Result: {result_type} | SPL: {spl:.3f}")
        
        # Save intermediate results every 100 episodes
        if (ep + 1) % 100 == 0:
            save_results_to_excel(results, f"evaluation_intermediate_ep{ep+1}.xlsx")
    
    # ============================================
    # SAVE FINAL RESULTS
    # ============================================
    final_filename = save_results_to_excel(results)
    
    # Print final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Total Episodes:    {len(results)}")
    print(f"Success Rate:      {(sum(1 for r in results if r['success']) / len(results)) * 100:.1f}%")
    print(f"Average Reward:    {np.mean([r['total_reward'] for r in results]):.2f}")
    print(f"Average Steps:     {np.mean([r['steps'] for r in results]):.1f}")
    print(f"Average SPL:       {np.mean([r['spl'] for r in results]):.3f}")
    print(f"Results saved to:  {final_filename}")
    print("="*60)


if __name__ == "__main__":
    main()