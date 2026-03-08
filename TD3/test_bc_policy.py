#!/usr/bin/env python3
"""
Test script for trained Behavior Cloning policy.

Usage:
    python test_bc_policy.py --model bc_policy.pth --episodes 10
    python test_bc_policy.py --model bc_policy.pth --episodes 20 --render
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# Import environment
from realsense_env import GazeboEnv


# ============================================
# BC Policy Model (must match training architecture)
# ============================================
class BCPolicy(nn.Module):
    """
    Behavior Cloning policy: observation → action
    """
    def __init__(self):
        super().__init__()
        
        # CNN for depth processing
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # MLP for scalars
        self.scalar_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, depth_image, scalars):
        # Process image
        img_features = self.cnn(depth_image)
        img_features = img_features.view(img_features.size(0), -1)
        
        # Process scalars
        scalar_features = self.scalar_mlp(scalars)
        
        # Combine and predict action
        combined = torch.cat([img_features, scalar_features], dim=1)
        action = self.action_head(combined)
        
        return action


# ============================================
# Environment Wrapper (same as training)
# ============================================
class GazeboGymWrapper:
    """
    Wraps GazeboEnv for testing.
    """
    def __init__(self):
        super(GazeboGymWrapper, self).__init__()
        self.env = GazeboEnv()
        
    def reset(self):
        obs = self.env.reset()
        return {"image": obs[0], "scalars": obs[1]}
    
    def step(self, action):
        next_obs, reward, done, target = self.env.step(action)
        info = {"target": target}
        return {"image": next_obs[0], "scalars": next_obs[1]}, reward, done, info


# ============================================
# Testing Functions
# ============================================
def load_model(model_path, device='cpu'):
    """Load trained BC model."""
    model = BCPolicy()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    return model


def predict_action(model, observation, device='cpu'):
    """
    Get action from model given observation.
    
    Args:
        model: trained BC policy
        observation: dict with 'image' (1, 64, 64) and 'scalars' (7,)
        
    Returns:
        action: numpy array (2,) [linear_vel, angular_vel]
    """
    with torch.no_grad():
        # Prepare inputs (add batch dimension)
        depth_image = torch.FloatTensor(observation['image']).unsqueeze(0).to(device)  # (1, 1, 64, 64)
        scalars = torch.FloatTensor(observation['scalars']).unsqueeze(0).to(device)    # (1, 7)
        
        # Predict action
        action = model(depth_image, scalars)  # (1, 2)
        action = action.cpu().numpy().squeeze()  # (2,)
        
    return action


def clip_action(action, linear_range=(0.0, 1.0), angular_range=(-1.0, 1.0)):
    """Clip action to valid range."""
    action[0] = np.clip(action[0], linear_range[0], linear_range[1])
    action[1] = np.clip(action[1], angular_range[0], angular_range[1])
    return action


def run_episode(env, model, device='cpu', max_steps=500, verbose=True):
    """
    Run a single episode with the trained policy.
    
    Returns:
        dict with episode statistics
    """
    obs = env.reset()
    
    total_reward = 0
    steps = 0
    done = False
    success = False
    
    # Track trajectory
    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
    }
    
    while not done and steps < max_steps:
        # Get action from model
        action = predict_action(model, obs, device)
        action = clip_action(action)
        
        # Store trajectory
        trajectory['observations'].append({
            'scalars': obs['scalars'].copy()
        })
        trajectory['actions'].append(action.copy())
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        trajectory['rewards'].append(reward)
        total_reward += reward
        steps += 1
        
        # Print status
        if verbose:
            dist_to_goal = obs['scalars'][4]
            angle_to_goal = obs['scalars'][5]
            min_laser = obs['scalars'][6]
            print(f"\rStep {steps:4d} | "
                  f"Lin: {action[0]:+.2f} | Ang: {action[1]:+.2f} | "
                  f"Dist: {dist_to_goal:.2f} | Angle: {angle_to_goal:+.2f} | "
                  f"Laser: {min_laser:.2f} | Reward: {reward:+.2f}", end="")
        
        obs = next_obs
        success = info.get('target', False)
    
    if verbose:
        print()  # New line after episode
    
    return {
        'success': success,
        'collision': done and not success,
        'steps': steps,
        'total_reward': total_reward,
        'trajectory': trajectory,
    }


def run_evaluation(env, model, num_episodes=10, device='cpu', verbose=True):
    """
    Run multiple episodes and compute statistics.
    """
    results = []
    
    print("\n" + "="*60)
    print(f"EVALUATING BC POLICY - {num_episodes} EPISODES")
    print("="*60 + "\n")
    
    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")
        
        result = run_episode(env, model, device=device, verbose=verbose)
        results.append(result)
        
        status = "🎯 SUCCESS" if result['success'] else "💥 FAILED"
        print(f"\n{status} | Steps: {result['steps']} | Total Reward: {result['total_reward']:.2f}")
    
    # Compute statistics
    successes = sum(1 for r in results if r['success'])
    collisions = sum(1 for r in results if r['collision'])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['total_reward'] for r in results])
    
    # Success-only statistics
    success_results = [r for r in results if r['success']]
    if success_results:
        avg_steps_success = np.mean([r['steps'] for r in success_results])
    else:
        avg_steps_success = 0
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes:        {num_episodes}")
    print(f"Successes:       {successes} ({100*successes/num_episodes:.1f}%)")
    print(f"Collisions:      {collisions} ({100*collisions/num_episodes:.1f}%)")
    print(f"Timeouts:        {num_episodes - successes - collisions}")
    print(f"Avg Steps:       {avg_steps:.1f}")
    print(f"Avg Steps (success): {avg_steps_success:.1f}")
    print(f"Avg Reward:      {avg_reward:.2f}")
    print("="*60)
    
    return {
        'num_episodes': num_episodes,
        'successes': successes,
        'success_rate': successes / num_episodes,
        'collisions': collisions,
        'collision_rate': collisions / num_episodes,
        'avg_steps': avg_steps,
        'avg_steps_success': avg_steps_success,
        'avg_reward': avg_reward,
        'results': results,
    }


# ============================================
# Interactive Mode
# ============================================
def run_interactive(env, model, device='cpu'):
    """
    Run in interactive mode - press Enter for next episode, 'q' to quit.
    """
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Press ENTER to run next episode")
    print("Press 'q' + ENTER to quit")
    print("="*60 + "\n")
    
    episode = 0
    total_successes = 0
    
    while True:
        user_input = input(f"\nRun episode {episode+1}? [Enter/q]: ").strip().lower()
        
        if user_input == 'q':
            break
        
        result = run_episode(env, model, device=device, verbose=True)
        episode += 1
        
        if result['success']:
            total_successes += 1
            print(f"\n🎯 SUCCESS! Steps: {result['steps']}")
        else:
            print(f"\n💥 FAILED. Steps: {result['steps']}")
        
        print(f"Running success rate: {total_successes}/{episode} ({100*total_successes/episode:.1f}%)")
    
    print(f"\n\nFinal: {total_successes}/{episode} successes ({100*total_successes/episode:.1f}%)")


# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Test trained BC policy')
    parser.add_argument('--model', type=str, default='bc_policy.pth',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run on')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Initialize environment
    print("Initializing environment...")
    env = GazeboGymWrapper()
    time.sleep(4)  # Let Gazebo stabilize
    
    # Load model
    model = load_model(args.model, device=args.device)
    
    # Run evaluation
    if args.interactive:
        run_interactive(env, model, device=args.device)
    else:
        results = run_evaluation(
            env, model, 
            num_episodes=args.episodes, 
            device=args.device,
            verbose=not args.quiet
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()