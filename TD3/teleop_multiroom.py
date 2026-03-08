#!/usr/bin/env python3
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from datetime import datetime
from gym import spaces
from geometry_msgs.msg import Twist
import rospy
from pynput import keyboard
import threading

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
        print('action', action[0], action[1])
        next_obs, reward, done, target = self.env.step(action)
        # Wrap the "target" flag into info
        info = {"target": target}
        return {"image": next_obs[0], "scalars": next_obs[1]}, reward, done, info


############################################
# Teleoperation Controller
############################################
class TeleoperationController:
    """
    Keyboard-based teleoperation controller.
    
    Controls:
        W / Up Arrow    : Increase linear velocity (forward)
        S / Down Arrow  : FULL STOP (zero both velocities)
        A / Left Arrow  : Turn left
        D / Right Arrow : Turn right
        Space           : Emergency stop (zero velocities)
        R               : Reset episode
        Q               : Quit and save data
    """
    def __init__(self):
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        # Velocity limits
        self.max_linear = 1.0
        self.min_linear = 0.0
        self.max_angular = 1.0
        self.min_angular = -1.0
        
        # Velocity increments (INCREASED for faster response)
        self.linear_increment = 0.15   # was 0.05
        self.angular_increment = 0.25  # was 0.1
        
        # Control flags
        self.reset_flag = False
        self.quit_flag = False
        
        # Key states for smooth control
        self.keys_pressed = set()
        
        # Start keyboard listener in separate thread
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
        
    def on_press(self, key):
        try:
            if key.char == 'w':
                self.keys_pressed.add('w')
            elif key.char == 's':
                # S = FULL STOP
                self.linear_vel = 0.0
                self.angular_vel = 0.0
            elif key.char == 'a':
                self.keys_pressed.add('a')
            elif key.char == 'd':
                self.keys_pressed.add('d')
            elif key.char == ' ':
                self.linear_vel = 0.0
                self.angular_vel = 0.0
            elif key.char == 'r':
                self.reset_flag = True
            elif key.char == 'q':
                self.quit_flag = True
        except AttributeError:
            # Handle special keys (arrows)
            if key == keyboard.Key.up:
                self.keys_pressed.add('up')
            elif key == keyboard.Key.down:
                # Down arrow = FULL STOP
                self.linear_vel = 0.0
                self.angular_vel = 0.0
            elif key == keyboard.Key.left:
                self.keys_pressed.add('left')
            elif key == keyboard.Key.right:
                self.keys_pressed.add('right')
            elif key == keyboard.Key.space:
                self.linear_vel = 0.0
                self.angular_vel = 0.0
                
    def on_release(self, key):
        try:
            if key.char in self.keys_pressed:
                self.keys_pressed.discard(key.char)
        except AttributeError:
            if key == keyboard.Key.up:
                self.keys_pressed.discard('up')
            elif key == keyboard.Key.left:
                self.keys_pressed.discard('left')
            elif key == keyboard.Key.right:
                self.keys_pressed.discard('right')
    
    def update_velocities(self):
        """Update velocities based on currently pressed keys."""
        # Linear velocity (W or Up)
        if 'w' in self.keys_pressed or 'up' in self.keys_pressed:
            self.linear_vel = min(self.linear_vel + self.linear_increment, self.max_linear)
            
        # Angular velocity (A/D or Left/Right)
        if 'a' in self.keys_pressed or 'left' in self.keys_pressed:
            self.angular_vel = min(self.angular_vel + self.angular_increment, self.max_angular)
        if 'd' in self.keys_pressed or 'right' in self.keys_pressed:
            self.angular_vel = max(self.angular_vel - self.angular_increment, self.min_angular)
            
        # Apply small decay to angular velocity for smoother control
        if 'a' not in self.keys_pressed and 'left' not in self.keys_pressed and \
           'd' not in self.keys_pressed and 'right' not in self.keys_pressed:
            self.angular_vel *= 0.85  # Decay factor (slightly faster decay)
            if abs(self.angular_vel) < 0.02:
                self.angular_vel = 0.0
    
    def get_action(self):
        """Get current action as numpy array."""
        self.update_velocities()
        return np.array([self.linear_vel, self.angular_vel], dtype=np.float32)
    
    def check_reset(self):
        """Check if reset was requested."""
        if self.reset_flag:
            self.reset_flag = False
            self.linear_vel = 0.0
            self.angular_vel = 0.0
            return True
        return False
    
    def check_quit(self):
        """Check if quit was requested."""
        return self.quit_flag
    
    def stop(self):
        """Stop the keyboard listener."""
        self.listener.stop()


############################################
# Data Recorder
############################################
class DataRecorder:
    """
    Records teleoperation data (observations and actions).
    
    Saves data in the format needed for behavior cloning:
    - depth_image: (1, 64, 64) normalized depth image
    - scalars: (7,) array [prev_lin, prev_ang, last_lin, last_ang, dist2goal, angle2goal, min_laser]
    - action: (2,) array [linear_vel, angular_vel]
    - metadata: episode_id, timestep, robot_pose, goal_pose
    """
    def __init__(self, save_dir="./teleop_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_data = []
        self.episode_id = self._get_next_episode_id()
        self.timestep = 0
        
        # Statistics
        self.total_steps = 0
        self.successful_episodes = 0
        self.failed_episodes = 0
        
    def _get_next_episode_id(self):
        """Find the next available episode ID."""
        existing = [f for f in os.listdir(self.save_dir) if f.startswith("episode_")]
        if not existing:
            return 0
        ids = [int(f.split("_")[1].split(".")[0]) for f in existing]
        return max(ids) + 1
    
    def record_step(self, observation, action, robot_pose=None, goal_pose=None):
        """Record a single timestep."""
        data_point = {
            # Observation components
            "depth_image": observation["image"].copy(),      # (1, 64, 64)
            "scalars": observation["scalars"].copy(),        # (7,)
            
            # Action (label for supervised learning)
            "action": action.copy(),                         # (2,)
            
            # Metadata
            "episode_id": self.episode_id,
            "timestep": self.timestep,
            "timestamp": time.time(),
        }
        
        # Optional pose information
        if robot_pose is not None:
            data_point["robot_pose"] = robot_pose.copy()
        if goal_pose is not None:
            data_point["goal_pose"] = goal_pose.copy()
            
        self.episode_data.append(data_point)
        self.timestep += 1
        self.total_steps += 1
        
    def save_episode(self, success=False, collision=False):
        """Save current episode to disk."""
        if len(self.episode_data) == 0:
            print("No data to save for this episode.")
            return
            
        # Add episode metadata
        episode_info = {
            "episode_id": self.episode_id,
            "num_steps": len(self.episode_data),
            "success": success,
            "collision": collision,
            "timestamp": datetime.now().isoformat(),
            "data": self.episode_data
        }
        
        # Save to file
        filename = f"episode_{self.episode_id:04d}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(episode_info, f)
            
        print(f"\n✅ Episode {self.episode_id} saved: {len(self.episode_data)} steps, "
              f"success={success}, collision={collision}")
        
        # Update statistics
        if success:
            self.successful_episodes += 1
        if collision:
            self.failed_episodes += 1
            
        # Reset for next episode
        self.episode_data = []
        self.episode_id += 1
        self.timestep = 0
        
    def get_statistics(self):
        """Return recording statistics."""
        return {
            "total_episodes": self.episode_id,
            "successful_episodes": self.successful_episodes,
            "failed_episodes": self.failed_episodes,
            "total_steps": self.total_steps,
        }
    
    def discard_episode(self):
        """Discard current episode without saving."""
        print(f"\n❌ Episode {self.episode_id} discarded: {len(self.episode_data)} steps")
        self.episode_data = []
        self.timestep = 0


############################################
# Main Teleoperation Loop
############################################
def print_controls():
    """Print control instructions."""
    print("\n" + "="*50)
    print("TELEOPERATION CONTROLS")
    print("="*50)
    print("W / ↑  : Accelerate forward")
    print("S / ↓  : FULL STOP (both velocities to zero)")
    print("A / ←  : Turn left")
    print("D / →  : Turn right")
    print("SPACE  : Emergency stop")
    print("R      : Reset episode (discard current)")
    print("Q      : Quit and save all data")
    print("="*50 + "\n")


def main():
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./teleop_data_{timestamp}"
    
    # Instantiate the wrapped environment
    print("Initializing environment...")
    env = GazeboGymWrapper()
    time.sleep(4)  # Let Gazebo stabilize
    
    # Initialize controller and recorder
    teleop = TeleoperationController()
    recorder = DataRecorder(save_dir=save_dir)
    
    print_controls()
    print(f"Data will be saved to: {save_dir}")
    print("Starting teleoperation...\n")
    
    # Initial reset
    obs = env.reset()
    episode_steps = 0
    
    try:
        while not teleop.check_quit():
            # Check for manual reset
            if teleop.check_reset():
                print("\n🔄 Manual reset requested...")
                recorder.discard_episode()
                obs = env.reset()
                episode_steps = 0
                continue
            
            # Get action from teleoperator
            action = teleop.get_action()
            
            # Record current observation and action
            recorder.record_step(
                observation=obs,
                action=action,
                robot_pose=np.array([env.env.odom_x, env.env.odom_y, env.env.odom_yaw]),
                goal_pose=np.array([env.env.goal_x, env.env.goal_y])
            )
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            episode_steps += 1
            
            # Display status
            print(f"\rEp: {recorder.episode_id} | Step: {episode_steps:4d} | "
                  f"Lin: {action[0]:+.2f} | Ang: {action[1]:+.2f} | "
                  f"Dist2Goal: {obs['scalars'][4]:.2f} | "
                  f"Total: {recorder.total_steps}", end="")
            
            # Handle episode termination
            if done:
                success = info.get("target", False)
                collision = not success  # If done but not success, assume collision/stuck
                
                if success:
                    print(f"\n🎯 GOAL REACHED!")
                else:
                    print(f"\n💥 Episode ended (collision/stuck)")
                
                recorder.save_episode(success=success, collision=collision)
                obs = env.reset()
                episode_steps = 0
            else:
                obs = next_obs
                
            # Small delay for control loop
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    
    finally:
        # Clean up
        teleop.stop()
        
        # Save any remaining episode data
        if len(recorder.episode_data) > 0:
            print("\nSaving remaining episode data...")
            recorder.save_episode(success=False, collision=False)
        
        # Print final statistics
        stats = recorder.get_statistics()
        print("\n" + "="*50)
        print("TELEOPERATION COMPLETE")
        print("="*50)
        print(f"Total episodes: {stats['total_episodes']}")
        print(f"Successful:     {stats['successful_episodes']}")
        print(f"Failed:         {stats['failed_episodes']}")
        print(f"Total steps:    {stats['total_steps']}")
        print(f"Data saved to:  {save_dir}")
        print("="*50)


if __name__ == "__main__":
    main()