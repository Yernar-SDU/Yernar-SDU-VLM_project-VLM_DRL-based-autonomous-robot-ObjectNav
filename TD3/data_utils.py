#!/usr/bin/env python3
"""
Utility script to load, inspect, and prepare teleoperation data for training.

Usage:
    python data_utils.py --data_dir ./teleop_data_XXXXXXXX --action inspect
    python data_utils.py --data_dir ./teleop_data_XXXXXXXX --action stats
    python data_utils.py --data_dir ./teleop_data_XXXXXXXX --action export --output ./training_data.pkl
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


def load_all_episodes(data_dir: str) -> List[Dict]:
    """Load all episode files from a directory."""
    episodes = []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f:
            episode = pickle.load(f)
            episodes.append(episode)
    
    print(f"Loaded {len(episodes)} episodes from {data_dir}")
    return episodes


def get_statistics(episodes: List[Dict]) -> Dict:
    """Compute statistics from episodes."""
    total_steps = sum(ep['num_steps'] for ep in episodes)
    successful = sum(1 for ep in episodes if ep.get('success', False))
    collisions = sum(1 for ep in episodes if ep.get('collision', False))
    
    # Action statistics
    all_actions = []
    for ep in episodes:
        for step in ep['data']:
            all_actions.append(step['action'])
    all_actions = np.array(all_actions)
    
    # Scalar statistics
    all_scalars = []
    for ep in episodes:
        for step in ep['data']:
            all_scalars.append(step['scalars'])
    all_scalars = np.array(all_scalars)
    
    stats = {
        'num_episodes': len(episodes),
        'total_steps': total_steps,
        'successful_episodes': successful,
        'collision_episodes': collisions,
        'avg_episode_length': total_steps / len(episodes) if episodes else 0,
        'action_stats': {
            'linear_vel': {
                'mean': all_actions[:, 0].mean(),
                'std': all_actions[:, 0].std(),
                'min': all_actions[:, 0].min(),
                'max': all_actions[:, 0].max(),
            },
            'angular_vel': {
                'mean': all_actions[:, 1].mean(),
                'std': all_actions[:, 1].std(),
                'min': all_actions[:, 1].min(),
                'max': all_actions[:, 1].max(),
            }
        },
        'scalar_stats': {
            'dist_to_goal': {
                'mean': all_scalars[:, 4].mean(),
                'std': all_scalars[:, 4].std(),
                'min': all_scalars[:, 4].min(),
                'max': all_scalars[:, 4].max(),
            },
            'angle_to_goal': {
                'mean': all_scalars[:, 5].mean(),
                'std': all_scalars[:, 5].std(),
                'min': all_scalars[:, 5].min(),
                'max': all_scalars[:, 5].max(),
            },
            'min_laser': {
                'mean': all_scalars[:, 6].mean(),
                'std': all_scalars[:, 6].std(),
                'min': all_scalars[:, 6].min(),
                'max': all_scalars[:, 6].max(),
            }
        }
    }
    return stats


def print_statistics(stats: Dict):
    """Pretty print statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Number of episodes:     {stats['num_episodes']}")
    print(f"Total steps:            {stats['total_steps']}")
    print(f"Successful episodes:    {stats['successful_episodes']}")
    print(f"Collision episodes:     {stats['collision_episodes']}")
    print(f"Avg episode length:     {stats['avg_episode_length']:.1f}")
    
    print("\n--- Action Statistics ---")
    for action_name, action_stats in stats['action_stats'].items():
        print(f"{action_name}:")
        print(f"  mean: {action_stats['mean']:+.4f}, std: {action_stats['std']:.4f}")
        print(f"  min:  {action_stats['min']:+.4f}, max: {action_stats['max']:+.4f}")
    
    print("\n--- Scalar Observation Statistics ---")
    for scalar_name, scalar_stats in stats['scalar_stats'].items():
        print(f"{scalar_name}:")
        print(f"  mean: {scalar_stats['mean']:+.4f}, std: {scalar_stats['std']:.4f}")
        print(f"  min:  {scalar_stats['min']:+.4f}, max: {scalar_stats['max']:+.4f}")
    print("="*60)


def visualize_episode(episode: Dict, save_path: Optional[str] = None):
    """Visualize a single episode."""
    data = episode['data']
    steps = len(data)
    
    # Extract data
    linear_vels = [d['action'][0] for d in data]
    angular_vels = [d['action'][1] for d in data]
    dist_to_goal = [d['scalars'][4] for d in data]
    angle_to_goal = [d['scalars'][5] for d in data]
    min_laser = [d['scalars'][6] for d in data]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Episode {episode['episode_id']} - "
                 f"{'SUCCESS' if episode.get('success') else 'FAILED'} - "
                 f"{steps} steps")
    
    # Actions
    axes[0, 0].plot(linear_vels, label='Linear Vel', color='blue')
    axes[0, 0].set_ylabel('Linear Velocity')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(angular_vels, label='Angular Vel', color='orange')
    axes[0, 1].set_ylabel('Angular Velocity')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Goal info
    axes[1, 0].plot(dist_to_goal, label='Distance to Goal', color='green')
    axes[1, 0].set_ylabel('Distance')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(angle_to_goal, label='Angle to Goal', color='red')
    axes[1, 1].set_ylabel('Angle (rad)')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Obstacle info
    axes[2, 0].plot(min_laser, label='Min Laser Distance', color='purple')
    axes[2, 0].set_ylabel('Distance')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # Sample depth images
    sample_indices = [0, steps//4, steps//2, 3*steps//4, steps-1]
    for i, idx in enumerate(sample_indices):
        if idx < steps:
            ax = axes[2, 1] if i == 0 else None
            # We'll just show the first one for now
            if i == 0:
                depth_img = data[idx]['depth_image'][0]  # Remove channel dim
                axes[2, 1].imshow(depth_img, cmap='gray')
                axes[2, 1].set_title(f'Depth at step {idx}')
                axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_action_distribution(episodes: List[Dict], save_path: Optional[str] = None):
    """Visualize the distribution of actions."""
    all_actions = []
    for ep in episodes:
        for step in ep['data']:
            all_actions.append(step['action'])
    all_actions = np.array(all_actions)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Linear velocity histogram
    axes[0].hist(all_actions[:, 0], bins=50, color='blue', alpha=0.7)
    axes[0].set_xlabel('Linear Velocity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Linear Velocity Distribution')
    
    # Angular velocity histogram
    axes[1].hist(all_actions[:, 1], bins=50, color='orange', alpha=0.7)
    axes[1].set_xlabel('Angular Velocity')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Angular Velocity Distribution')
    
    # 2D scatter
    axes[2].scatter(all_actions[:, 0], all_actions[:, 1], alpha=0.1, s=1)
    axes[2].set_xlabel('Linear Velocity')
    axes[2].set_ylabel('Angular Velocity')
    axes[2].set_title('Action Space Coverage')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved action distribution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def export_for_training(episodes: List[Dict], output_path: str, 
                        filter_success_only: bool = False,
                        filter_min_steps: int = 10):
    """
    Export episodes to a single file optimized for training.
    
    Output format:
    {
        'depth_images': np.array of shape (N, 1, 64, 64),
        'scalars': np.array of shape (N, 7),
        'actions': np.array of shape (N, 2),
        'episode_ids': np.array of shape (N,),
        'timesteps': np.array of shape (N,),
        'metadata': {...}
    }
    """
    depth_images = []
    scalars = []
    actions = []
    episode_ids = []
    timesteps = []
    
    filtered_count = 0
    
    for ep in episodes:
        # Apply filters
        if filter_success_only and not ep.get('success', False):
            filtered_count += 1
            continue
        if ep['num_steps'] < filter_min_steps:
            filtered_count += 1
            continue
            
        for step in ep['data']:
            depth_images.append(step['depth_image'])
            scalars.append(step['scalars'])
            actions.append(step['action'])
            episode_ids.append(ep['episode_id'])
            timesteps.append(step['timestep'])
    
    training_data = {
        'depth_images': np.array(depth_images, dtype=np.float32),
        'scalars': np.array(scalars, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'episode_ids': np.array(episode_ids, dtype=np.int32),
        'timesteps': np.array(timesteps, dtype=np.int32),
        'metadata': {
            'total_samples': len(actions),
            'num_episodes_used': len(episodes) - filtered_count,
            'num_episodes_filtered': filtered_count,
            'filter_success_only': filter_success_only,
            'filter_min_steps': filter_min_steps,
            'observation_shape': {
                'depth_image': (1, 64, 64),
                'scalars': (7,),
            },
            'action_shape': (2,),
            'scalar_names': ['prev_lin', 'prev_ang', 'last_lin', 'last_ang', 
                           'dist2goal', 'angle2goal', 'min_laser'],
            'action_names': ['linear_vel', 'angular_vel'],
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"\n✅ Exported {len(actions)} samples to {output_path}")
    print(f"   Episodes used: {len(episodes) - filtered_count}")
    print(f"   Episodes filtered: {filtered_count}")
    print(f"   Depth images shape: {training_data['depth_images'].shape}")
    print(f"   Scalars shape: {training_data['scalars'].shape}")
    print(f"   Actions shape: {training_data['actions'].shape}")
    
    return training_data


def create_train_val_split(data_path: str, val_ratio: float = 0.2, 
                           split_by_episode: bool = True):
    """
    Create train/validation split from exported data.
    
    Args:
        data_path: Path to exported training data
        val_ratio: Fraction of data to use for validation
        split_by_episode: If True, split by episode (no data leakage). 
                         If False, random split.
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    n_samples = len(data['actions'])
    
    if split_by_episode:
        # Split by episode to avoid data leakage
        unique_episodes = np.unique(data['episode_ids'])
        np.random.shuffle(unique_episodes)
        
        n_val_episodes = int(len(unique_episodes) * val_ratio)
        val_episodes = set(unique_episodes[:n_val_episodes])
        
        train_mask = np.array([ep not in val_episodes for ep in data['episode_ids']])
        val_mask = ~train_mask
    else:
        # Random split
        indices = np.random.permutation(n_samples)
        n_val = int(n_samples * val_ratio)
        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[indices[:n_val]] = True
        train_mask = ~val_mask
    
    train_data = {
        'depth_images': data['depth_images'][train_mask],
        'scalars': data['scalars'][train_mask],
        'actions': data['actions'][train_mask],
        'metadata': data['metadata'].copy()
    }
    
    val_data = {
        'depth_images': data['depth_images'][val_mask],
        'scalars': data['scalars'][val_mask],
        'actions': data['actions'][val_mask],
        'metadata': data['metadata'].copy()
    }
    
    # Save splits
    base_path = data_path.rsplit('.', 1)[0]
    train_path = f"{base_path}_train.pkl"
    val_path = f"{base_path}_val.pkl"
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    
    print(f"\n✅ Created train/val split:")
    print(f"   Train: {len(train_data['actions'])} samples → {train_path}")
    print(f"   Val:   {len(val_data['actions'])} samples → {val_path}")
    
    return train_data, val_data


############################################
# PyTorch Dataset Class for Training
############################################
class TeleoperationDataset:
    """
    PyTorch-compatible dataset for loading teleoperation data.
    
    Usage:
        dataset = TeleoperationDataset('./training_data.pkl')
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.depth_images = data['depth_images']
        self.scalars = data['scalars']
        self.actions = data['actions']
        self.metadata = data['metadata']
        
        print(f"Loaded dataset with {len(self)} samples")
        
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return {
            'image': self.depth_images[idx],
            'scalars': self.scalars[idx],
            'action': self.actions[idx]
        }


############################################
# Main CLI
############################################
def main():
    parser = argparse.ArgumentParser(description='Teleoperation data utilities')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing episode files')
    parser.add_argument('--action', type=str, required=True,
                       choices=['inspect', 'stats', 'visualize', 'export', 'split'],
                       help='Action to perform')
    parser.add_argument('--output', type=str, default='./training_data.pkl',
                       help='Output path for export')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode index to visualize')
    parser.add_argument('--success_only', action='store_true',
                       help='Filter to successful episodes only')
    parser.add_argument('--min_steps', type=int, default=10,
                       help='Minimum steps per episode')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    if args.action == 'inspect':
        episodes = load_all_episodes(args.data_dir)
        for i, ep in enumerate(episodes[:5]):  # Show first 5
            print(f"\nEpisode {i}: {ep['num_steps']} steps, "
                  f"success={ep.get('success', 'N/A')}, "
                  f"collision={ep.get('collision', 'N/A')}")
            if ep['data']:
                sample = ep['data'][0]
                print(f"  Sample keys: {sample.keys()}")
                print(f"  Depth shape: {sample['depth_image'].shape}")
                print(f"  Scalars shape: {sample['scalars'].shape}")
                print(f"  Action shape: {sample['action'].shape}")
                
    elif args.action == 'stats':
        episodes = load_all_episodes(args.data_dir)
        stats = get_statistics(episodes)
        print_statistics(stats)
        
    elif args.action == 'visualize':
        episodes = load_all_episodes(args.data_dir)
        if args.episode < len(episodes):
            visualize_episode(episodes[args.episode])
            visualize_action_distribution(episodes)
        else:
            print(f"Episode {args.episode} not found. Available: 0-{len(episodes)-1}")
            
    elif args.action == 'export':
        episodes = load_all_episodes(args.data_dir)
        export_for_training(
            episodes, 
            args.output,
            filter_success_only=args.success_only,
            filter_min_steps=args.min_steps
        )
        
    elif args.action == 'split':
        create_train_val_split(args.output, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()