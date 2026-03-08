#!/usr/bin/env python3
"""
Visualize and compare real vs synthetic data to diagnose diffusion quality.

Usage:
    python visualize_synthetic.py --real training_data.pkl --synthetic synthetic_data.pkl
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def compare_distributions(real_data, synthetic_data):
    """Compare distributions of real vs synthetic data."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Real (blue) vs Synthetic (orange) Distributions', fontsize=14)
    
    # Action distributions
    axes[0, 0].hist(real_data['actions'][:, 0], bins=50, alpha=0.7, label='Real', color='blue')
    axes[0, 0].hist(synthetic_data['actions'][:, 0], bins=50, alpha=0.7, label='Synthetic', color='orange')
    axes[0, 0].set_title('Linear Velocity')
    axes[0, 0].legend()
    
    axes[0, 1].hist(real_data['actions'][:, 1], bins=50, alpha=0.7, label='Real', color='blue')
    axes[0, 1].hist(synthetic_data['actions'][:, 1], bins=50, alpha=0.7, label='Synthetic', color='orange')
    axes[0, 1].set_title('Angular Velocity')
    axes[0, 1].legend()
    
    # Action scatter
    axes[0, 2].scatter(real_data['actions'][:, 0], real_data['actions'][:, 1], 
                       alpha=0.1, s=1, label='Real', color='blue')
    axes[0, 2].scatter(synthetic_data['actions'][:, 0], synthetic_data['actions'][:, 1], 
                       alpha=0.1, s=1, label='Synthetic', color='orange')
    axes[0, 2].set_xlabel('Linear Vel')
    axes[0, 2].set_ylabel('Angular Vel')
    axes[0, 2].set_title('Action Space Coverage')
    
    # Scalar distributions
    scalar_names = ['prev_lin', 'prev_ang', 'last_lin', 'last_ang', 'dist2goal', 'angle2goal', 'min_laser']
    
    for i, name in enumerate([4, 5, 6]):  # dist2goal, angle2goal, min_laser
        ax = axes[1, i]
        ax.hist(real_data['scalars'][:, name], bins=50, alpha=0.7, label='Real', color='blue')
        ax.hist(synthetic_data['scalars'][:, name], bins=50, alpha=0.7, label='Synthetic', color='orange')
        ax.set_title(scalar_names[name])
        ax.legend()
    
    # Depth image statistics
    real_depth_mean = real_data['depth_images'].mean(axis=(1, 2, 3))
    synthetic_depth_mean = synthetic_data['depth_images'].mean(axis=(1, 2, 3))
    
    axes[2, 0].hist(real_depth_mean, bins=50, alpha=0.7, label='Real', color='blue')
    axes[2, 0].hist(synthetic_depth_mean, bins=50, alpha=0.7, label='Synthetic', color='orange')
    axes[2, 0].set_title('Depth Image Mean Value')
    axes[2, 0].legend()
    
    real_depth_std = real_data['depth_images'].std(axis=(1, 2, 3))
    synthetic_depth_std = synthetic_data['depth_images'].std(axis=(1, 2, 3))
    
    axes[2, 1].hist(real_depth_std, bins=50, alpha=0.7, label='Real', color='blue')
    axes[2, 1].hist(synthetic_depth_std, bins=50, alpha=0.7, label='Synthetic', color='orange')
    axes[2, 1].set_title('Depth Image Std Dev')
    axes[2, 1].legend()
    
    # Depth image range
    real_depth_range = real_data['depth_images'].max(axis=(1, 2, 3)) - real_data['depth_images'].min(axis=(1, 2, 3))
    synthetic_depth_range = synthetic_data['depth_images'].max(axis=(1, 2, 3)) - synthetic_data['depth_images'].min(axis=(1, 2, 3))
    
    axes[2, 2].hist(real_depth_range, bins=50, alpha=0.7, label='Real', color='blue')
    axes[2, 2].hist(synthetic_depth_range, bins=50, alpha=0.7, label='Synthetic', color='orange')
    axes[2, 2].set_title('Depth Image Range')
    axes[2, 2].legend()
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=150)
    plt.show()
    print("Saved: distribution_comparison.png")


def visualize_sample_images(real_data, synthetic_data, n_samples=10):
    """Show sample depth images side by side."""
    
    fig, axes = plt.subplots(4, n_samples, figsize=(20, 8))
    fig.suptitle('Real vs Synthetic Depth Images', fontsize=14)
    
    # Random indices
    real_idx = np.random.choice(len(real_data['depth_images']), n_samples, replace=False)
    synth_idx = np.random.choice(len(synthetic_data['depth_images']), n_samples, replace=False)
    
    for i in range(n_samples):
        # Real depth image
        axes[0, i].imshow(real_data['depth_images'][real_idx[i], 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Real', fontsize=12)
        
        # Real action overlay
        real_action = real_data['actions'][real_idx[i]]
        axes[1, i].text(0.5, 0.5, f'L:{real_action[0]:.2f}\nA:{real_action[1]:.2f}', 
                       ha='center', va='center', fontsize=10)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Real Act', fontsize=12)
        
        # Synthetic depth image
        axes[2, i].imshow(synthetic_data['depth_images'][synth_idx[i], 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Synthetic', fontsize=12)
        
        # Synthetic action overlay
        synth_action = synthetic_data['actions'][synth_idx[i]]
        axes[3, i].text(0.5, 0.5, f'L:{synth_action[0]:.2f}\nA:{synth_action[1]:.2f}', 
                       ha='center', va='center', fontsize=10)
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_ylabel('Synth Act', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('sample_comparison.png', dpi=150)
    plt.show()
    print("Saved: sample_comparison.png")


def check_action_validity(synthetic_data):
    """Check if synthetic actions are within valid ranges."""
    actions = synthetic_data['actions']
    
    linear_valid = np.logical_and(actions[:, 0] >= 0, actions[:, 0] <= 1)
    angular_valid = np.logical_and(actions[:, 1] >= -1, actions[:, 1] <= 1)
    
    print("\n" + "="*50)
    print("ACTION VALIDITY CHECK")
    print("="*50)
    print(f"Linear velocity in [0, 1]:  {linear_valid.mean()*100:.1f}%")
    print(f"Angular velocity in [-1, 1]: {angular_valid.mean()*100:.1f}%")
    
    print(f"\nLinear vel range:  [{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}]")
    print(f"Angular vel range: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]")
    
    # Check for NaN or Inf
    print(f"\nNaN in actions: {np.isnan(actions).any()}")
    print(f"Inf in actions: {np.isinf(actions).any()}")


def check_consistency(synthetic_data):
    """Check if scalars and depth images are consistent."""
    
    print("\n" + "="*50)
    print("CONSISTENCY CHECK")
    print("="*50)
    
    # min_laser is scalar index 6
    # It should correlate with minimum values in depth image
    min_laser = synthetic_data['scalars'][:, 6]
    depth_min = synthetic_data['depth_images'].min(axis=(1, 2, 3))
    
    correlation = np.corrcoef(min_laser, depth_min)[0, 1]
    print(f"Correlation (min_laser vs depth_min): {correlation:.3f}")
    
    if correlation < 0.3:
        print("⚠️  LOW CORRELATION - Scalars and depth images may be inconsistent!")
    else:
        print("✓ Reasonable correlation")
    
    # Check prev_action vs last_action consistency
    # In real data, these should often be similar (smooth motion)
    prev_lin = synthetic_data['scalars'][:, 0]
    last_lin = synthetic_data['scalars'][:, 2]
    action_lin = synthetic_data['actions'][:, 0]
    
    diff_prev_last = np.abs(prev_lin - last_lin).mean()
    diff_last_action = np.abs(last_lin - action_lin).mean()
    
    print(f"\nAvg |prev_lin - last_lin|: {diff_prev_last:.3f}")
    print(f"Avg |last_lin - action_lin|: {diff_last_action:.3f}")


def print_statistics(real_data, synthetic_data):
    """Print comparative statistics."""
    
    print("\n" + "="*50)
    print("STATISTICAL COMPARISON")
    print("="*50)
    
    print("\n--- Actions ---")
    print(f"{'Metric':<20} {'Real':>12} {'Synthetic':>12}")
    print("-" * 46)
    print(f"{'Linear mean':<20} {real_data['actions'][:, 0].mean():>12.4f} {synthetic_data['actions'][:, 0].mean():>12.4f}")
    print(f"{'Linear std':<20} {real_data['actions'][:, 0].std():>12.4f} {synthetic_data['actions'][:, 0].std():>12.4f}")
    print(f"{'Angular mean':<20} {real_data['actions'][:, 1].mean():>12.4f} {synthetic_data['actions'][:, 1].mean():>12.4f}")
    print(f"{'Angular std':<20} {real_data['actions'][:, 1].std():>12.4f} {synthetic_data['actions'][:, 1].std():>12.4f}")
    
    print("\n--- Key Scalars ---")
    print(f"{'dist2goal mean':<20} {real_data['scalars'][:, 4].mean():>12.4f} {synthetic_data['scalars'][:, 4].mean():>12.4f}")
    print(f"{'dist2goal std':<20} {real_data['scalars'][:, 4].std():>12.4f} {synthetic_data['scalars'][:, 4].std():>12.4f}")
    print(f"{'angle2goal mean':<20} {real_data['scalars'][:, 5].mean():>12.4f} {synthetic_data['scalars'][:, 5].mean():>12.4f}")
    print(f"{'min_laser mean':<20} {real_data['scalars'][:, 6].mean():>12.4f} {synthetic_data['scalars'][:, 6].mean():>12.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', type=str, required=True)
    parser.add_argument('--synthetic', type=str, required=True)
    args = parser.parse_args()
    
    print("Loading data...")
    real_data = load_data(args.real)
    synthetic_data = load_data(args.synthetic)
    
    print(f"Real samples: {len(real_data['actions'])}")
    print(f"Synthetic samples: {len(synthetic_data['actions'])}")
    
    print_statistics(real_data, synthetic_data)
    check_action_validity(synthetic_data)
    check_consistency(synthetic_data)
    compare_distributions(real_data, synthetic_data)
    visualize_sample_images(real_data, synthetic_data)


if __name__ == "__main__":
    main()