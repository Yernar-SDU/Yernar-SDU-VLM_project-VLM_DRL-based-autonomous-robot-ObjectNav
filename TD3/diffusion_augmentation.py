#!/usr/bin/env python3
"""
Diffusion Model for Teleoperation Data Augmentation

This module provides:
1. Diffusion model architecture for (depth_image, scalars, action) generation
2. Training pipeline
3. Synthetic data generation

Usage:
    # Train diffusion model
    python diffusion_augmentation.py --mode train --data training_data.pkl --epochs 500
    
    # Generate synthetic data
    python diffusion_augmentation.py --mode generate --model diffusion_model.pth --num_samples 5000 --output synthetic_data.pkl
    
    # Combine real + synthetic data
    python diffusion_augmentation.py --mode combine --real training_data.pkl --synthetic synthetic_data.pkl --output combined_data.pkl
"""

import os
import argparse
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================
# Diffusion Utilities
# ============================================

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear schedule for noise levels."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule (better for images)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def get_index_from_list(vals, t, x_shape):
    """Get value at timestep t and reshape for broadcasting."""
    batch_size = t.shape[0]
    # out = vals.gather(-1, t.cpu())
    out = vals.gather(-1, t)

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionUtils:
    """Utility class for diffusion forward/reverse process."""
    
    def __init__(self, timesteps=1000, beta_schedule='cosine', device='cpu'):
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps).to(device)
        else:
            self.betas = cosine_beta_schedule(timesteps).to(device)
        
        # Pre-compute useful values
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to data."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """Reverse diffusion: denoise one step."""
        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict noise
        predicted_noise = model(x, t)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """Generate samples by running full reverse diffusion."""
        device = self.device
        b = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)
        
        return x


# ============================================
# Positional Encoding
# ============================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ============================================
# U-Net for Depth Image Denoising
# ============================================

class Block(nn.Module):
    """Basic conv block with time embedding."""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]  # Extend to [B, C, 1, 1]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SimpleUNet(nn.Module):
    """Simple U-Net for depth image denoising."""
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=32):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Encoder (downsampling)
        self.conv0 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.downs = nn.ModuleList([
            Block(32, 64, time_emb_dim),
            Block(64, 128, time_emb_dim),
            Block(128, 256, time_emb_dim),
        ])
        
        # Decoder (upsampling)
        self.ups = nn.ModuleList([
            Block(256, 128, time_emb_dim, up=True),
            Block(128, 64, time_emb_dim, up=True),
            Block(64, 32, time_emb_dim, up=True),
        ])
        
        self.output = nn.Conv2d(32, out_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        
        # Encoder with skip connections
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
        
        # Decoder with skip connections
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
        
        return self.output(x)


# ============================================
# MLP for Scalars + Action Denoising
# ============================================

class ScalarActionMLP(nn.Module):
    """MLP for denoising scalars (7D) + action (2D) = 9D vector."""
    def __init__(self, input_dim=9, hidden_dim=256, time_emb_dim=32):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x_t = torch.cat([x, t], dim=-1)
        return self.net(x_t)


# ============================================
# Combined Diffusion Model
# ============================================

class CombinedDiffusionModel(nn.Module):
    """
    Combined model that denoises:
    - Depth image (1, 64, 64) via U-Net
    - Scalars (7,) + Action (2,) = (9,) via MLP
    """
    def __init__(self, time_emb_dim=32):
        super().__init__()
        self.image_model = SimpleUNet(in_channels=1, out_channels=1, time_emb_dim=time_emb_dim)
        self.scalar_action_model = ScalarActionMLP(input_dim=9, time_emb_dim=time_emb_dim)
    
    def forward(self, depth_image, scalar_action, timestep):
        """
        Predict noise for both modalities.
        
        Args:
            depth_image: (B, 1, 64, 64)
            scalar_action: (B, 9) - [scalars(7) + action(2)]
            timestep: (B,)
        
        Returns:
            noise_image: (B, 1, 64, 64)
            noise_scalar_action: (B, 9)
        """
        noise_image = self.image_model(depth_image, timestep)
        noise_scalar_action = self.scalar_action_model(scalar_action, timestep)
        return noise_image, noise_scalar_action


# ============================================
# Dataset
# ============================================

class DiffusionDataset(Dataset):
    """Dataset for diffusion training."""
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.depth_images = torch.FloatTensor(data['depth_images'])  # (N, 1, 64, 64)
        self.scalars = torch.FloatTensor(data['scalars'])            # (N, 7)
        self.actions = torch.FloatTensor(data['actions'])            # (N, 2)
        
        # Combine scalars and actions
        self.scalar_actions = torch.cat([self.scalars, self.actions], dim=1)  # (N, 9)
        
        # Normalize for better diffusion training
        self.image_mean = self.depth_images.mean()
        self.image_std = self.depth_images.std()
        self.scalar_action_mean = self.scalar_actions.mean(dim=0)
        self.scalar_action_std = self.scalar_actions.std(dim=0) + 1e-6
        
        print(f"Loaded {len(self)} samples for diffusion training")
        print(f"  Depth images: {self.depth_images.shape}")
        print(f"  Scalars + Actions: {self.scalar_actions.shape}")
    
    def normalize_image(self, x):
        return (x - self.image_mean) / (self.image_std + 1e-6)
    
    def denormalize_image(self, x):
        return x * (self.image_std + 1e-6) + self.image_mean
    
    def normalize_scalar_action(self, x):
        return (x - self.scalar_action_mean) / self.scalar_action_std
    
    def denormalize_scalar_action(self, x):
        return x * self.scalar_action_std + self.scalar_action_mean
    
    def __len__(self):
        return len(self.depth_images)
    
    def __getitem__(self, idx):
        return {
            'depth_image': self.normalize_image(self.depth_images[idx]),
            'scalar_action': self.normalize_scalar_action(self.scalar_actions[idx])
        }
    
    def get_normalization_params(self):
        """Return normalization parameters for generation."""
        return {
            'image_mean': self.image_mean,
            'image_std': self.image_std,
            'scalar_action_mean': self.scalar_action_mean,
            'scalar_action_std': self.scalar_action_std
        }


# ============================================
# Training
# ============================================

def train_diffusion(data_path, epochs=500, batch_size=64, lr=1e-4, 
                    timesteps=1000, device='cuda', save_path='diffusion_model.pth'):
    """Train the diffusion model."""
    
    # Load data
    dataset = DiffusionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model and diffusion utilities
    model = CombinedDiffusionModel().to(device)
    diffusion = DiffusionUtils(timesteps=timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining diffusion model on {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Timesteps: {timesteps}")
    print()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for batch in dataloader:
            depth_image = batch['depth_image'].to(device)      # (B, 1, 64, 64)
            scalar_action = batch['scalar_action'].to(device)  # (B, 9)
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (depth_image.shape[0],), device=device).long()
            
            # Add noise
            noise_image = torch.randn_like(depth_image)
            noise_scalar_action = torch.randn_like(scalar_action)
            
            noisy_image = diffusion.q_sample(depth_image, t, noise_image)
            noisy_scalar_action = diffusion.q_sample(scalar_action, t, noise_scalar_action)
            
            # Predict noise
            pred_noise_image, pred_noise_scalar_action = model(noisy_image, noisy_scalar_action, t)
            
            # Loss
            loss_image = F.mse_loss(pred_noise_image, noise_image)
            loss_scalar_action = F.mse_loss(pred_noise_scalar_action, noise_scalar_action)
            loss = loss_image + loss_scalar_action
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'normalization_params': dataset.get_normalization_params()
            }
            torch.save(checkpoint, f"{save_path.replace('.pth', '')}_epoch{epoch+1}.pth")
            print(f"  Checkpoint saved!")
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'normalization_params': dataset.get_normalization_params(),
        'timesteps': timesteps
    }
    torch.save(final_checkpoint, save_path)
    print(f"\n✅ Model saved to {save_path}")
    
    return model


# ============================================
# Generation
# ============================================

@torch.no_grad()
def generate_samples(model_path, num_samples=1000, batch_size=64, device='cuda'):
    """Generate synthetic samples using trained diffusion model."""
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = CombinedDiffusionModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    norm_params = checkpoint['normalization_params']
    timesteps = checkpoint.get('timesteps', 1000)
    
    diffusion = DiffusionUtils(timesteps=timesteps, device=device)
    
    print(f"\nGenerating {num_samples} synthetic samples...")
    
    all_depth_images = []
    all_scalars = []
    all_actions = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # Start from noise
        noisy_image = torch.randn(current_batch_size, 1, 64, 64, device=device)
        noisy_scalar_action = torch.randn(current_batch_size, 9, device=device)
        
        # Reverse diffusion
        for i in reversed(range(timesteps)):
            t = torch.full((current_batch_size,), i, device=device, dtype=torch.long)
            
            # Predict noise
            pred_noise_image, pred_noise_scalar_action = model(noisy_image, noisy_scalar_action, t)
            
            # Denoise step for image
            betas_t = get_index_from_list(diffusion.betas, t, noisy_image.shape)
            sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
                diffusion.sqrt_one_minus_alphas_cumprod, t, noisy_image.shape)
            sqrt_recip_alphas_t = get_index_from_list(diffusion.sqrt_recip_alphas, t, noisy_image.shape)
            
            model_mean_image = sqrt_recip_alphas_t * (
                noisy_image - betas_t * pred_noise_image / sqrt_one_minus_alphas_cumprod_t)
            
            # Denoise step for scalar_action
            betas_t_sa = get_index_from_list(diffusion.betas, t, noisy_scalar_action.shape)
            sqrt_one_minus_alphas_cumprod_t_sa = get_index_from_list(
                diffusion.sqrt_one_minus_alphas_cumprod, t, noisy_scalar_action.shape)
            sqrt_recip_alphas_t_sa = get_index_from_list(diffusion.sqrt_recip_alphas, t, noisy_scalar_action.shape)
            
            model_mean_scalar_action = sqrt_recip_alphas_t_sa * (
                noisy_scalar_action - betas_t_sa * pred_noise_scalar_action / sqrt_one_minus_alphas_cumprod_t_sa)
            
            if i > 0:
                posterior_variance_t = get_index_from_list(diffusion.posterior_variance, t, noisy_image.shape)
                posterior_variance_t_sa = get_index_from_list(diffusion.posterior_variance, t, noisy_scalar_action.shape)
                
                noise_image = torch.randn_like(noisy_image)
                noise_scalar_action = torch.randn_like(noisy_scalar_action)
                
                noisy_image = model_mean_image + torch.sqrt(posterior_variance_t) * noise_image
                noisy_scalar_action = model_mean_scalar_action + torch.sqrt(posterior_variance_t_sa) * noise_scalar_action
            else:
                noisy_image = model_mean_image
                noisy_scalar_action = model_mean_scalar_action
        
        # Denormalize
        image_mean = norm_params['image_mean']
        image_std = norm_params['image_std']
        sa_mean = norm_params['scalar_action_mean'].to(device)
        sa_std = norm_params['scalar_action_std'].to(device)
        
        final_image = noisy_image * (image_std + 1e-6) + image_mean
        final_scalar_action = noisy_scalar_action * sa_std + sa_mean
        
        # Split scalar_action back into scalars and actions
        final_scalars = final_scalar_action[:, :7]
        final_actions = final_scalar_action[:, 7:]
        
        # Clip to valid ranges
        final_image = torch.clamp(final_image, 0, 1)
        final_actions[:, 0] = torch.clamp(final_actions[:, 0], 0, 1)      # linear_vel
        final_actions[:, 1] = torch.clamp(final_actions[:, 1], -1, 1)    # angular_vel
        
        all_depth_images.append(final_image.cpu())
        all_scalars.append(final_scalars.cpu())
        all_actions.append(final_actions.cpu())
    
    # Concatenate all batches
    depth_images = torch.cat(all_depth_images, dim=0).numpy()
    scalars = torch.cat(all_scalars, dim=0).numpy()
    actions = torch.cat(all_actions, dim=0).numpy()
    
    print(f"\n✅ Generated {len(depth_images)} samples")
    print(f"  Depth images: {depth_images.shape}")
    print(f"  Scalars: {scalars.shape}")
    print(f"  Actions: {actions.shape}")
    
    return {
        'depth_images': depth_images,
        'scalars': scalars,
        'actions': actions
    }


def save_synthetic_data(synthetic_data, output_path):
    """Save generated synthetic data."""
    with open(output_path, 'wb') as f:
        pickle.dump(synthetic_data, f)
    print(f"✅ Synthetic data saved to {output_path}")


# ============================================
# Combine Real + Synthetic Data
# ============================================

def combine_datasets(real_path, synthetic_path, output_path, synthetic_ratio=1.0):
    """
    Combine real and synthetic data.
    
    Args:
        real_path: Path to real training data
        synthetic_path: Path to synthetic data
        output_path: Path to save combined data
        synthetic_ratio: Ratio of synthetic data to use (1.0 = all)
    """
    # Load real data
    with open(real_path, 'rb') as f:
        real_data = pickle.load(f)
    
    # Load synthetic data
    with open(synthetic_path, 'rb') as f:
        synthetic_data = pickle.load(f)
    
    # Optionally subsample synthetic data
    n_synthetic = int(len(synthetic_data['actions']) * synthetic_ratio)
    indices = np.random.choice(len(synthetic_data['actions']), n_synthetic, replace=False)
    
    # Combine
    combined = {
        'depth_images': np.concatenate([
            real_data['depth_images'],
            synthetic_data['depth_images'][indices]
        ], axis=0),
        'scalars': np.concatenate([
            real_data['scalars'],
            synthetic_data['scalars'][indices]
        ], axis=0),
        'actions': np.concatenate([
            real_data['actions'],
            synthetic_data['actions'][indices]
        ], axis=0),
        'metadata': {
            'num_real': len(real_data['actions']),
            'num_synthetic': n_synthetic,
            'total': len(real_data['actions']) + n_synthetic,
            'synthetic_ratio': synthetic_ratio
        }
    }
    
    # Shuffle
    perm = np.random.permutation(len(combined['actions']))
    combined['depth_images'] = combined['depth_images'][perm]
    combined['scalars'] = combined['scalars'][perm]
    combined['actions'] = combined['actions'][perm]
    
    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(combined, f)
    
    print(f"\n✅ Combined dataset saved to {output_path}")
    print(f"  Real samples:      {combined['metadata']['num_real']}")
    print(f"  Synthetic samples: {combined['metadata']['num_synthetic']}")
    print(f"  Total samples:     {combined['metadata']['total']}")
    
    return combined


# ============================================
# Main CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Diffusion model for data augmentation')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'generate', 'combine'],
                       help='Mode: train, generate, or combine')
    
    # Training args
    parser.add_argument('--data', type=str, default='training_data.pkl',
                       help='Path to training data (for train mode)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    
    # Generation args
    parser.add_argument('--model', type=str, default='diffusion_model.pth',
                       help='Path to trained diffusion model')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--output', type=str, default='synthetic_data.pkl',
                       help='Output path')
    
    # Combine args
    parser.add_argument('--real', type=str, default='training_data.pkl',
                       help='Path to real data')
    parser.add_argument('--synthetic', type=str, default='synthetic_data.pkl',
                       help='Path to synthetic data')
    parser.add_argument('--synthetic_ratio', type=float, default=1.0,
                       help='Ratio of synthetic data to use')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    if args.mode == 'train':
        train_diffusion(
            data_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            timesteps=args.timesteps,
            device=args.device,
            save_path=args.model
        )
    
    elif args.mode == 'generate':
        synthetic_data = generate_samples(
            model_path=args.model,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device
        )
        save_synthetic_data(synthetic_data, args.output)
    
    elif args.mode == 'combine':
        combine_datasets(
            real_path=args.real,
            synthetic_path=args.synthetic,
            output_path=args.output,
            synthetic_ratio=args.synthetic_ratio
        )


if __name__ == "__main__":
    main()