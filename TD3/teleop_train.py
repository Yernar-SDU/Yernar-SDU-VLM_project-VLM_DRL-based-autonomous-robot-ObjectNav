import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np


# ============================================
# Dataset: Loads your teleoperation data
# ============================================
class BCDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # X values (observations)
        self.depth_images = torch.FloatTensor(data['depth_images'])  # (N, 1, 64, 64)
        self.scalars = torch.FloatTensor(data['scalars'])            # (N, 7)
        
        # Y values (actions) - THIS IS WHAT WE PREDICT
        self.actions = torch.FloatTensor(data['actions'])            # (N, 2)
        
        print(f"Loaded {len(self)} samples")
        print(f"  X: depth_images {self.depth_images.shape}, scalars {self.scalars.shape}")
        print(f"  Y: actions {self.actions.shape}")
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return {
            'depth_image': self.depth_images[idx],  # (1, 64, 64)
            'scalars': self.scalars[idx],           # (7,)
            'action': self.actions[idx]             # (2,) - TARGET
        }


# ============================================
# Model: Predicts action from observation
# ============================================
class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN processes depth image
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),   # (1,64,64) -> (32,15,15)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (32,15,15) -> (64,6,6)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (64,6,6) -> (64,4,4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)                       # (64,4,4) -> (64,1,1)
        )
        
        # MLP processes scalars
        self.scalar_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined layers -> predict action
        self.action_head = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: [linear_vel, angular_vel]
        )
    
    def forward(self, depth_image, scalars):
        # Process image
        img_features = self.cnn(depth_image)           # (B, 64, 1, 1)
        img_features = img_features.view(img_features.size(0), -1)  # (B, 64)
        
        # Process scalars
        scalar_features = self.scalar_mlp(scalars)     # (B, 64)
        
        # Combine and predict action
        combined = torch.cat([img_features, scalar_features], dim=1)  # (B, 128)
        action = self.action_head(combined)            # (B, 2)
        
        return action


# ============================================
# Training Loop
# ============================================
def train_bc():
    # Load data
    dataset = BCDataset('./combined_data.pkl')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Create model
    model = BCPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(1000):
        total_loss = 0
        
        for batch in dataloader:
            # Get batch data
            depth_images = batch['depth_image']  # (B, 1, 64, 64)
            scalars = batch['scalars']           # (B, 7)
            target_actions = batch['action']     # (B, 2) - GROUND TRUTH
            
            # Forward pass: predict action
            predicted_actions = model(depth_images, scalars)  # (B, 2)
            
            # Loss: how different is prediction from human action?
            loss = F.mse_loss(predicted_actions, target_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/100, Loss: {avg_loss:.6f}")
    
    # Save trained model
    torch.save(model.state_dict(), 'bc_policy.pth')
    print("Model saved!")


if __name__ == "__main__":
    train_bc()