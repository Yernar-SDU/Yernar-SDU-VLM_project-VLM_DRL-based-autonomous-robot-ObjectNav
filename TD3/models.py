# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# === CNN for image processing ===
class ImageEncoder(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: (B, 64, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # (B, 64)

# === MLP for scalar state ===
class StateProcessor(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)  # (B, 64)

# === Actor Network ===
class ActorLSTM(nn.Module):
    def __init__(self, img_channels=1, scalar_dim=7, action_dim=2, hidden_size=512):
        super().__init__()
        self.image_encoder = ImageEncoder(img_channels)
        self.scalar_encoder = StateProcessor(scalar_dim)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_dim)

    def forward(self, image_seq, scalar_seq, hidden_state=None):
        # image_seq: (B, T, C, H, W)
        # scalar_seq: (B, T, D)

        B, T, C, H, W = image_seq.shape
        image_seq = image_seq.view(B * T, C, H, W)
        scalar_seq = scalar_seq.view(B * T, -1)

        img_feat = self.image_encoder(image_seq)
        scal_feat = self.scalar_encoder(scalar_seq)

        fused = torch.cat([img_feat, scal_feat], dim=1)  # (B*T, 128)
        fused = fused.view(B, T, -1)  # (B, T, 128)

        self.lstm.flatten_parameters()
        lstm_out, hidden_state = self.lstm(fused, hidden_state)  # (B, T, H)
        last_output = lstm_out[:, -1, :]  # (B, H)

        raw_action = self.fc(last_output)  # (B, action_dim)

        # Normalize: linear ∈ [0,1], angular ∈ [-1,1]
        linear = torch.sigmoid(raw_action[:, 0:1])
        angular = torch.tanh(raw_action[:, 1:2])
        action = torch.cat([linear, angular], dim=1)  # (B, 2)

        return action, hidden_state

# === Critic Network ===
class CriticLSTM(nn.Module):
    def __init__(self, img_channels=1, scalar_dim=7, action_dim=2, hidden_size=256):
        super().__init__()
        self.image_encoder = ImageEncoder(img_channels)
        self.scalar_encoder = StateProcessor(scalar_dim)
        self.action_encoder = nn.Linear(action_dim, 64)

        self.lstm = nn.LSTM(input_size=128 + 64, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, image_seq, scalar_seq, action_seq, hidden_state=None):
        # image_seq: (B, T, C, H, W)
        # scalar_seq: (B, T, D)
        # action_seq: (B, T, action_dim)

        B, T, C, H, W = image_seq.shape
        image_seq = image_seq.view(B * T, C, H, W)
        scalar_seq = scalar_seq.view(B * T, -1)
        action_seq = action_seq.view(B * T, -1)

        img_feat = self.image_encoder(image_seq)
        scal_feat = self.scalar_encoder(scalar_seq)
        act_feat = self.action_encoder(action_seq)

        state_feat = torch.cat([img_feat, scal_feat], dim=1)  # (B*T, 128)
        state_feat = state_feat.view(B, T, -1)
        act_feat = act_feat.view(B, T, -1)

        fused = torch.cat([state_feat, act_feat], dim=2)  # (B, T, 192)

        self.lstm.flatten_parameters()
        lstm_out, hidden_state = self.lstm(fused, hidden_state)
        last_output = lstm_out[:, -1, :]  # (B, H)

        q_value = self.fc(last_output)  # (B, 1)
        return q_value, hidden_state