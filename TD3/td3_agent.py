# td3_agent.py

import torch
import torch.nn.functional as F
import copy
from models import ActorLSTM, CriticLSTM

class TD3LSTMAgent:
    def __init__(self, obs_shape, scalar_dim, action_dim, device, seq_len=10,
                 actor_lr=1e-4, critic_lr=1e-4, tau=0.005, gamma=0.99,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        
        self.device = device
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.tau = tau
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

        # --- Actor and target ---
        self.actor = ActorLSTM(img_channels=obs_shape[0], scalar_dim=scalar_dim, action_dim=action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # --- Critic 1 and 2 and targets ---
        self.critic1 = CriticLSTM(img_channels=obs_shape[0], scalar_dim=scalar_dim, action_dim=action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1).to(device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)

        self.critic2 = CriticLSTM(img_channels=obs_shape[0], scalar_dim=scalar_dim, action_dim=action_dim).to(device)
        self.critic2_target = copy.deepcopy(self.critic2).to(device)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

    def select_action(self, image_seq, scalar_seq, hidden_state=None):
        self.actor.eval()
        with torch.no_grad():
            action, hidden_state = self.actor(image_seq, scalar_seq, hidden_state)
        self.actor.train()
        return action.cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        self.total_it += 1

        # Sample from buffer (your buffer code is correct)
        image_seq, scalar_seq, action_seq, reward_seq, next_image_seq, next_scalar_seq, done_seq = replay_buffer.sample(batch_size)
        
        # Unsqueeze rewards and dones for broadcasting
        reward_seq = reward_seq.unsqueeze(-1)
        done_seq = done_seq.unsqueeze(-1)

        with torch.no_grad():
            # Select next action according to policy and add clipped noise
            next_action, _ = self.actor_target(next_image_seq, next_scalar_seq)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # Clip the noisy action to the valid range
            # Note: Assuming your action space is [0, 1] for linear and [-1, 1] for angular
            next_action_clipped = next_action + noise
            next_action_clipped[:, 0] = next_action_clipped[:, 0].clamp(0.0, 1.0)
            next_action_clipped[:, 1] = next_action_clipped[:, 1].clamp(-1.0, 1.0)

            # To pass a single action to the recurrent critic, repeat it across the sequence length
            next_action_seq = next_action_clipped.unsqueeze(1).repeat(1, self.seq_len, 1)

            # === CORRECTED: Compute the target Q value using the FULL sequence ===
            target_Q1, _ = self.critic1_target(next_image_seq, next_scalar_seq, next_action_seq)
            target_Q2, _ = self.critic2_target(next_image_seq, next_scalar_seq, next_action_seq)
            target_Q = torch.min(target_Q1, target_Q2)

            # Use the reward from the LAST step of the sequence
            target_Q = reward_seq[:, -1] + (1 - done_seq[:, -1]) * self.gamma * target_Q
            
        # === CORRECTED: Get current Q estimates using the FULL sequence ===
        current_Q1, _ = self.critic1(image_seq, scalar_seq, action_seq)
        current_Q2, _ = self.critic2(image_seq, scalar_seq, action_seq)

        # === CORRECTED: Critic loss calculation ===
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        # Optimize the critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss = None
        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Generate actor's action for the original state sequence
            actor_action, _ = self.actor(image_seq, scalar_seq)
            
            # Repeat the single action across the sequence for the critic
            actor_action_seq = actor_action.unsqueeze(1).repeat(1, self.seq_len, 1)
            
            # Compute actor loss
            actor_loss = -self.critic1(image_seq, scalar_seq, actor_action_seq)[0].mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
            self.soft_update(self.actor, self.actor_target)


        print(f"Current_Q1: {current_Q1.shape}, Target_Q: {target_Q.shape}")
        print(f"Current_Q2: {current_Q2.shape}")
        print(f"Critic1 Loss: {critic1_loss.item():.4f}, Critic2 Loss: {critic2_loss.item():.4f}")
        print(f"Actor Loss: {actor_loss.item() if actor_loss is not None else 'N/A'}")
        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else None
        }

    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
