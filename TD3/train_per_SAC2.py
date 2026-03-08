#!/usr/bin/env python3
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from numpy import inf

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
#          Actor Network (SAC)
########################################
class ActorSAC(nn.Module):
    """
    Actor network for SAC that processes:
      - a single-channel image (batch,1,64,64)
      - 7D scalars (batch,7)
    Outputs:
      - action: tanh-squashed sample from a Gaussian
      - log_prob: log probability of the sampled action
      - mean: deterministic action mean (useful for evaluation)
    """
    def __init__(self, action_dim=2, log_std_min=-20, log_std_max=2):
        super(ActorSAC, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # CNN for image input
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (32,32,32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (64,16,16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# (128,8,8)
            nn.ReLU(),
        )
        # Determine conv output size dynamically
        dummy = torch.zeros(1, 1, 64, 64)
        with torch.no_grad():
            conv_out = self.conv(dummy)
            self.conv_out_size = conv_out.view(1, -1).size(1)

        # MLP for scalar input (7D)
        self.scalar_fc = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
        )

        # Final fully-connected layers to produce mean and log_std
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size + 64, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, img, scalars):
        c = self.conv(img)
        c = c.view(c.size(0), -1)
        s = self.scalar_fc(scalars)
        x = torch.cat([c, s], dim=1)
        x = self.fc(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        # Clamp log_std within [log_std_min, log_std_max]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, img, scalars):
        mean, log_std = self.forward(img, scalars)
        std = log_std.exp()
        normal = Normal(mean, std)
        # Reparameterization trick
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t

        # Compute log_prob, including tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - y_t.pow(2) + 1e-6), dim=-1, keepdim=True)
        return action, log_prob, mean

########################################
#          Critic Network (SAC)
########################################
class CriticSAC(nn.Module):
    """
    Critic network with two Q-functions.
    Each processes:
      - image (batch,1,64,64)
      - scalars (batch,7)
      - action (batch, action_dim)
    """
    def __init__(self, action_dim=2):
        super(CriticSAC, self).__init__()
        # Q1 network
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        dummy = torch.zeros(1, 1, 64, 64)
        with torch.no_grad():
            out1 = self.conv1(dummy)
            self.conv1_out_size = out1.view(1, -1).size(1)
        self.scl_fc1 = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
        )
        self.q1_fc = nn.Sequential(
            nn.Linear(self.conv1_out_size + 64 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2 network
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        with torch.no_grad():
            out2 = self.conv2(dummy)
            self.conv2_out_size = out2.view(1, -1).size(1)
        self.scl_fc2 = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
        )
        self.q2_fc = nn.Sequential(
            nn.Linear(self.conv2_out_size + 64 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, img, scalars, action):
        # Q1 forward
        c1 = self.conv1(img).view(img.size(0), -1)
        s1 = self.scl_fc1(scalars)
        x1 = torch.cat([c1, s1, action], dim=1)
        q1 = self.q1_fc(x1)

        # Q2 forward
        c2 = self.conv2(img).view(img.size(0), -1)
        s2 = self.scl_fc2(scalars)
        x2 = torch.cat([c2, s2, action], dim=1)
        q2 = self.q2_fc(x2)
        return q1, q2

########################################
#             SAC Agent
########################################
class SAC(object):
    def __init__(self, action_dim=2, max_action=1.0, discount=0.99, tau=0.005,
                 actor_lr=1e-4, critic_lr=1e-4, alpha_lr=1e-4, target_entropy=None):
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau

        # Actor and optimizer
        self.actor = ActorSAC(action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic and target critic
        self.critic = CriticSAC(action_dim).to(device)
        self.critic_target = CriticSAC(action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Temperature parameter (log_alpha) for entropy regularization
        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # TensorBoard writer (logs saved in the "runs" folder)
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, img_np, scl_np, evaluate=False):
        img = torch.from_numpy(img_np).unsqueeze(0).to(device)   # (1,1,64,64)
        scl = torch.from_numpy(scl_np).unsqueeze(0).to(device)    # (1,7)
        with torch.no_grad():
            action, _, mean = self.actor.sample(img, scl)
            if evaluate:
                action = torch.tanh(mean)
        action = action.cpu().numpy().flatten()
        return action.clip(-self.max_action, self.max_action)

    def train(self, replay_buffer, iterations, batch_size=128, beta=0.4):
        total_critic_loss = 0
        total_actor_loss = 0
        total_alpha_loss = 0

        for it in range(iterations):
            sample = replay_buffer.sample_batch(batch_size, beta)
            if sample is None:
                break
            (batch_img, batch_scl, batch_actions, batch_rewards, batch_dones,
             batch_next_img, batch_next_scl, weights, indices) = sample

            # Convert to torch tensors
            batch_img      = torch.from_numpy(batch_img).to(device)
            batch_scl      = torch.from_numpy(batch_scl).to(device)
            batch_actions  = torch.from_numpy(batch_actions).to(device)
            batch_rewards  = torch.from_numpy(batch_rewards).unsqueeze(-1).to(device)
            batch_dones    = torch.from_numpy(batch_dones).unsqueeze(-1).to(device)
            batch_next_img = torch.from_numpy(batch_next_img).to(device)
            batch_next_scl = torch.from_numpy(batch_next_scl).to(device)
            weights        = torch.from_numpy(weights).unsqueeze(-1).to(device)

            # ---------------------- Update Critic ---------------------- #
            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(batch_next_img, batch_next_scl)
                target_Q1, target_Q2 = self.critic_target(batch_next_img, batch_next_scl, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = batch_rewards + (1 - batch_dones) * self.discount * (target_Q - self.log_alpha.exp() * next_log_prob)

            current_Q1, current_Q2 = self.critic(batch_img, batch_scl, batch_actions)
            critic_loss1 = F.mse_loss(current_Q1, target_Q, reduction='none')
            critic_loss2 = F.mse_loss(current_Q2, target_Q, reduction='none')
            critic_loss = (critic_loss1 + critic_loss2) * weights
            critic_loss = critic_loss.mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update PER priorities based on TD error
            with torch.no_grad():
                td_error = 0.5 * (torch.abs(current_Q1 - target_Q) + torch.abs(current_Q2 - target_Q))
            new_priorities = td_error.detach().cpu().numpy().flatten()
            replay_buffer.update_priorities(indices, new_priorities)
            total_critic_loss += critic_loss.item()

            # ---------------------- Update Actor ---------------------- #
            action_new, log_prob_new, _ = self.actor.sample(batch_img, batch_scl)
            q1_new, q2_new = self.critic(batch_img, batch_scl, action_new)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.log_alpha.exp() * log_prob_new - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            total_actor_loss += actor_loss.item()

            # ---------------------- Update Temperature ---------------------- #
            alpha_loss = -(self.log_alpha * (log_prob_new + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            total_alpha_loss += alpha_loss.item()

            # ---------------------- Soft Update ---------------------- #
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.iter_count += 1
        # Logging to TensorBoard
        self.writer.add_scalar("Loss/critic", total_critic_loss / iterations, self.iter_count)
        self.writer.add_scalar("Loss/actor", total_actor_loss / iterations, self.iter_count)
        self.writer.add_scalar("Loss/alpha", total_alpha_loss / iterations, self.iter_count)
        self.writer.add_scalar("Alpha", self.log_alpha.exp().item(), self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth", map_location=device))

########################################
#         Evaluation Function
########################################
def evaluate(network, env, epoch, eval_episodes=5):
    avg_reward = 0.0
    collisions = 0
    for _ in range(eval_episodes):
        state = env.reset()  # Expected to return (img, scalars)
        done = False
        ep_rew = 0
        steps = 0
        while not done and steps < 500:
            img, scl = state
            action = network.get_action(img, scl, evaluate=True)
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, _ = env.step(a_in)
            ep_rew += reward
            state = next_state
            steps += 1
            if reward <= -1500.0:
                collisions += 1
        avg_reward += ep_rew
    avg_reward /= eval_episodes
    avg_coll = collisions / eval_episodes
    print("----------------------------------------")
    print(f"Eval Epoch={epoch}, AvgReward={avg_reward:.1f}, Collisions={avg_coll:.1f}")
    print("----------------------------------------")
    return avg_reward

########################################
#       Training Script with Logs
########################################
if __name__ == "__main__":
    # Set random seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Import environment and PER buffer (assumed available)
    from realsense_env import GazeboEnv
    from per_replay_buffer import PrioritizedReplayBuffer

    env = GazeboEnv()
    time.sleep(4)  # wait for environment initialization

    # SAC agent configuration
    max_action = 1.0
    agent = SAC(action_dim=2, max_action=max_action)
    replay_buffer = PrioritizedReplayBuffer(capacity=300000, alpha=0.6)

    # Training hyperparameters
    eval_freq = 5000
    eval_episodes = 5
    max_timesteps = int(2e5)
    file_name = "SAC_PER_CNN_singlechannel"
    batch_size = 64

    # Create directories for saving models and results
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Exploration noise parameters (for initial exploration)
    expl_noise = 1.0
    expl_decay_steps = 100000
    expl_min = 0.1

    timestep = 0
    timesteps_since_eval = 0
    evaluations = []
    episode_rewards = []  # For plotting per-episode rewards
    epoch = 0

    state = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Main Training Loop
    while timestep < max_timesteps:
        if done:
            # Train after each episode
            if timestep != 0:
                agent.train(replay_buffer, episode_timesteps, batch_size)
                print(f"Episode {episode_num} finished | Reward: {episode_reward:.1f} | Steps: {episode_timesteps} | Total Timesteps: {timestep}")

                # Log episode reward to TensorBoard
                agent.writer.add_scalar("Episode/Reward", episode_reward, episode_num)
                episode_rewards.append(episode_reward)

            # Periodic Evaluation
            if timesteps_since_eval >= eval_freq:
                timesteps_since_eval %= eval_freq
                val_rew = evaluate(agent, env, epoch, eval_episodes)
                evaluations.append(val_rew)
                agent.save(file_name, "./pytorch_models")
                np.save(f"./results/{file_name}", evaluations)
                epoch += 1

            # Reset environment for next episode
            state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Decay exploration noise
        if expl_noise > expl_min:
            expl_noise -= (1.0 - expl_min) / expl_decay_steps
            expl_noise = max(expl_noise, expl_min)

        img, scl = state
        action = agent.get_action(img, scl)
        # Add exploration noise
        action = (action + np.random.normal(0, expl_noise, size=2)).clip(-1, 1)
        a_in = [(action[0] + 1) / 2, action[1]]
        next_state, reward, done, _ = env.step(a_in)
        episode_reward += reward

        # Store transition in PER buffer
        done_bool = float(done)
        (n_img, n_scl) = next_state
        replay_buffer.add(img, scl, action, reward, done_bool, n_img, n_scl)

        state = next_state
        episode_timesteps += 1
        timestep += 1
        timesteps_since_eval += 1

    # Final Evaluation after training
    val_rew = evaluate(agent, env, epoch, eval_episodes)
    evaluations.append(val_rew)
    agent.save(file_name, "./pytorch_models")
    np.save(f"./results/{file_name}", evaluations)

    # Plot episode rewards using matplotlib
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results/episode_rewards.png")
    plt.show()

    print("Training complete! To view TensorBoard logs, run:\n  tensorboard --logdir=runs")
