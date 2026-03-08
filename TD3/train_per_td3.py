#!/usr/bin/env python3
import os
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from per_replay_buffer import PrioritizedReplayBuffer  # Our updated PER
from realsense_env import GazeboEnv                   # Now returns (1,64,64) + (7,)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Weight initialization helper using Xavier uniform
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


################################
#   CNN + FC Actor
################################
class ActorCNN(nn.Module):
    """
    Actor that receives:
      - single-channel image (batch,1,64,64)
      - 7D scalars (batch,7)
    Then merges the features, outputs 2D action in [-1,1].
    """
    def __init__(self, action_dim=2):
        super(ActorCNN, self).__init__()
        # CNN for the 1×64×64 input
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # out: 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # out: 64 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# out:128 x 8 x 8
            nn.ReLU(),
        )
        # Determine conv output size
        dummy = torch.zeros(1, 1, 64, 64)
        with torch.no_grad():
            conv_out = self.conv(dummy)
            conv_size = conv_out.view(1, -1).size(1)  # e.g. 128*8*8 = 8192

        # MLP for scalars (7D) with LayerNorm (instead of BatchNorm)
        self.scalar_fc = nn.Sequential(
            nn.Linear(7, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # Final fully connected layers after concatenating conv + scalar features
        self.fc_final = nn.Sequential(
            nn.Linear(conv_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # outputs in [-1,1]
        )

        # Apply weight initialization
        self.apply(init_weights)

    def forward(self, img, scalars):
        """
        img: (batch,1,64,64)
        scalars: (batch,7)
        returns: (batch,2) => action in [-1,1]
        """
        c = self.conv(img)
        c = c.view(c.size(0), -1)
        s = self.scalar_fc(scalars)
        x = torch.cat([c, s], dim=1)
        out = self.fc_final(x)
        return out


################################
#   CNN + FC Critic
################################
class CriticCNN(nn.Module):
    """
    Critic that receives (image, scalars, action) and outputs Q-value.
    Implements two separate Q-networks for TD3.
    """
    def __init__(self, action_dim=2):
        super(CriticCNN, self).__init__()
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
            o = self.conv1(dummy)
            conv_size1 = o.view(1, -1).size(1)
        self.scl_fc1 = nn.Sequential(
            nn.Linear(7, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.q1_fc = nn.Sequential(
            nn.Linear(conv_size1 + 64 + action_dim, 256),
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
            o2 = self.conv2(dummy)
            conv_size2 = o2.view(1, -1).size(1)
        self.scl_fc2 = nn.Sequential(
            nn.Linear(7, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.q2_fc = nn.Sequential(
            nn.Linear(conv_size2 + 64 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Apply weight initialization
        self.apply(init_weights)

    def forward(self, img, scl, action):
        """
        img: (batch,1,64,64)
        scl: (batch,7)
        action: (batch,2)
        returns: Q1, Q2 (each of shape (batch,1))
        """
        # Q1 computation
        c1 = self.conv1(img).view(img.size(0), -1)
        s1 = self.scl_fc1(scl)
        x1 = torch.cat([c1, s1, action], dim=1)
        q1 = self.q1_fc(x1)

        # Q2 computation
        c2 = self.conv2(img).view(img.size(0), -1)
        s2 = self.scl_fc2(scl)
        x2 = torch.cat([c2, s2, action], dim=1)
        q2 = self.q2_fc(x2)

        return q1, q2


################################
#   TD3 with CNN
################################
class TD3(object):
    def __init__(self, action_dim=2, max_action=1.0):
        self.actor = ActorCNN(action_dim).to(device)
        self.actor_target = ActorCNN(action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = CriticCNN(action_dim).to(device)
        self.critic_target = CriticCNN(action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

        # Gradient clipping value
        self.grad_clip = 0.5

        # Learning rate schedulers (step every 10000 training steps)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=10000, gamma=0.99)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=10000, gamma=0.99)

        # Enable mixed precision training if using CUDA
        self.use_amp = (device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()

    def get_action(self, img_np, scl_np):
        """
        img_np: (1,64,64) np.float32
        scl_np: (7,) np.float32
        Returns: action (2,) in [-1,1]
        """
        img_torch = torch.from_numpy(img_np).unsqueeze(0).to(device)   # shape: (1,1,64,64)
        scl_torch = torch.from_numpy(scl_np).unsqueeze(0).to(device)    # shape: (1,7)
        # During evaluation, set actor to eval mode to avoid issues with LayerNorm if needed.
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(img_torch, scl_torch)
        self.actor.train()
        return action.cpu().numpy().flatten()

    def train(self,
              replay_buffer,
              iterations,
              batch_size=256,
              discount=0.99,
              tau=0.005,
              policy_noise=0.2,
              noise_clip=0.5,
              policy_freq=2,
              beta=0.4):
        av_Q = 0
        max_Q = -inf
        av_critic_loss = 0
        av_actor_loss = 0

        for it in range(iterations):
            sample = replay_buffer.sample_batch(batch_size, beta)
            if sample is None:
                break
            (batch_img, batch_scl, batch_actions, batch_rewards,
             batch_dones, batch_next_img, batch_next_scl, weights, indices) = sample

            # Convert to torch tensors
            batch_img      = torch.from_numpy(batch_img).to(device)
            batch_scl      = torch.from_numpy(batch_scl).to(device)
            batch_actions  = torch.from_numpy(batch_actions).to(device)
            batch_rewards  = torch.from_numpy(batch_rewards).unsqueeze(-1).to(device)
            batch_dones    = torch.from_numpy(batch_dones).unsqueeze(-1).to(device)
            batch_next_img = torch.from_numpy(batch_next_img).to(device)
            batch_next_scl = torch.from_numpy(batch_next_scl).to(device)
            ws = torch.from_numpy(weights).unsqueeze(-1).to(device)

            # --------- Critic update ---------
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    next_action = self.actor_target(batch_next_img, batch_next_scl)
                    noise = (torch.randn_like(next_action) * policy_noise).clamp(-noise_clip, noise_clip)
                    next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                    target_Q1, target_Q2 = self.critic_target(batch_next_img, batch_next_scl, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = batch_rewards + (1.0 - batch_dones) * discount * target_Q
                    current_Q1, current_Q2 = self.critic(batch_img, batch_scl, batch_actions)
                    loss_Q1 = F.smooth_l1_loss(current_Q1, target_Q, reduction='none')
                    loss_Q2 = F.smooth_l1_loss(current_Q2, target_Q, reduction='none')
                    c_loss = (loss_Q1 + loss_Q2) * ws
                    c_loss = c_loss.mean()
                self.critic_optimizer.zero_grad()
                self.scaler.scale(c_loss).backward()
                self.scaler.unscale_(self.critic_optimizer)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.scaler.step(self.critic_optimizer)
            else:
                next_action = self.actor_target(batch_next_img, batch_next_scl)
                noise = (torch.randn_like(next_action) * policy_noise).clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                target_Q1, target_Q2 = self.critic_target(batch_next_img, batch_next_scl, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = batch_rewards + (1.0 - batch_dones) * discount * target_Q
                current_Q1, current_Q2 = self.critic(batch_img, batch_scl, batch_actions)
                loss_Q1 = F.smooth_l1_loss(current_Q1, target_Q, reduction='none')
                loss_Q2 = F.smooth_l1_loss(current_Q2, target_Q, reduction='none')
                c_loss = (loss_Q1 + loss_Q2) * ws
                c_loss = c_loss.mean()
                self.critic_optimizer.zero_grad()
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.critic_optimizer.step()

            # Update PER priorities
            with torch.no_grad():
                td_error1 = (current_Q1 - target_Q).abs()
                td_error2 = (current_Q2 - target_Q).abs()
                td_errors = 0.5 * (td_error1 + td_error2)
            new_priorities = td_errors.detach().cpu().numpy().flatten() + 1e-6
            replay_buffer.update_priorities(indices, new_priorities)

            av_critic_loss += c_loss.item()
            av_Q += target_Q.mean().item()
            max_Q = max(max_Q, target_Q.max().item())

            # --------- Delayed Actor update ---------
            if it % policy_freq == 0:
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        actor_actions = self.actor(batch_img, batch_scl)
                        actor_Q1, _ = self.critic(batch_img, batch_scl, actor_actions)
                        actor_loss = -actor_Q1.mean()
                    self.actor_optimizer.zero_grad()
                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.update()
                else:
                    actor_actions = self.actor(batch_img, batch_scl)
                    actor_Q1, _ = self.critic(batch_img, batch_scl, actor_actions)
                    actor_loss = -actor_Q1.mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                    self.actor_optimizer.step()
                av_actor_loss += actor_loss.item()

                # Soft-update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.iter_count += iterations
        if iterations > 0:
            self.writer.add_scalar("Critic_Loss", av_critic_loss / iterations, self.iter_count)
            if av_actor_loss != 0:
                self.writer.add_scalar("Actor_Loss", av_actor_loss / (iterations // policy_freq + 1), self.iter_count)
            self.writer.add_scalar("Avg_Q", av_Q / iterations, self.iter_count)
            self.writer.add_scalar("Max_Q", max_Q, self.iter_count)

        # Step the learning rate schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth", map_location=device))


###########################################
#             Evaluation Function
###########################################
def evaluate(network, env, epoch, eval_episodes=5):
    avg_reward = 0.0
    collisions = 0
    for _ in range(eval_episodes):
        state = env.reset()  # (img, scalars)
        done = False
        ep_rew = 0
        steps = 0
        while not done and steps < 500:
            img, scl = state
            action = network.get_action(img, scl)
            # Scale action if needed (e.g. linear in [0,1])
            a_in = [(action[0] + 1)/2, action[1]]
            next_state, reward, done, _ = env.step(a_in)
            ep_rew += reward
            state = next_state
            steps += 1
            if reward <= -1500.0:  # collision or stuck
                collisions += 1
        avg_reward += ep_rew
    avg_reward /= eval_episodes
    avg_coll = collisions / eval_episodes
    print("----------------------------------------")
    print(f"Eval/Epoch={epoch}, AvgReward={avg_reward:.1f}, Collisions={avg_coll:.1f}")
    print("----------------------------------------")
    return avg_reward


###########################################
#             Main Training Loop
###########################################
if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = GazeboEnv()
    time.sleep(4)  # wait for environment to stabilize

    # TD3 configuration
    max_action = 1.0
    policy = TD3(action_dim=2, max_action=max_action)
    replay_buffer = PrioritizedReplayBuffer(capacity=200000, alpha=0.6)

    # Training hyperparameters
    eval_freq = 5000         # evaluate every eval_freq timesteps
    eval_episodes = 5
    max_timesteps = int(2e5)
    save_model = True
    file_name = "TD3_PER_CNN_improved"
    batch_size = 128
    discount = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2

    # Exploration noise schedule
    expl_noise = 1.5
    expl_decay_steps = 100000
    expl_min = 0.1

    # Replay buffer warm-up threshold and training frequency
    start_timesteps = 1000  # no training until this many steps
    train_freq = 50         # perform a training update every 'train_freq' timesteps

    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Initialize variables for random action near obstacle logic
    ENABLE_RANDOM_NEAR_OBSTACLE = True
    count_rand_actions = 0
    random_action = np.zeros(2)

    # Main Loop
    timestep = 0
    timesteps_since_eval = 0
    evaluations = []
    epoch = 0

    state = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    while timestep < max_timesteps:
        # Decrease exploration noise gradually
        if expl_noise > expl_min:
            expl_noise -= (1.0 - expl_min) / expl_decay_steps
            expl_noise = max(expl_noise, expl_min)

        img, scl = state
        action = policy.get_action(img, scl)
        # Add exploration noise and clip action
        action = (action + np.random.normal(0, expl_noise, size=2)).clip(-1, 1)

        # If the robot is facing an obstacle, randomly force it to take a consistent random action.
        # This is done to increase exploration in situations near obstacles.
        # Training can also be performed without it.
        if ENABLE_RANDOM_NEAR_OBSTACLE:
            if np.random.uniform(0, 1) > 0.85 and scl[-1] < 0.6 and count_rand_actions < 1:
                count_rand_actions = np.random.randint(8, 15)
                random_action = np.random.uniform(-1, 1, 2)
            if count_rand_actions > 0:
                count_rand_actions -= 1
                action = random_action.copy()
                action[0] = -1  # force linear action (e.g., reverse)
                print("Random action near obstacle:", action)

        # Scale action for environment if needed
        a_in = [(action[0] + 1)/2, action[1]]  # e.g. linear mapped to [0,1]

        next_state, reward, done, _ = env.step(a_in)
        episode_reward += reward

        # Add experience to replay buffer
        (n_img, n_scl) = next_state
        replay_buffer.add(img, scl, action, reward, float(done), n_img, n_scl)

        state = next_state
        episode_timesteps += 1
        timestep += 1
        timesteps_since_eval += 1

        # Perform training step if enough timesteps have been collected
        if timestep > start_timesteps and (timestep % train_freq == 0):
            policy.train(replay_buffer, iterations=1, batch_size=batch_size,
                         discount=discount, tau=tau, policy_noise=policy_noise,
                         noise_clip=noise_clip, policy_freq=policy_freq, beta=0.4)

        # Evaluation and episode reset
        if done:
            print(f"Episode {episode_num}, Reward={episode_reward:.1f}, Steps={episode_timesteps}, Timestep={timestep}")

            if timesteps_since_eval >= eval_freq:
                timesteps_since_eval %= eval_freq
                val_rew = evaluate(policy, env, epoch, eval_episodes)
                evaluations.append(val_rew)
                if save_model:
                    policy.save(file_name, "./pytorch_models")
                np.save(f"./results/{file_name}", evaluations)
                epoch += 1

            state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    # Final evaluation
    val_rew = evaluate(policy, env, epoch, eval_episodes)
    evaluations.append(val_rew)
    policy.save(file_name, "./pytorch_models")
    np.save(f"./results/{file_name}", evaluations)
