#!/usr/bin/env python3
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from per_replay_buffer import PrioritizedReplayBuffer  # Our updated PER
from realsense_env import GazeboEnv                   # Now returns (1,64,128) + (7,)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################
#   CNN + FC Actor
################################
class ActorCNN(nn.Module):
    """
    Actor that receives:
      - single-channel image (batch,1,64,128)
      - 7D scalars (batch,7)
    Then merges the features, outputs 2D action in [-1,1].
    """
    def __init__(self, action_dim=2):
        super(ActorCNN, self).__init__()
        # CNN for the 1×64×128 input
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # out: 32 x 32 x 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # out: 64 x 16 x 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# out: 128 x 8 x 16
            nn.ReLU(),
        )
        # figure out conv output size
        dummy = torch.zeros(1, 1, 64, 128)
        with torch.no_grad():
            conv_out = self.conv(dummy)
            conv_size = conv_out.view(1, -1).size(1)  # e.g. 128*8*16 = 16384

        # small MLP for scalars (7D)
        self.scalar_fc = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
        )

        # final fc after concatenating conv features + scalar features
        self.fc_final = nn.Sequential(
            nn.Linear(conv_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # final in [-1,1]
        )

    def forward(self, img, scalars):
        """
        img: (batch,1,64,128)
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
    Critic that receives (image, scalars, action) => outputs Q-value.
    We build 2 Q-networks internally (for TD3).
    """
    def __init__(self, action_dim=2):
        super(CriticCNN, self).__init__()
        # Q1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        dummy = torch.zeros(1, 1, 64, 128)
        with torch.no_grad():
            o = self.conv1(dummy)
            conv_size1 = o.view(1, -1).size(1)

        self.scl_fc1 = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
        )
        # Merge with action: final dimension = conv_size1 + 64 + action_dim
        self.q1_fc = nn.Sequential(
            nn.Linear(conv_size1 + 64 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2
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
            nn.ReLU(),
        )
        self.q2_fc = nn.Sequential(
            nn.Linear(conv_size2 + 64 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, img, scl, action):
        """
        img: (batch,1,64,128)
        scl: (batch,7)
        action: (batch,2)
        returns: Q1, Q2 (each shape: (batch,1))
        """
        # Q1
        c1 = self.conv1(img).view(img.size(0), -1)
        s1 = self.scl_fc1(scl)
        x1 = torch.cat([c1, s1, action], dim=1)
        q1 = self.q1_fc(x1)

        # Q2
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

    def get_action(self, img_np, scl_np):
        """
        img_np: (1,64,128) in np.float32
        scl_np: (7,) in np.float32
        Feed to actor.
        """
        img_torch = torch.from_numpy(img_np).unsqueeze(0).to(device)  # (1,1,64,128)
        scl_torch = torch.from_numpy(scl_np).unsqueeze(0).to(device)  # (1,7)
        with torch.no_grad():
            action = self.actor(img_torch, scl_torch)
        return action.cpu().numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=64,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        beta=0.4,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0

        for it in range(iterations):
            sample = replay_buffer.sample_batch(batch_size, beta)
            if sample is None:
                break
            (
                batch_img,
                batch_scl,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_img,
                batch_next_scl,
                weights,
                indices
            ) = sample

            # Convert to torch
            batch_img      = torch.from_numpy(batch_img).to(device)        # (batch,1,64,128)
            batch_scl      = torch.from_numpy(batch_scl).to(device)        # (batch,7)
            batch_actions  = torch.from_numpy(batch_actions).to(device)    # (batch,2)
            batch_rewards  = torch.from_numpy(batch_rewards).unsqueeze(-1).to(device)
            batch_dones    = torch.from_numpy(batch_dones).unsqueeze(-1).to(device)
            batch_next_img = torch.from_numpy(batch_next_img).to(device)
            batch_next_scl = torch.from_numpy(batch_next_scl).to(device)
            ws = torch.from_numpy(weights).unsqueeze(-1).to(device)

            # Actor target + noise
            with torch.no_grad():
                next_action = self.actor_target(batch_next_img, batch_next_scl)
                noise = (torch.randn_like(next_action) * policy_noise).clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                # Q target
                target_Q1, target_Q2 = self.critic_target(batch_next_img, batch_next_scl, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = batch_rewards + (1.0 - batch_dones) * discount * target_Q

            # Current Q
            current_Q1, current_Q2 = self.critic(batch_img, batch_scl, batch_actions)

            # TD error
            td_error1 = (current_Q1 - target_Q).abs()
            td_error2 = (current_Q2 - target_Q).abs()
            td_errors = 0.5 * (td_error1 + td_error2)

            # Critic loss
            c_loss = F.mse_loss(current_Q1, target_Q, reduction='none') + \
                     F.mse_loss(current_Q2, target_Q, reduction='none')
            c_loss = c_loss * ws
            c_loss = c_loss.mean()

            self.critic_optimizer.zero_grad()
            c_loss.backward()
            self.critic_optimizer.step()

            # update priorities in PER
            new_priorities = td_errors.detach().cpu().numpy().flatten()
            replay_buffer.update_priorities(indices, new_priorities)

            # Delayed policy updates
            if it % policy_freq == 0:
                actor_actions = self.actor(batch_img, batch_scl)
                actor_Q1, _ = self.critic(batch_img, batch_scl, actor_actions)
                actor_loss = -actor_Q1.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            av_loss += c_loss.item()
            av_Q += target_Q.mean().item()
            max_Q = max(max_Q, target_Q.max().item())

        self.iter_count += 1
        if iterations > 0:
            self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
            self.writer.add_scalar("Avg_Q", av_Q / iterations, self.iter_count)
            self.writer.add_scalar("Max_Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth", map_location=device))


###########################################
#             Training Loop
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
            # Scale action: map [-1,1] to [0,1] for linear, keep angular in [-1,1]
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

if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = GazeboEnv()
    time.sleep(4)  # wait for environment

    # TD3 config
    max_action = 1.0
    policy = TD3(action_dim=2, max_action=max_action)
    replay_buffer = PrioritizedReplayBuffer(capacity=200000, alpha=0.6)

    # Training hyperparameters
    eval_freq = 5000
    eval_episodes = 5
    max_timesteps = int(2e5)
    save_model = True
    file_name = "TD3_PER_CNN_W64x128"
    batch_size = 64
    discount = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2

    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Exploration noise parameters
    expl_noise = 1.0
    expl_decay_steps = 100000
    expl_min = 0.1

    # Variables for forced random action near obstacles:
    random_near_obstacle = True
    count_rand_actions = 0
    forced_action = np.array([0.0, 0.0])

    # Main Loop
    timestep = 0
    timesteps_since_eval = 0
    evaluations = []
    epoch = 0

    state = env.reset()  # state: (img, scalars)
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    while timestep < max_timesteps:
        if done:
            if timestep != 0:
                policy.train(
                    replay_buffer,
                    episode_timesteps,
                    batch_size,
                    discount,
                    tau,
                    policy_noise,
                    noise_clip,
                    policy_freq
                )
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

        # Decrease exploration noise
        if expl_noise > expl_min:
            expl_noise -= (1.0 - expl_min) / expl_decay_steps
            expl_noise = max(expl_noise, expl_min)

        # Select action from policy
        img, scl = state
        action = policy.get_action(img, scl)
        # Add exploration noise
        action = (action + np.random.normal(0, expl_noise, size=2)).clip(-1, 1)

        # Forced random action near obstacles:
        if random_near_obstacle:
            # scl is a vector of 7 scalars; index 6 corresponds to min_laser
            min_laser = scl[6]
            # If near an obstacle and not already forcing, with a probability, force random action
            if (count_rand_actions == 0) and (min_laser < 0.6) and (np.random.rand() < 0.3):
                count_rand_actions = np.random.randint(5, 10)
                forced_action = np.random.uniform(-1, 1, size=2)
            # If forcing random action, override the action from policy
            if count_rand_actions > 0:
                count_rand_actions -= 1
                action = forced_action

        # Scale action for environment: linear from [-1,1] -> [0,1], angular remains in [-1,1]
        a_in = [(action[0] + 1) / 2, action[1]]
        next_state, reward, done, _ = env.step(a_in)
        episode_reward += reward

        done_bool = float(done)

        # Add transition to replay buffer
        (n_img, n_scl) = next_state
        replay_buffer.add(img, scl, action, reward, done_bool, n_img, n_scl)

        state = next_state
        episode_timesteps += 1
        timestep += 1
        timesteps_since_eval += 1

    # Final evaluation
    val_rew = evaluate(policy, env, epoch, eval_episodes)
    evaluations.append(val_rew)
    policy.save(file_name, "./pytorch_models")
    np.save(f"./results/{file_name}", evaluations)
