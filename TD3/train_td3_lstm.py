import os
import time
import torch
import numpy as np
from td3_agent import TD3LSTMAgent
from replay_buffer import SequenceReplayBuffer
from realsense_env import GazeboEnv
import gym
from gym import spaces
from torch.utils.tensorboard import SummaryWriter

# --- ENV WRAPPER TO MATCH GYM FORMAT ---
class GazeboGymWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = GazeboEnv()
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(1, 64, 64), dtype=np.float32),
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]),
                                       high=np.array([1.0, 1.0]),
                                       dtype=np.float32)

    def reset(self):
        image, scalars = self.env.reset()
        return {"image": image, "scalars": scalars}

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        return {"image": next_obs[0], "scalars": next_obs[1]}, reward, done, {}

# --- TRAINING FUNCTION ---
def train_td3_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GazeboGymWrapper()
    time.sleep(4)

    image_shape = env.observation_space.spaces["image"].shape  # (1, 64, 64)
    scalar_dim = env.observation_space.spaces["scalars"].shape[0]
    action_dim = env.action_space.shape[0]
    seq_len = 10
    
    buffer = SequenceReplayBuffer(
        buffer_size=15000,
        obs_shape=image_shape,
        scalar_dim=scalar_dim,
        action_dim=action_dim,
        seq_len=seq_len,
        device=device,
    )

    agent = TD3LSTMAgent(
        obs_shape=image_shape,
        scalar_dim=scalar_dim,
        action_dim=action_dim,
        device=device,
        seq_len=seq_len
    )

    writer = SummaryWriter(log_dir="./runs/td3_lstm_gazebo_7")
    total_episodes = 200000
    warmup_episodes = 10000
    batch_size = 64

    for episode in range(total_episodes):
        image_seq, scalar_seq, action_seq, reward_seq, done_seq = [], [], [], [], []
        obs = env.reset()
        done = False
        step = 0
        
        while not done and step < 500:
            img = obs['image']
            scal = obs['scalars']
            image_seq.append(img)
            scalar_seq.append(scal)

            if episode < warmup_episodes:
                action = env.action_space.sample()
            else:
                recent_images = np.array(image_seq[-seq_len:])
                recent_scalars = np.array(scalar_seq[-seq_len:])
                if len(recent_images) < seq_len:
                    pad_len = seq_len - len(recent_images)
                    recent_images = np.pad(recent_images, ((pad_len, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
                    recent_scalars = np.pad(recent_scalars, ((pad_len, 0), (0, 0)), mode='constant')
                    print(f"Padded image_seq shape: {recent_images.shape}, scalar_seq shape: {recent_scalars.shape}")
                img_tensor = torch.tensor(recent_images, dtype=torch.float32).unsqueeze(0).to(device)
                scalar_tensor = torch.tensor(recent_scalars, dtype=torch.float32).unsqueeze(0).to(device)
                action = agent.select_action(img_tensor, scalar_tensor)
                action += np.random.normal(0, 0.1, size=action.shape)
                # print(f"Raw action: {action}")
                action[0] = np.clip(action[0], 0.0, 1.0)
                action[1] = np.clip(action[1], -1.0, 1.0)

            next_obs, reward, done, _ = env.step(action)

            if step == 499 and not done:
                reward -= 100  # punish timeouts

            action_seq.append(action)
            reward_seq.append(reward)
            done_seq.append(done)

            obs = next_obs
            step += 1

        # Save entire episode
        if len(image_seq) >= seq_len + 1:
            buffer.add_episode(
                image_seq=np.array(image_seq),
                scalar_seq=np.array(scalar_seq),
                action_seq=np.array(action_seq),
                reward_seq=np.array(reward_seq),
                done_seq=np.array(done_seq)
            )
            print(f"Buffer: {len(buffer)} transitions, {buffer.num_sequences()} sequences available")


        print(f"[Episode {episode}] Steps: {len(reward_seq)}, Buffer size: {len(buffer)}")
        losses = []
        if episode >= warmup_episodes:
            total_reward = sum(reward_seq)
            # Train for a number of steps proportional to the episode length
            losses = []
            num_updates = 5
            print(f"[Episode {episode}] Training for {num_updates} steps...")
            
            if total_reward > 0: 
                # For successful episodes, study them thoroughly
                num_updates = len(reward_seq) 
                print(f"[Episode {episode}] Successful! Training for {num_updates} steps...")

            for _ in range(num_updates):
            #    if buffer.num_sequences() > batch_size: # More robust check
                loss = agent.train(buffer, batch_size=batch_size)
                if loss:
                    losses.append(loss)

        print(f"[Episode {episode}] Total Reward: {sum(reward_seq):.2f}, Steps: {len(reward_seq)}, Done: {done}")
        print(f"[Episode {episode}] Losses: {losses}")  
        # --- Logging ---
        # print(f"\n[Train Step] Total it: {episode}, Batch: {image_seq.shape}")
        print(f"Image_seq length: {len(image_seq)}, first shape: {image_seq[0].shape}")
        print(f"Scalar_seq length: {len(scalar_seq)}, first shape: {scalar_seq[0].shape}")
        print(f"Action_seq length: {len(action_seq)}, first shape: {action_seq[0].shape}")
        print(f"Reward_seq length: {len(reward_seq)}, first value: {reward_seq[0]}")
        print(f"Done_seq length: {len(done_seq)}, first value: {done_seq[0]}")
        # print(f"Image_seq: {image_seq.shape}")        # Expect (B, T, C, H, W)
        # print(f"Scalar_seq: {scalar_seq.shape}")      # Expect (B, T, D)
        # print(f"Action_seq: {action_seq.shape}")      # (B, T, A)
        # print(f"Reward_seq: {reward_seq.shape}")      # (B, T, 1)
        # print(f"Done_seq: {done_seq.shape}")          # (B, T, 1)
        print(f"Reward min: {np.min(reward_seq):.2f}, max: {np.max(reward_seq):.2f}, mean: {np.mean(reward_seq):.2f}")

        if losses:
            
            critic1_loss = np.mean([l["critic1_loss"] for l in losses])
            critic2_loss = np.mean([l["critic2_loss"] for l in losses])
            actor_losses = [l["actor_loss"] for l in losses if l["actor_loss"] is not None]
            actor_loss = np.mean(actor_losses) if actor_losses else 0.0

            writer.add_scalar("Loss/Critic1", critic1_loss, episode)
            writer.add_scalar("Loss/Critic2", critic2_loss, episode)
            writer.add_scalar("Loss/Actor", actor_loss, episode)
            # if actor_loss is not None:
            #     print(f"Actor Loss: {actor_loss.item():.4f}")

        writer.add_scalar("Reward/Step", np.mean(reward_seq), episode)
        writer.add_scalar("Reward/Episode", sum(reward_seq), episode)
        writer.add_scalar("Length/Episode", len(reward_seq), episode)

        if episode % 100 == 0:
            model_dir = "./models"
            os.makedirs(model_dir, exist_ok=True)
            
            # Define the file path
            file_path = f"{model_dir}/td3_lstm_model_ep{episode}.zip"
            torch.save(agent.actor.state_dict(), f"td3_lstm_actor_ep{episode}.pth")

if __name__ == "__main__":
    train_td3_lstm()
