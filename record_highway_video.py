import argparse
import os
import gymnasium as gym
import highway_env
import torch

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def make_env(seed, render_mode):
    config = dict(SHARED_CORE_CONFIG)
    config["offscreen_rendering"] = False
    env = gym.make(SHARED_CORE_ENV_ID, config=config, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def to_state_tensor(obs, device):
    return torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.video_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args.seed, render_mode="rgb_array")
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = int(torch.tensor(state).numel())

    # Lazy import to avoid notebook dependency
    import torch.nn as nn
    import torch.nn.functional as F

    class DQN(nn.Module):
        def __init__(self, n_obs, n_act):
            super().__init__()
            self.layer1 = nn.Linear(n_obs, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_act)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)

    policy_net = DQN(n_observations, n_actions).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy_net.load_state_dict(ckpt["policy_state_dict"])
    policy_net.eval()

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args.video_dir,
        episode_trigger=lambda ep: ep < args.episodes,
        name_prefix=f"seed{args.seed}",
    )

    with torch.no_grad():
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            state = to_state_tensor(obs, device)
            done = False
            while not done:
                action = policy_net(state).max(1).indices.view(1, 1)
                obs, _, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated
                if not done:
                    state = to_state_tensor(obs, device)

    env.close()
    print(f"Saved videos to: {os.path.abspath(args.video_dir)}")


if __name__ == "__main__":
    main()
