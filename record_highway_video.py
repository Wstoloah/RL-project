import argparse
import os
import gymnasium as gym
import highway_env
import torch

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def make_env(seed, render_mode):
    config = dict(SHARED_CORE_CONFIG)
    config["offscreen_rendering"] = True

    env = gym.make(
        SHARED_CORE_ENV_ID,
        config=config,
        render_mode=render_mode,
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def to_state_tensor(obs, device):
    return torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.video_dir is None:
        run_dir = os.path.dirname(args.checkpoint)
        args.video_dir = os.path.join(run_dir, "videos")

    os.makedirs(args.video_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Load checkpoint
    print("Loading checkpoint:", args.checkpoint)
    ckpt = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False
)

    cfg = ckpt.get("config", {})
    hidden_size = cfg.get("hidden_size", 128)

    print("Checkpoint episode:", ckpt.get("episode"))
    print("Seed:", ckpt.get("seed"))
    print("Hidden size:", hidden_size)

    print("Checkpoint episode:", ckpt.get("episode"))
    print("Hidden size:", hidden_size)

    env = make_env(args.seed, render_mode="rgb_array")

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = int(torch.tensor(state).numel())

    # Define network matching training config
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        """3-layer MLP: obs -> hidden -> hidden -> n_actions."""

        def __init__(self, n_obs: int, n_actions: int, hidden_size: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_obs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    policy_net = Net(n_observations, n_actions, hidden_size).to(device)
    policy_net.load_state_dict(ckpt["policy_state_dict"])
    policy_net.eval()

    # Video recording
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

    print("Videos saved to:", os.path.abspath(args.video_dir))


if __name__ == "__main__":
    main()