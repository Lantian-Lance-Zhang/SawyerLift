# Custom script to evaluate model
# Not fully implemented
import argparse

import gym
from gym.wrappers import Monitor
import torch
import argparse
import numpy as np
from sac import SAC
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument('--model_path', default='',
                    help='Path to model weights for evaluation (default: no model path, random initialization)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env = Monitor(env, './videos', force=True)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
if args.model_path:
    # Load model weights, set networks to evaluation mode
    agent.load_checkpoint(args.model_path, evaluate=True)

state = env.reset()
episode_reward = 0
max_steps = 1000

for _ in tqdm(range(max_steps), ncols=100):
    action = agent.select_action(state, evaluate=True)
    next_state, reward, done, _ = env.step(action)

    state = next_state
    episode_reward += reward
    if done:
        break

env.close()
