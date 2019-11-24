#!/usr/bin/env python

import numpy as np
import os
import argparse
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Graph rewards')
parser.add_argument('name', nargs='?', default='recent', help='Name of directory in ./saves to find rewards')
parser.add_argument('--smooth', type=int, help='Size of window of rewards to average', default=1000)
args = parser.parse_args()

save_path = os.path.join('saves/', args.name)
rewards_path = os.path.join(save_path, 'rewards.txt')

# rewards_path = "saves/TestRefactor/rewards.txt"
rewards_path = os.path.join("saves",  args.name, "rewards.txt")

rewards = []
with open(rewards_path, 'r') as f:
    for line in f:
        rewards.append(float(line))
rewards = np.array(rewards)
rewards[rewards < 0] = 0
smooth_n = args.smooth

smoothed_rewards = []

# cumulative_rewards = [0]
# for i in range(len(rewards)):
#     cumulative_rewards.append(cumulative_rewards[i] + rewards[i])
# for i in range(len(rewards)):
#     right_index = min(i + smooth_n // 2, len(cumulative_rewards) - 1)
#     left_index = max(i - smooth_n // 2, 0)
#     smoothed_rewards.append((cumulative_rewards[right_index] - cumulative_rewards[left_index])
#                             / (right_index - left_index))

for i in range(len(rewards) - smooth_n):
    smoothed_rewards.append(np.sum(rewards[i:i+smooth_n]) / smooth_n)

print("Number of Episodes:", len(rewards))
print("Best average:", np.max(smoothed_rewards + [0]))
print("Current average:", ([0] + smoothed_rewards)[-1])
plt.plot(smoothed_rewards)
plt.savefig(os.path.join(save_path, 'reward-plot.png'))
plt.savefig(os.path.join("", 'reward-plot.png'))
