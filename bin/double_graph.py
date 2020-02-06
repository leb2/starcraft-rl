#!/usr/bin/env python

import numpy as np
import os
import argparse
import matplotlib
import sys
matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.style.use('ggplot')

parser = argparse.ArgumentParser(description='Graph rewards')
parser.add_argument('names', nargs='+', default='recent', help='Name of directory in ./saves to find rewards')
# parser.add_argument('name2', nargs='?', default='recent', help='Name of directory in ./saves to find rewards')
parser.add_argument('--smooth', type=int, help='Size of window of rewards to average', default=1000)
args = parser.parse_args()


def get_smoothed_rewards(name):
    save_path = os.path.join('saves/', name)
    rewards_path = os.path.join(save_path, 'rewards.txt')

    rewards = []
    with open(rewards_path, 'r') as f:
        for line in f:
            rewards.append(float(line))
    rewards = np.array(rewards)
    rewards[rewards < 0] = 0
    smooth_n = args.smooth
    smoothed_rewards = []

    for i in range(30, len(rewards)):
        section = rewards[max(0, i-smooth_n):i+1]
        smoothed_rewards.append(np.sum(section) / len(section))

    print("Best average:", np.max(smoothed_rewards + [0]))
    print("Current average:", ([0] + smoothed_rewards)[-1])
    return smoothed_rewards

for name in args.names:
    if "," in name:
        name, display_name = name.split(",")
    else:
        display_name = name
    smoothed_rewards = get_smoothed_rewards(name)
    plt.plot(smoothed_rewards, label=display_name)
    plt.legend()

# smoothed_rewards2 = get_smoothed_rewards(args.name2)
#
# plt.plot(smoothed_rewards1, label=args.name1)
# plt.legend()
# plt.plot(smoothed_rewards2, label=args.name2)
# plt.legend()

plt.xlabel('episodes')
plt.ylabel('reward')


# plt.savefig(os.path.join(save_path, 'reward-plot.png'))
plt.savefig(os.path.join("", 'reward-plot.png'))
