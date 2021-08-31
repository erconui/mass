from __future__ import division


import time
import random
import numpy as np
from scipy.stats import norm
import scipy.io as sio
import matplotlib as mpl
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import collections as mc
from scipy.stats import multivariate_normal as mvn
from math import atan2
from tqdm import tqdm, trange
import copy
import datetime

#Torch stuff

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pdb
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from functools import reduce
from tqdm import tqdm
import random
from importlib import reload

from simulator import Simulator
from rl_net import PolicyNetwork, ValueNetwork, AdvantageDataset, PolicyDataset, calculate_returns, calculate_advantages

def visualizeSimulation(policy, actions, algorithm_type, visualize=True, desired_steps = 50):
    nodes = [
    (0.,0.), (100.,0.), (200.,0),
    (0.,100), (100.,100.), (200.,100.),
    (0.,200), (100.,200.), (200.,200.)
    ]
    edges = [(0, 1),
             (0, 3),
             (1, 2),
             (1, 4),
             (2, 5),
             (3, 4),
             (3, 6),
             (4, 5),
             (4, 7),
             (5, 8),
             (6, 7),
             (7, 8)
             ]
    roadmap = policy.roadmap
    env = Simulator(nodes, edges, roadmap, state_resolution=10, N=500, num_targets=2)
    state = env.reset(visualize=visualize)
    actions = env.get_action_dimensions()
#     print(env.max_entropy)
    reward = []
    for ii in range(desired_steps):

#         print(ii)
        if algorithm_type == 'rl':
#             print(len(state))
            action = policy(torch.from_numpy(state).float().view(1,-1))
            action_index = np.random.choice(range(actions),p=action.detach().numpy().reshape((actions)))
#             print(action_index)
            wp=None
        elif algorithm_type == 'greedy':
            wp = env.get_agent_greedy_receeding_waypoint(lookahead=7)
            options = env.agent.get_options()
            action_index = options.index(wp)
        elif algorithm_type == 'exhaustive':
            wp = env.get_agent_exhaustive_waypoint()
            options = env.agent.get_options()
            action_index = options.index(wp)
        elif algorithm_type == 'perfect':
            wp = env.get_agent_perfect_waypoint()
            options = env.agent.get_options()
            action_index = options.index(wp)
        elif algorithm_type == 'random':
            wp = env.agent.get_random_waypoint()
            options = env.agent.get_options()
            action_index = options.index(wp)
        else:
            raise ValueError("Incorrect algorithm type")
        s_prime, rr, end_roll = env.step(action_index, wp=wp)

        reward.append(rr)
        # reward += rr

        # current_rollout.append((state, action.detach().reshape(-1), action_index, rr))
        if end_roll:
            print('illegal action: {}, {}'.format(algorithm_type, ii))
            break
        #
        state = s_prime

    return reward, ii, env.get_entropy_over_sim()

actions = 3
USE_RL_NET = True
policy = torch.load("policy-3x3-0.8-0.2-10-20")#160
entropy_list_rl = []
entropy_list_greedy = []
entropy_list_random = []
entropy_list_exhaustive = []

num_sims = 1000
steps_per_sim = 120
for i in trange(num_sims):
    reward, num_steps, sim_entropy = visualizeSimulation(
        policy, actions, 'rl', visualize=False,
        desired_steps=steps_per_sim)
    entropy_list_rl += [sim_entropy]

    reward, num_steps, sim_entropy = visualizeSimulation(
        policy, actions, 'greedy', visualize=False,
        desired_steps=steps_per_sim)
    entropy_list_greedy += [sim_entropy]

    reward, num_steps, sim_entropy = visualizeSimulation(
        policy, actions, 'random', visualize=False,
        desired_steps=steps_per_sim)
    entropy_list_random += [sim_entropy]

    reward, num_steps, sim_entropy = visualizeSimulation(
        policy, actions, 'exhaustive', visualize=False,
        desired_steps=steps_per_sim)
    entropy_list_exhaustive += [sim_entropy]
    if i % 10 == 0:
        np.save("path_planner_data_rl", entropy_list_rl)
        np.save("path_planner_data_greedy", entropy_list_greedy)
        np.save("path_planner_data_random", entropy_list_random)
        np.save("path_planner_data_exhaustive", entropy_list_exhaustive)
