from __future__ import division

import warnings
# warnings.filterwarnings("ignore")
# %matplotlib notebook
# %pylab inline
import matplotlib
matplotlib.use('Agg')
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
from tqdm import tqdm_notebook as tqdm
import copy
import datetime

#Torch stuff
print(matplotlib.get_backend())

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


from IPython.display import display, Markdown, Latex

def visualizeSimulation(policy, actions, use_rl, desired_steps = 50):
    nodes = [
    (0.,0.), (100.,0.), (200.,0),
    (0.,100), (100.,100.), (200.,100.),
#     (0.,200), (100.,200.), (200.,200.)
    ]
    edges = [(0, 1),
             (0, 3),
             (1, 2),
             (1, 4),
             (2, 5),
             (3, 4),
#              (3, 6),
             (4, 5),
#              (4, 7),
#              (5, 8),
#              (6, 7),
#              (7, 8)
             ]
    roadmap = policy.roadmap
    env = Simulator(nodes, edges, roadmap,N=150, num_targets=2)
    state = env.reset(visualize=True)
    actions = env.get_action_dimensions()
#     print(env.max_entropy)
    reward = []
    for ii in range(desired_steps):

#         print(ii)
        if use_rl:
            action = policy(torch.from_numpy(state).float().view(1,-1))
            action_index = np.random.choice(range(actions),p=action.detach().numpy().reshape((actions)))
#             print(action_index)
        else:
            wp = env.get_agent_greedy_waypoint()
            options = env.agent.get_options()
            action_index = options.index(wp)
        s_prime, rr, end_roll = env.step(action_index)
        reward.append(rr)
        # reward += rr

        # current_rollout.append((state, action.detach().reshape(-1), action_index, rr))
        if end_roll:
            break
        #
        state = s_prime

    return reward, ii

    from importlib import reload
    # import deep_input
    # reload(deep_input)
    # from deep_input import Simulator
    import simulator
    reload(simulator)
    from simulator import Simulator
    from rl_net import PolicyNetwork, ValueNetwork, AdvantageDataset, PolicyDataset, calculate_returns, calculate_advantages
    print(matplotlib.get_backend())
    USE_RL_NET=True

    states = 1202
    actions = 3
    if False:
        # env = gym.make('LunarLander-v2')
        policy = PolicyNetwork(states, actions)
        value = ValueNetwork(states)
        # policy_optim = optim.Adam(policy.parameters(), lr=1e-4, weight_decay=0.01)
        # value_optim = optim.Adam(value.parameters(), lr=1e-3, weight_decay=1)
    else:
        policy = torch.load("policy-80")
    #     value = torch.load("value-900")
        reward, num_steps = visualizeSimulation(
            policy, actions, USE_RL_NET, desired_steps=40)
        print(reward, num_steps, np.linalg.norm(np.array(reward)))
