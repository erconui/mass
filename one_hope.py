from IPython.core.debugger import set_trace

import torch
import torch.nn as nn
import pdb
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm
from importlib import reload

import simulator
reload(simulator)
from simulator import Simulator
import rl_net
reload(rl_net)
from rl_net import PolicyNetwork, ValueNetwork, AdvantageDataset, PolicyDataset, calculate_returns, calculate_advantages
import roadmap
reload(roadmap)
from roadmap import Roadmap

nodes = [
    (0.,0.), (100.,0.), (200.,0),
    (0.,100), (100.,100.), (200.,100.),
    (0.,200), (100.,200.), (200.,200.)
    ]
#
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
#

roadmap = Roadmap(nodes, edges, rotate=False)

env = Simulator(nodes, edges, roadmap, N=150, num_targets=2)
states = env.get_state_dimensions()
actions = 9 #env.get_action_dimensions()
print(states, actions)
# set_trace()
if True:
    policy = PolicyNetwork(states, actions)
    value = ValueNetwork(states)
    reward_list = []
else:
    policy = torch.load("policy-small-20")
    # roadmap = policy.roadmap
    env = Simulator(nodes, edges, roadmap, N=150, num_targets=2)
    states = env.get_state_dimensions()
    actions = env.get_action_dimensions()
    value = torch.load("value-small-20")
    reward_list = np.load("reward_list.npy", allow_pickle=False, fix_imports=False).tolist()
for row in roadmap.select_act:
    print(*row, sep='\t')
#
# print(roadmap.select_act)
policy_optim = optim.Adam(policy.parameters(), lr=1e-4, weight_decay=0.01)
value_optim = optim.Adam(value.parameters(), lr=1e-4, weight_decay=1)
epochs = 901

value_criteria = nn.MSELoss()

# Hyperparameters
env_samples = 100
episode_length = 900
gamma = 0.8
value_epochs = 5
value_batch_size = 64
policy_epochs = 20
policy_batch_size = 256
epsilon = 0.2
# standing_time_list = []
max_x_list = []
loss_list = []

loop = tqdm(total=epochs, position=0, leave=False)

for epoch in range(epochs):
    # generate rollouts
    rollouts = []
    total_reward = 0
    sim_length = 0
    for _ in range(env_samples):
        # don't forget to reset the environment at the beginning of each episode!
        # rollout for a certain number of steps!
        current_rollout = []
        state = env.reset()
        reward = 0
        end_roll = False
        current_sim_length = 0
        for ii in range(episode_length):
            action = policy(torch.from_numpy(state).float().view(1,-1))
            action_index = np.random.choice(range(actions),p=action.detach().numpy().reshape((actions)))

            # set_trace()
            # curse = list(roadmap.graph.keys()).index(tuple(env.agent.end_node))
            # node_cmd = roadmap.select_act[env.agent.end_node,action_index]
            # s_prime, rr, end_roll = env.step(node_cmd)

            s_prime, rr, end_roll = env.step(action_index)

            reward += rr
            current_sim_length += 1

            current_rollout.append((state, action.detach().reshape(-1), action_index, rr))
            if end_roll:
                break
            #
            state = s_prime
        #
        sim_length += current_sim_length
        rollouts.append(current_rollout)
        total_reward += reward
    #
    avg_sim_length = sim_length / env_samples
    avg_reward = total_reward / env_samples

    reward_list.append(avg_reward)

    calculate_returns(rollouts, gamma)

    # Approximate the value function
    value_dataset = AdvantageDataset(rollouts)
    value_loader = DataLoader(value_dataset, batch_size=value_batch_size, shuffle=True, pin_memory=True)
    for _ in range(value_epochs):
        # train value network
        total_loss = 0
        for state, returns in value_loader:
            value_optim.zero_grad()
            returns = returns.unsqueeze(1).float()
            expected_returns = value(state.float())
            loss = value_criteria(expected_returns, returns)
            total_loss += loss.item()
            loss.backward()
            value_optim.step()
        #
        loss_list.append(total_loss)
    #
    calculate_advantages(rollouts, value)

    # Learn a policy
    policy_dataset = PolicyDataset(rollouts)
    policy_loader = DataLoader(policy_dataset, batch_size=policy_batch_size, shuffle=True, pin_memory=True)
    for _ in range(policy_epochs):
        # train policy network
        for state, probs, action_index, reward, ret, advantage in policy_loader:
            policy_optim.zero_grad()
            current_batch_size = reward.size()[0]
            advantage = advantage.detach().float()#ret.float()
            p = policy(state.float())
            ratio = p[range(current_batch_size), action_index] / probs[range(current_batch_size), action_index]

            lhs = ratio*advantage
            rhs = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage
            loss = -torch.mean(torch.min(lhs, rhs))
            loss.backward()
            policy_optim.step()
        #
    #
    # loop.set_description('standing time: {}'.format(avg_standing_time))
    # if epoch % 1 == 0 and epoch != 0:
    if epoch % 10 == 0:
        torch.save(policy, 'policy-3x3-{}'.format(epoch % 100))
        torch.save(value, 'value-3x3-{}'.format(epoch % 100))
        np.save('reward_list_3x3', reward_list, allow_pickle=False, fix_imports=False)
    #
    loop.set_description('3x3 Last save: {} Reward: {:.4f}'.format(epoch % 100 - epoch%10, avg_reward.item()))
    # loop.set_description(f'Reward: {avg_reward.item():.4f}')
    loop.update(1)
    #
#
